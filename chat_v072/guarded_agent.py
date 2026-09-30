"""Chat-only tool composition and answer validation; frozen evaluation is unchanged."""
from decimal import Decimal, InvalidOperation
import inspect
import re

from finsight_ashare.tools import METRICS
from resolved_agent import ResolvedAgent


class ScopedTools:
    def __init__(self, delegate, company):
        self.delegate, self.company = delegate, company
        self.allowed = {company}
        self.receipts, self.missing, self.calls = [], [], []

    def call(self, tool_name, **kwargs):
        if tool_name in {"table_read", "report_search"} and kwargs.get("company") != self.company:
            raise ValueError("COMPANY_MISMATCH: 本题公司固定为" + self.company + "，不得改查其他公司。这是参数错误，不是缺数据。")
        if tool_name == "calculate":
            ids = kwargs.get("fact_ids")
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                raise ValueError("INVALID_ARGUMENTS: fact_ids must be a list of strings")
            for fact_id in ids:
                fact = self.delegate.facts.get(fact_id)
                if fact is None or fact["company"] != self.company:
                    raise ValueError("COMPANY_MISMATCH: calculation facts must belong to " + self.company)
        if tool_name == "table_read" and kwargs.get("metric") == "gross_margin":
            try:
                inspect.signature(self.delegate.table_read).bind(**kwargs)
            except TypeError as error:
                raise ValueError("INVALID_ARGUMENTS: " + str(error)) from error
            company, year = kwargs["company"], kwargs["year"]
            scope = kwargs.get("scope", "consolidated")
            reads = [self.delegate.call("table_read", company=company, year=year, metric=metric, scope=scope)
                     for metric in ("revenue", "cost")]
            facts = [f for read in reads for f in read["facts"]]
            subcalls = [{"tool_name": "table_read", "company": company, "year": year,
                         "metric": metric, "scope": scope, "observation": read}
                        for metric, read in zip(("revenue", "cost"), reads)]
            if any(read["status"] != "ok" for read in reads):
                result = {"status": "missing", "company": company, "year": year, "scope": scope,
                          "metric": "gross_margin", "facts": facts,
                          "missing_metrics": [m for m, r in zip(("revenue", "cost"), reads) if r["status"] != "ok"],
                          "subcalls": subcalls}
            else:
                ids = [f["id"] for f in facts]
                calculated = self.delegate.call("calculate", operation="gross_margin", fact_ids=ids, unit="%")
                subcalls.append({"tool_name": "calculate", "operation": "gross_margin", "fact_ids": ids,
                                 "unit": "%", "observation": calculated})
                result = {**calculated, "metric": "gross_margin", "facts": facts, "subcalls": subcalls,
                          "derivation": "(revenue - cost) / revenue * 100; computed, not a raw reported field"}
        else:
            if tool_name == "table_read" and kwargs.get("metric") not in METRICS:
                raise ValueError("INVALID_METRIC: allowed=" + ",".join([*METRICS, "gross_margin"]) +
                                 "; 毛利率可用metric='gross_margin'，由工具读取收入和成本后计算。不得把参数错误解释为数据缺失。")
            result = self.delegate.call(tool_name, **kwargs)
        self.calls.append({"tool_name": tool_name, "arguments": kwargs, "observation": result})
        if result.get("status") == "missing":
            self.missing.append({**kwargs, "observation": result})
        if result.get("status") == "ok" and "value" in result:
            self.receipts.append(result)
        for fact in result.get("facts", []):
            self.receipts.append({**fact, "citations": [fact["id"]], "operation": "raw_fact"})
        return result

    def validate_final(self, final, seen, question):
        if final["company"] != self.company:
            return "FINAL_COMPANY_MISMATCH"
        years = {int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)}
        if len(years) == 1 and final["year"] not in years:
            return "FINAL_YEAR_MISMATCH"
        if any(word in question for word in ("母公司", "单体")) and final["scope"] != "parent":
            return "FINAL_SCOPE_MISMATCH"
        if "合并" in question and "母公司" not in question and final["scope"] != "consolidated":
            return "FINAL_SCOPE_MISMATCH"
        gross_margin = "毛利率" in question or "gross margin" in question.lower()
        matching = [r for r in self.receipts if all(r[k] == final[k] for k in ("company", "year", "scope"))
                    and (not gross_margin or r.get("operation") == "gross_margin")]
        if final["status"] == "insufficient_evidence":
            if matching:
                return "FALSE_MISSING: successful tool evidence exists"
            missing = [r for r in self.missing if r.get("year") == final["year"]
                       and r.get("scope", "consolidated") == final["scope"]
                       and (not gross_margin or r.get("metric") in {"gross_margin", "revenue", "cost"})]
            if not missing:
                return "UNSUPPORTED_MISSING: no successful lookup established missing facts; tool errors are not missing data"
            return None
        for receipt in matching:
            try:
                same_value = abs(Decimal(str(final["value"])) - Decimal(str(receipt["value"]))) <= Decimal("0.00015")
            except (InvalidOperation, TypeError):
                same_value = False
            if (same_value and final["unit"] == receipt["unit"]
                    and set(final["citations"]) == set(receipt["citations"])
                    and set(final["citations"]) <= set(seen)):
                return None
        return "UNSUPPORTED_VALUE: final number/unit/citations do not match a successful tool result"


class ContractModel:
    def __init__(self, model, company):
        self.model, self.company = model, company
        self.system_prompt = None

    def generate(self, messages):
        history = [dict(m) for m in messages]
        history[0]["content"] += (
            "\n本题公司已确定为" + self.company + "，所有工具调用和答案必须保持该代码。"
            "\n本入口新增派生指标：table_read(metric='gross_margin', company=本题代码, year=所问年份, scope=所问口径)"
            "会读取已核验营业收入与营业成本，并通过calculate计算毛利率，返回数值、原始事实和引用。"
            "毛利率问题可直接调用该工具，无需先检索PDF。不得虚构gross_margin事实ID，引用返回的收入和成本事实ID。"
            "\nINVALID_METRIC、COMPANY_MISMATCH及其他动作失败均是调用错误，不是数据缺失。"
            "只有成功读表明确返回missing时才可声明当前事实库缺数据；算得结果后不得再称缺数据。"
        )
        self.system_prompt = history[0]["content"]
        return self.model.generate(history)


class GuardedAgent:
    def __init__(self, model, tools, resolver, max_steps=8):
        self.model, self.tools, self.resolver, self.max_steps = model, tools, resolver, max_steps

    def run(self, question):
        resolution = self.resolver.resolve(question)
        if resolution["status"] != "resolved":
            return {"question": question, "status": resolution["status"], "resolution": resolution,
                    "final": None, "events": [], "messages": [], "seen_fact_ids": [], "seconds": 0}
        company = resolution["company"]
        if self.tools.allowed is not None and company not in self.tools.allowed:
            return {"question": question, "status": "company_outside_evaluation_scope", "resolution": resolution,
                    "final": None, "events": [], "messages": [], "seen_fact_ids": [], "seconds": 0}
        tools = ScopedTools(self.tools, company)
        model = ContractModel(self.model, company)
        trace = ResolvedAgent(model, tools, self.resolver, self.max_steps).run(question)
        trace["tool_audit"] = tools.calls
        trace["protocol"] = "chat-v0.7.2"
        trace["effective_system_prompt"] = model.system_prompt
        if trace.get("final") is not None:
            error = tools.validate_final(trace["final"], trace["seen_fact_ids"], question)
            trace["answer_validation"] = {"passed": error is None, "error": error}
            if error:
                trace["rejected_final"] = trace["final"]
                trace["final"] = None
                trace["status"] = "answer_validation_failed"
        return trace
