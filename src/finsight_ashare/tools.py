"""Audited facts and deterministic tools; financial arithmetic never uses eval."""
from decimal import Decimal, InvalidOperation
import json
import inspect
import re

from .common import ROOT, read_json, read_jsonl

METRICS = {
    "revenue": "营业收入", "net_profit": "归属于上市公司股东的净利润",
    "operating_cashflow": "经营活动产生的现金流量净额", "assets": "资产总计",
    "liabilities": "负债合计", "cost": "营业成本",
}
UNITS = {"元": Decimal(1), "千元": Decimal(1000), "万元": Decimal(10000), "亿元": Decimal(100000000)}


# 模块：财报工具；作用：检索已核验事实并保留来源、口径；核心业务：是。
class FinancialTools:
    def __init__(self, facts, pages=None, allowed_companies=None, search_backend="lexical", root=ROOT):
        self.facts = {f["id"]: f for f in facts}
        if len(self.facts) != len(facts):
            raise ValueError("Duplicate fact IDs")
        self.allowed = set(allowed_companies) if allowed_companies is not None else None
        self.pages = pages or []
        self.search_backend = search_backend
        self.root = root
        self.encoder = None
        self.embeddings = {}

    @classmethod
    def load(cls, root=ROOT, **kwargs):
        facts_path = root / "data/verified/facts.json"
        if not facts_path.exists():
            raise ValueError("No audited facts. Run prepare --stage ingest, review audit.csv, then prepare --stage audit.")
        pages_path = root / "data/processed/pages.jsonl"
        return cls(read_json(facts_path), read_jsonl(pages_path) if pages_path.exists() else [], root=root, **kwargs)

    def _permitted(self, company):
        if self.allowed is not None and company not in self.allowed:
            raise ValueError("Company is outside this split")

    def _fact(self, fact_id):
        fact = self.facts.get(fact_id)
        if fact is None:
            raise ValueError("Unknown fact_id: " + str(fact_id))
        self._permitted(fact["company"])
        if fact.get("verified") is not True:
            raise ValueError("Unverified fact")
        return fact

    def table_read(self, company, year, metric, scope="consolidated"):
        self._permitted(company)
        if metric not in METRICS or type(year) is not int:
            raise ValueError("Invalid metric or year")
        found = [f for f in self.facts.values() if (f["company"], f["year"], f["metric"], f["scope"]) == (company, year, metric, scope)]
        if len(found) > 1:
            raise ValueError("Ambiguous restatement: choose one audited reporting basis first")
        return {"status": "ok" if found else "missing", "facts": [self._fact(f["id"]) for f in found]}

    def report_search(self, query, company, year=None, top_k=3):
        self._permitted(company)
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("Invalid search query")
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise ValueError("top_k must be 1..5")
        candidates = []
        for f in self.facts.values():
            if f["company"] == company and (year is None or f["year"] == year):
                candidates.append({"kind": "audited_fact", "fact": f, "text": f"{f['year']} {METRICS[f['metric']]} {f['value']} {f['unit']} {f['evidence']}"})
        for page in self.pages:
            if page["company"] == company and (year is None or page["report_year"] == year):
                for start in range(0, len(page["text"]), 380):
                    candidates.append({"kind": "raw_excerpt_not_audited", "report_id": page["report_id"], "page": page["page"], "text": page["text"][start:start+450]})
        if not candidates:
            return {"status": "missing", "hits": []}
        if self.search_backend == "bge":
            import numpy as np
            from sentence_transformers import SentenceTransformer
            if self.encoder is None:
                self.encoder = SentenceTransformer(str(self.root / "models/bge-m3"), device="cpu", local_files_only=True)
                self.encoder.max_seq_length = 512
            cache_key = (company, year)
            if cache_key not in self.embeddings:
                self.embeddings[cache_key] = self.encoder.encode([c["text"] for c in candidates], normalize_embeddings=True, show_progress_bar=False)
            query_vector = self.encoder.encode([query], normalize_embeddings=True)[0]
            scores = np.asarray(self.embeddings[cache_key]) @ query_vector
        else:
            tokens = set(re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]{2}", query))
            scores = [sum(token in c["text"] for token in tokens) for c in candidates]
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))[:top_k]
        return {"status": "ok", "hits": [candidates[i] for i in order]}

    def calculate(self, operation, fact_ids, unit="亿元"):
        if not isinstance(fact_ids, list) or not 1 <= len(fact_ids) <= 2:
            raise ValueError("Supply one or two fact_ids")
        facts = [self._fact(i) for i in fact_ids]
        if len({(f["company"], f["scope"]) for f in facts}) != 1:
            raise ValueError("Mixed company or statement scope")
        values = [Decimal(f["value"]) * UNITS[f["unit"]] for f in facts]
        if operation == "convert" and len(values) == 1:
            if unit not in UNITS:
                raise ValueError("Unsupported unit")
            result = values[0] / UNITS[unit]
        elif operation in {"growth", "difference"} and len(values) == 2:
            if facts[0]["metric"] != facts[1]["metric"] or facts[0]["year"] != facts[1]["year"] + 1:
                raise ValueError("Growth/difference requires the same metric, current year then previous year")
            if facts[0]["report_id"] != facts[1]["report_id"]:
                comparison = facts[0].get("comparison")
                if not comparison or comparison.get("verified") is not True or comparison["year"] != facts[1]["year"]:
                    raise ValueError("Cross-report comparison requires an audited prior-year column")
                prior = Decimal(comparison["value"]) * UNITS[comparison["unit"]]
                if prior != values[1]:
                    raise ValueError("Reporting basis mismatch: prior comparative differs from the older report (restatement or rounding)")
            if operation == "growth":
                if values[1] <= 0:
                    raise ValueError("Non-positive prior value: percentage growth is not meaningful")
                result, unit = (values[0] / values[1] - 1) * 100, "%"
            else:
                if unit not in UNITS:
                    raise ValueError("Unsupported unit")
                result = (values[0] - values[1]) / UNITS[unit]
        elif operation in {"ratio", "gross_margin"} and len(values) == 2:
            if facts[0]["year"] != facts[1]["year"]:
                raise ValueError("Ratio inputs must use the same fiscal year")
            if facts[0]["report_id"] != facts[1]["report_id"]:
                raise ValueError("Ratio inputs must use the same report snapshot")
            if operation == "gross_margin":
                if [f["metric"] for f in facts] != ["revenue", "cost"] or values[0] <= 0 or values[1] < 0:
                    raise ValueError("Gross margin requires positive revenue followed by cost")
                result = (values[0] - values[1]) / values[0] * 100
            else:
                if values[1] <= 0:
                    raise ValueError("Ratio denominator must be positive")
                result = values[0] / values[1] * 100
            unit = "%"
        else:
            raise ValueError("Unsupported operation/operand count")
        return {"status": "ok", "value": format(result.quantize(Decimal("0.0001")), "f"), "unit": unit,
                "company": facts[0]["company"], "year": facts[0]["year"], "scope": facts[0]["scope"],
                "operation": operation, "citations": [f["id"] for f in facts],
                "sources": [{k: f[k] for k in ("id", "report_id", "report_year", "page", "source_url", "pdf_sha256")} for f in facts]}

    def call(self, tool_name, **kwargs):
        handlers = {"report_search": self.report_search, "table_read": self.table_read, "calculate": self.calculate}
        if tool_name not in handlers:
            raise ValueError("Tool is not allowed")
        handler = handlers[tool_name]
        signature = inspect.signature(handler)
        usage = f"{tool_name}{signature}"
        unknown = sorted(set(kwargs) - set(signature.parameters))
        if unknown:
            raise ValueError(f"Invalid tool arguments: unsupported keywords {unknown}. Expected: {usage}. "
                             "请将各参数直接写为call_tool的关键字参数，不要包装成字典字段。")
        try:
            signature.bind(**kwargs)
        except TypeError as error:
            raise ValueError(f"Invalid tool arguments: {error}. Expected: {usage}") from error
        try:
            return handler(**kwargs)
        except (TypeError, KeyError, InvalidOperation) as error:
            raise ValueError("Invalid tool arguments: " + str(error)) from error
