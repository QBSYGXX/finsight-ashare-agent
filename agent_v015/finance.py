"""Declarative metric dependency graph over audited annual financial facts."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import json
from pathlib import Path

SCALES = {"元": Decimal(1), "千元": Decimal(1000), "万元": Decimal(10000), "亿元": Decimal(100000000)}
TRANSFORMS = {"value", "yoy_growth", "yoy_difference", "yoy_pp"}


class MetricError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}


def load_registry(path=None):
    registry = json.loads(Path(path or Path(__file__).with_name("registry.json")).read_text(encoding="utf-8"))["metrics"]
    visiting, visited = set(), set()

    def visit(key):
        if key in visiting:
            raise ValueError("Metric dependency cycle: " + key)
        if key in visited:
            return
        spec = registry[key]
        if ("raw" in spec) == ("op" in spec):
            raise ValueError("Metric must define exactly one raw field or formula")
        visiting.add(key)
        if "op" in spec:
            if spec["op"] not in {"subtract", "average", "ratio", "ratio_percent"} or len(spec["inputs"]) != 2:
                raise ValueError("Unsupported formula primitive")
            for dependency in spec["inputs"]:
                if type(dependency.get("offset", 0)) is not int or dependency.get("offset", 0) not in {0, -1}:
                    raise ValueError("Unsupported period offset")
                visit(dependency["metric"])
        visiting.remove(key)
        visited.add(key)

    for key in registry:
        visit(key)
    return registry


@dataclass
class Node:
    value: Decimal
    unit: str
    facts: dict


class MetricEngine:
    def __init__(self, facts, registry=None):
        self.registry = registry or load_registry()
        self.facts = {}
        ids = set()
        for fact in facts:
            key = (fact["company"], fact["year"], fact["scope"], fact["metric"])
            if key in self.facts or fact["id"] in ids:
                raise ValueError("Duplicate or ambiguous financial fact")
            self.facts[key] = fact
            ids.add(fact["id"])
        self.cache, self.trace = {}, []

    def available_years(self, company, scope):
        return sorted({year for (issuer, year, basis, _), fact in self.facts.items()
                       if issuer == company and basis == scope and fact.get("verified") is True})

    def _raw(self, company, year, scope, field):
        fact = self.facts.get((company, year, scope, field))
        if fact is None:
            name = next((s["name"] for s in self.registry.values() if s.get("raw") == field), field)
            raise MetricError("missing_fields", "缺少所需口径的已核验字段", {"company": company, "year": year, "scope": scope, "field": field, "field_name": name})
        required = {"id", "report_id", "report_year", "page", "pdf_sha256", "source_url", "evidence"}
        if fact.get("verified") is not True or not required <= fact.keys() or not fact["pdf_sha256"] or type(fact["page"]) is not int or fact["page"] < 1:
            raise MetricError("invalid_evidence", "事实未经核验或缺少来源信息")
        if fact["unit"] not in SCALES:
            raise MetricError("invalid_unit", "原始字段单位不受支持")
        try:
            value = Decimal(fact["value"]) * SCALES[fact["unit"]]
        except (InvalidOperation, TypeError):
            raise MetricError("invalid_value", "原始数字格式错误")
        if not value.is_finite():
            raise MetricError("invalid_value", "原始数字不是有限值")
        self.trace.append({"action": "read", "company": company, "year": year, "scope": scope, "field": field, "fact_id": fact["id"]})
        return Node(value, "money", {fact["id"]: fact})

    @staticmethod
    def check_basis(facts):
        values = list(facts.values())
        if len({(f["company"], f["scope"]) for f in values}) > 1:
            raise MetricError("basis_mismatch", "不能混用公司或报表口径")
        years = {f["year"] for f in values}
        for year in years:
            if len({f["report_id"] for f in values if f["year"] == year}) != 1:
                raise MetricError("basis_mismatch", "同年依赖项不属于同一报告快照")
        if len(years) > 1:
            newest = max(years)
            if years != {newest, newest - 1}:
                raise MetricError("basis_mismatch", "跨期计算仅支持相邻年度")
            current = {f["metric"]: f for f in values if f["year"] == newest}
            for old in (f for f in values if f["year"] < newest):
                comparison = current.get(old["metric"], {}).get("comparison", {})
                try:
                    comparable = comparison.get("verified") is True and comparison["year"] == old["year"] and (
                        Decimal(comparison["value"]) * SCALES[comparison["unit"]] == Decimal(old["value"]) * SCALES[old["unit"]])
                except (KeyError, InvalidOperation, TypeError):
                    comparable = False
                if not comparable:
                    raise MetricError("basis_mismatch", "上年数与本年报告已核验比较列不一致，或缺少比较列核验", {"fact_id": old["id"]})

    def compute(self, company, year, scope, metric):
        key = (company, year, scope, metric)
        if key in self.cache:
            return self.cache[key]
        if metric not in self.registry:
            raise MetricError("unsupported_metric", "指标库未定义此指标", {"metric": metric})
        spec = self.registry[metric]
        if "raw" in spec:
            node = self._raw(company, year, scope, spec["raw"])
        else:
            inputs, missing = [], []
            for item in spec["inputs"]:
                try:
                    inputs.append(self.compute(company, year + item.get("offset", 0), scope, item["metric"]))
                except MetricError as error:
                    if error.code != "missing_fields":
                        raise
                    missing.append(error.details)
            if missing:
                raise MetricError("missing_fields", "缺少计算依赖，未用其他字段替代", {"dependencies": missing})
            left, right = inputs
            facts = {**left.facts, **right.facts}
            self.check_basis(facts)
            for item, node_in in zip(spec["inputs"], inputs):
                if item["metric"] in spec.get("nonnegative", []) and node_in.value < 0:
                    raise MetricError("invalid_domain", "公式要求该输入为非负", {"metric": item["metric"]})
            if left.unit != right.unit:
                raise MetricError("invalid_unit", "公式两项量纲不一致")
            operation = spec["op"]
            if operation == "subtract":
                value = left.value - right.value
            elif operation == "average":
                value = (left.value + right.value) / 2
            else:
                if right.value <= 0:
                    raise MetricError("invalid_denominator", "分母非正，当前指标定义不适用")
                value = left.value / right.value * (100 if operation == "ratio_percent" else 1)
            node = Node(value, spec["unit"], facts)
            self.trace.append({"action": "formula", "metric": metric, "year": year, "formula": spec["formula"], "value": str(value), "unit": node.unit})
        self.cache[key] = node
        return node

    def execute(self, company, year, scope, requests, money_unit="亿元", *, _reset=True):
        if type(year) is not int or scope not in {"consolidated", "parent"} or money_unit not in SCALES:
            raise ValueError("Invalid year, scope or money unit")
        if not isinstance(requests, list) or not 1 <= len(requests) <= 24:
            raise ValueError("Expected 1..24 metric requests")
        if _reset:
            self.cache, self.trace = {}, []
        results = []
        for request in requests:
            metric, transform = request.get("metric"), request.get("transform", "value")
            spec = self.registry.get(metric, {})
            row = {"metric": metric, "name": spec.get("name", metric), "transform": transform,
                   "company": company, "year": year, "scope": scope, "period": "annual"}
            try:
                if transform not in TRANSFORMS:
                    raise MetricError("unsupported_transform", "不支持该计算方式")
                with localcontext() as context:
                    context.prec = 40
                    node = self.compute(company, year, scope, metric)
                    value, unit, facts = node.value, node.unit, node.facts
                    formula = spec.get("formula", spec.get("name", ""))
                    if transform != "value":
                        if transform == "yoy_pp" and unit != "%":
                            raise MetricError("invalid_transform", "百分点变化仅适用于百分比指标")
                        if transform == "yoy_difference" and unit == "%":
                            raise MetricError("needs_clarification", "百分比指标的差额请明确为百分点变化")
                        previous = self.compute(company, year - 1, scope, metric)
                        facts = {**facts, **previous.facts}
                        self.check_basis(facts)
                        if transform == "yoy_growth":
                            if previous.value <= 0:
                                raise MetricError("invalid_denominator", "上年值非正，不计算通常意义的同比增长率")
                            value, unit = (value / previous.value - 1) * 100, "%"
                            formula = "(本年指标 / 上年指标 - 1) * 100；基础定义：" + formula
                        else:
                            value = value - previous.value
                            unit = "百分点" if transform == "yoy_pp" else unit
                            formula = "本年指标 - 上年指标；基础定义：" + formula
                    if unit == "money":
                        value, unit = value / SCALES[money_unit], money_unit
                    sources = [{k: f[k] for k in ("id", "company", "year", "metric", "value", "unit", "scope", "report_id", "page", "pdf_sha256", "source_url", "evidence")}
                               for f in facts.values()]
                    row.update(status="ok", value=format(value.quantize(Decimal("0.0001")), "f"), unit=unit,
                               formula=formula, citations=list(facts), sources=sources)
            except MetricError as error:
                row.update(status=error.code, value=None, unit=None, message=str(error), details=error.details, citations=[])
            results.append(row)
        ok = sum(r["status"] == "ok" for r in results)
        return {"status": "completed" if ok == len(results) else "partial" if ok else "no_results",
                "company": company, "year": year, "scope": scope, "results": results,
                "execution_trace": self.trace, "raw_reads": sum(t["action"] == "read" for t in self.trace)}

    def execute_years(self, company, years, scope, requests, money_unit="亿元"):
        if not isinstance(years, list) or not years or any(type(y) is not int for y in years) or len(set(years)) != len(years):
            raise MetricError("invalid_plan", "查询年份必须是非空且无重复的整数列表")
        if len(years) > 12 or len(years) * len(requests) > 120:
            raise MetricError("request_too_large", "一次最多12个年度、120个结果，请缩小范围")
        self.cache, self.trace = {}, []
        results = []
        years = sorted(years)
        for year in years:
            result = self.execute(company, year, scope, requests, money_unit, _reset=False)
            results.extend(result["results"])
        ok = sum(row["status"] == "ok" for row in results)
        return {"status": "completed" if ok == len(results) else "partial" if ok else "no_results",
                "company": company, "years": years, "scope": scope, "results": results,
                "execution_trace": self.trace, "raw_reads": sum(t["action"] == "read" for t in self.trace)}
