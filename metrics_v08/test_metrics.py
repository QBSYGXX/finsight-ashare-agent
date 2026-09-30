"""Numerical, provenance and plan-boundary tests; no GPU or test-company queries."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "evaluation_v07"
if not REFERENCE.exists():
    REFERENCE = ROOT.parent / "_release_work/finsight-evaluation-v0.7/evaluation_v07"
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "src"), str(REFERENCE), str(ROOT / "chat_v071")]
from engine import MetricEngine, MetricError, load_registry
from planner import model_plan, request_context, rules_plan
from resolver import CompanyResolver


def fixtures():
    annual = {2023: {"revenue": 800, "cost": 520, "net_profit": 100, "assets": 1600, "liabilities": 640, "operating_cashflow": 200},
              2024: {"revenue": 1000, "cost": 600, "net_profit": 200, "assets": 2000, "liabilities": 800, "operating_cashflow": 300}}
    facts = []
    for year, metrics in annual.items():
        for metric, value in metrics.items():
            fact = {"id": f"600519-{year}-{metric}", "company": "600519", "year": year, "scope": "consolidated",
                    "metric": metric, "value": str(value), "unit": "元", "verified": True,
                    "report_id": f"600519-{year}", "report_year": year, "page": 63,
                    "pdf_sha256": "fixture-sha", "source_url": "https://example.invalid/report", "evidence": f"{metric}: {value}"}
            if year == 2024:
                fact["comparison"] = {"year": 2023, "value": str(annual[2023][metric]), "unit": "元", "verified": True}
            facts.append(fact)
    return facts


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.facts = fixtures()
        self.engine = MetricEngine(self.facts)

    def query(self, metric, transform="value", scope="consolidated", year=2024):
        return self.engine.execute("600519", year, scope, [{"metric": metric, "transform": transform}], "元")["results"][0]

    def test_multiple_metrics_partial_and_shared_reads(self):
        result = self.engine.execute("600519", 2024, "consolidated", [{"metric": m} for m in ("gross_margin", "net_margin", "debt_ratio", "parent_profit_margin")])
        self.assertEqual([r["status"] for r in result["results"]], ["ok", "missing_fields", "ok", "ok"])
        self.assertEqual([r["value"] for r in result["results"]], ["40.0000", None, "40.0000", "20.0000"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["raw_reads"], 5)
        self.assertIn("total_net_profit", json.dumps(result["results"][1]))

    def test_net_profit_is_not_parent_profit(self):
        self.assertEqual(self.query("net_margin")["status"], "missing_fields")
        extra = {**deepcopy(self.facts[-1]), "id": "total-profit", "metric": "total_net_profit", "value": "250"}
        self.engine = MetricEngine([*self.facts, extra])
        self.assertEqual(self.query("net_margin")["value"], "25.0000")
        self.assertEqual(self.query("parent_profit_margin")["value"], "20.0000")

    def test_growth_and_pp_are_distinct(self):
        self.assertEqual(self.query("gross_margin", "yoy_growth")["value"], "14.2857")
        pp = self.query("gross_margin", "yoy_pp")
        self.assertEqual((pp["value"], pp["unit"]), ("5.0000", "百分点"))
        self.assertEqual(len(pp["citations"]), 4)

    def test_amount_difference_and_units(self):
        row = self.query("revenue", "yoy_difference")
        self.assertEqual((row["value"], row["unit"]), ("200.0000", "元"))
        next(f for f in self.facts if f["year"] == 2024 and f["metric"] == "cost").update(value="0.6", unit="千元")
        self.assertEqual(self.query("gross_margin")["value"], "40.0000")

    def test_assets_average_and_equity_graph(self):
        self.assertEqual(self.query("asset_turnover")["value"], "0.5556")
        self.assertEqual(self.query("equity_ratio")["value"], "60.0000")
        self.assertEqual(self.query("cash_parent_profit_ratio")["value"], "150.0000")

    def test_missing_parent_scope(self):
        self.assertEqual(self.query("gross_margin", scope="parent")["status"], "missing_fields")

    def test_wrong_report_basis(self):
        next(f for f in self.facts if f["year"] == 2024 and f["metric"] == "cost")["report_id"] = "other"
        self.assertEqual(self.query("gross_margin")["status"], "basis_mismatch")

    def test_restatement_prevents_yoy(self):
        next(f for f in self.facts if f["year"] == 2024 and f["metric"] == "revenue")["comparison"]["value"] = "801"
        self.assertEqual(self.query("gross_margin", "yoy_pp")["status"], "basis_mismatch")
        self.assertEqual(self.query("gross_margin")["status"], "ok")

    def test_comparison_is_required(self):
        next(f for f in self.facts if f["year"] == 2024 and f["metric"] == "assets").pop("comparison")
        self.assertEqual(self.query("asset_turnover")["status"], "basis_mismatch")

    def test_zero_denominator(self):
        next(f for f in self.facts if f["year"] == 2024 and f["metric"] == "revenue")["value"] = "0"
        self.assertEqual(self.query("gross_margin")["status"], "invalid_denominator")

    def test_negative_prior_no_percentage_growth(self):
        next(f for f in self.facts if f["year"] == 2023 and f["metric"] == "net_profit")["value"] = "-100"
        next(f for f in self.facts if f["year"] == 2024 and f["metric"] == "net_profit")["comparison"]["value"] = "-100"
        self.assertEqual(self.query("parent_net_profit", "yoy_growth")["status"], "invalid_denominator")

    def test_source_required(self):
        next(f for f in self.facts if f["year"] == 2024 and f["metric"] == "cost").pop("pdf_sha256")
        self.assertEqual(self.query("gross_margin")["status"], "invalid_evidence")

    def test_invalid_transform_and_unknown_metric(self):
        self.assertEqual(self.query("revenue", "yoy_pp")["status"], "invalid_transform")
        self.assertEqual(self.query("gross_margin", "yoy_difference")["status"], "needs_clarification")
        self.assertEqual(self.query("invented")["status"], "unsupported_metric")

    def test_duplicate_facts_fail(self):
        with self.assertRaises(ValueError):
            MetricEngine([*self.facts, self.facts[0]])

    def test_new_formula_requires_no_engine_change(self):
        registry = load_registry()
        registry["cost_ratio"] = {"name": "成本率", "aliases": ["成本率"], "unit": "%", "op": "ratio_percent",
                                  "inputs": [{"metric": "cost"}, {"metric": "revenue"}], "formula": "成本 / 收入 * 100"}
        self.engine = MetricEngine(self.facts, registry)
        self.assertEqual(self.query("cost_ratio")["value"], "60.0000")

    def test_registry_cycle_rejected(self):
        registry = load_registry()
        registry["gross_profit"]["inputs"][0]["metric"] = "gross_margin"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps({"metrics": registry}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cycle"):
                load_registry(path)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry()
        self.resolver = CompanyResolver([{"code": "600519", "name": "贵州茅台"}], {"600519": ["茅台"]})

    def plan(self, question):
        return rules_plan(request_context(question, self.resolver), self.registry)

    def test_multi_metric_plain_question(self):
        plan = self.plan("贵州茅台2024年毛利率、净利率和资产负债率")
        self.assertEqual([r["metric"] for r in plan["requests"]], ["gross_margin", "net_margin", "debt_ratio"])

    def test_longest_alias_prevents_parent_confusion(self):
        plan = self.plan("茅台2024年归母净利润及其同比增长率")
        self.assertEqual(plan["requests"], [{"metric": "parent_net_profit", "transform": t} for t in ("value", "yoy_growth")])

    def test_metric_values_and_shared_yoy(self):
        plan = self.plan("茅台2024年毛利率、资产负债率以及它们的同比增长率")
        self.assertEqual(len(plan["requests"]), 4)

    def test_each_metric_keeps_its_own_transform(self):
        plan = self.plan("茅台2024年营业收入及其同比增长率、毛利率")
        self.assertEqual(plan["requests"], [{"metric": "revenue", "transform": "value"},
                                           {"metric": "revenue", "transform": "yoy_growth"},
                                           {"metric": "gross_margin", "transform": "value"}])

    def test_rate_ambiguity(self):
        with self.assertRaisesRegex(MetricError, "明确"):
            self.plan("茅台2024年毛利率同比增幅")
        self.assertEqual(self.plan("茅台2024年毛利率同比变化多少个百分点")["requests"][0]["transform"], "yoy_pp")

    def test_unknown_compound_not_silently_dropped(self):
        with self.assertRaises(MetricError):
            self.plan("茅台2024年毛利率和ROIC")

    def test_subannual_never_planned(self):
        for period in ("2024H1", "2024Q1", "2024年一季报", "2024年6月"):
            with self.subTest(period=period), self.assertRaises(MetricError):
                self.plan("茅台" + period + "毛利率")

    def test_company_conflict(self):
        with self.assertRaises(MetricError):
            self.plan("贵州茅台600418在2024年毛利率")

    def test_explicit_period_and_scope(self):
        with self.assertRaises(MetricError):
            self.plan("贵州茅台毛利率")
        p = self.plan("贵州茅台2024年母公司营业收入多少万元")
        self.assertEqual((p["scope"], p["money_unit"]), ("parent", "万元"))

    def model(self, output, question="贵州茅台2024年毛利率、资产负债率"):
        class Model:
            def generate(self, messages):
                return json.dumps(output)
        return model_plan(request_context(question, self.resolver), self.registry, Model())

    def test_model_plan_is_validated(self):
        plan = self.model({"requests": [{"metric": m, "transform": "value"} for m in ("gross_margin", "debt_ratio")], "clarification": None})
        self.assertEqual(plan["company"], "600519")

    def test_model_cannot_supply_formula_or_answer(self):
        with self.assertRaises(MetricError):
            self.model({"requests": [{"metric": "gross_margin", "transform": "value", "value": 99}], "clarification": None})

    def test_model_cannot_drop_requested_metric(self):
        with self.assertRaises(MetricError):
            self.model({"requests": [{"metric": "gross_margin", "transform": "value"}], "clarification": None})

    def test_model_cannot_invent_yoy(self):
        with self.assertRaises(MetricError):
            self.model({"requests": [{"metric": m, "transform": "yoy_growth"} for m in ("gross_margin", "debt_ratio")], "clarification": None})


class MultiYearTests(unittest.TestCase):
    def setUp(self):
        self.engine = MetricEngine(fixtures())
        self.resolver = CompanyResolver([{"code": "600519", "name": "贵州茅台"}], {"600519": ["茅台"]})

    def context(self, question):
        return request_context(question, self.resolver, self.engine.available_years)

    def run_question(self, question):
        plan = rules_plan(self.context(question), self.engine.registry)
        return self.engine.execute_years(plan["company"], plan["years"], plan["scope"], plan["requests"], plan["money_unit"])

    def test_exact_user_question_uses_actual_coverage(self):
        question = "告诉我你能查询到的所有年份贵州茅台的毛利率"
        result = self.run_question(question)
        self.assertEqual(result["years"], [2023, 2024])
        self.assertEqual([r["value"] for r in result["results"]], ["35.0000", "40.0000"])
        self.assertEqual(result["raw_reads"], 4)

    def test_all_year_aliases(self):
        for phrase in ("全部年份", "所有年度", "历年", "各年", "所有可用年份"):
            with self.subTest(phrase=phrase):
                self.assertEqual(self.run_question("贵州茅台" + phrase + "毛利率")["years"], [2023, 2024])

    def test_range_and_list_have_distinct_meanings(self):
        for separator in ("-", "至", "到", "～", "—"):
            with self.subTest(separator=separator):
                result = self.run_question("茅台2023" + separator + "2025年毛利率")
                self.assertEqual(result["years"], [2023, 2024, 2025])
                self.assertEqual([r["status"] for r in result["results"]], ["ok", "ok", "missing_fields"])
        self.assertEqual(self.run_question("茅台2023、2025年毛利率")["years"], [2023, 2025])

    def test_absent_year_retained(self):
        result = self.run_question("茅台2022至2024年毛利率")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["results"][0]["year"], 2022)
        self.assertEqual(result["results"][0]["status"], "missing_fields")
        self.assertEqual(len(result["results"]), 3)

    def test_scope_company_and_verification_filter_coverage(self):
        extra = {**deepcopy(fixtures()[0]), "company": "000333", "year": 2025, "id": "other-company"}
        unverified = {**deepcopy(fixtures()[0]), "year": 2026, "id": "unverified", "verified": False}
        self.engine = MetricEngine([*fixtures(), extra, unverified])
        self.assertEqual(self.engine.available_years("600519", "consolidated"), [2023, 2024])
        with self.assertRaisesRegex(MetricError, "没有已核验"):
            self.run_question("贵州茅台历年母公司毛利率")

    def test_missing_metric_not_dropped_from_all_years(self):
        result = self.run_question("贵州茅台历年毛利率、净利率")
        self.assertEqual(len(result["results"]), 4)
        self.assertEqual([r["status"] for r in result["results"]], ["ok", "missing_fields", "ok", "missing_fields"])

    def test_yoy_per_year_preserves_first_year_missing(self):
        result = self.run_question("贵州茅台历年毛利率同比增长率")
        self.assertEqual(result["results"][0]["status"], "missing_fields")
        self.assertEqual(result["results"][1]["value"], "14.2857")
        self.assertEqual(result["raw_reads"], 4)

    def test_dates_are_not_annual_ranges(self):
        for phrase in ("2024-06-30", "2023至2025年第一季度", "2024H1"):
            with self.subTest(phrase=phrase), self.assertRaises(MetricError):
                self.run_question("贵州茅台" + phrase + "毛利率")

    def test_ambiguous_mixed_and_oversized_requests(self):
        for question in ("贵州茅台2024年所有年份毛利率", "贵州茅台2025至2023年毛利率",
                         "贵州茅台2000至2025年毛利率", "贵州茅台2023年毛利率、2024年资产负债率",
                         "贵州茅台2025年比2023年毛利率同比增长率"):
            with self.subTest(question=question), self.assertRaises(MetricError):
                self.run_question(question)

    def test_model_uses_same_program_selected_years(self):
        class Model:
            def generate(self, messages):
                self_messages.append(messages)
                return json.dumps({"requests": [{"metric": "gross_margin", "transform": "value"}], "clarification": None})
        self_messages = []
        plan = model_plan(self.context("告诉我你能查询到的所有年份贵州茅台的毛利率"), self.engine.registry, Model())
        self.assertEqual(plan["years"], [2023, 2024])
        self.assertIn("[2023, 2024]", self_messages[0][0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
