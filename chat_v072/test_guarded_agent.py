"""Scripted model regressions exercise real tools; they are not GPU inference."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "evaluation_v07"
if not REFERENCE.exists():
    REFERENCE = ROOT.parent / "_release_work/finsight-evaluation-v0.7/evaluation_v07"
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "src"), str(REFERENCE)]
from finsight_ashare.tools import FinancialTools
from finsight_ashare.protocol import action
from guarded_agent import GuardedAgent, ScopedTools
from resolver import CompanyResolver
from chat import inspect_chat_request


def fact(metric, value, company="600519", unit="元", scope="consolidated"):
    return {"id": company + "-2024-" + metric, "company": company, "year": 2024,
            "metric": metric, "value": value, "unit": unit, "scope": scope,
            "report_id": company + "-2024", "report_year": 2024, "page": 63,
            "source_url": "https://example.invalid/report", "pdf_sha256": "fixture",
            "verified": True}


def final(value="90.0000", status="ok", company="600519", scope="consolidated"):
    return {"status": status, "company": company, "year": 2024, "scope": scope,
            "value": value, "unit": "%", "citations": [] if value is None else ["600519-2024-revenue", "600519-2024-cost"],
            "answer": "fixture answer"}


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.facts = [fact("revenue", "1000"), fact("cost", "100")]
        self.raw = FinancialTools(self.facts)
        self.tools = ScopedTools(self.raw, "600519")

    def margin(self, **kwargs):
        return self.tools.call("table_read", company="600519", year=2024, metric="gross_margin", **kwargs)

    def test_derived_value_and_audit(self):
        result = self.margin()
        self.assertEqual(result["value"], "90.0000")
        self.assertEqual([c["tool_name"] for c in result["subcalls"]], ["table_read", "table_read", "calculate"])
        self.assertEqual(result["citations"], [f["id"] for f in self.facts])
        self.assertIsNone(self.tools.validate_final(final(), result["citations"], "贵州茅台2024年毛利率"))

    def test_units_are_normalized(self):
        self.facts[1].update(value="0.01", unit="万元")
        self.assertEqual(self.margin()["value"], "90.0000")

    def test_missing_operand_and_parent_scope(self):
        for facts, scope in [(self.facts[:1], "consolidated"), (self.facts, "parent")]:
            with self.subTest(scope=scope):
                self.tools = ScopedTools(FinancialTools(facts), "600519")
                result = self.margin(scope=scope)
                self.assertEqual(result["status"], "missing")
                self.assertNotIn("value", result)
                self.assertEqual(len(result["subcalls"]), 2)
                self.assertIsNone(self.tools.validate_final(final(None, "insufficient_evidence", scope=scope), [], "贵州茅台2024年毛利率"))

    def test_no_company_switch(self):
        for tool, kwargs in [("table_read", {"company": "600418", "year": 2024, "metric": "gross_margin"}),
                             ("report_search", {"company": "600418", "query": "收入"}),
                             ("calculate", {"operation": "convert", "fact_ids": ["600418-2024-revenue"]})]:
            with self.subTest(tool=tool), self.assertRaisesRegex(ValueError, "COMPANY_MISMATCH"):
                self.tools.call(tool, **kwargs)

    def test_invalid_parameters_and_basis(self):
        with self.assertRaisesRegex(ValueError, "INVALID_ARGUMENTS"):
            self.margin(unexpected=True)
        self.facts[1]["report_id"] = "other-snapshot"
        with self.assertRaisesRegex(ValueError, "same report snapshot"):
            self.margin()

    def test_invalid_revenue(self):
        self.facts[0]["value"] = "0"
        with self.assertRaisesRegex(ValueError, "positive revenue"):
            self.margin()

    def test_false_missing_is_not_an_answer(self):
        missing = final(None, "insufficient_evidence")
        self.assertIn("UNSUPPORTED_MISSING", self.tools.validate_final(missing, [], "贵州茅台2024年毛利率"))
        self.margin()
        self.assertIn("FALSE_MISSING", self.tools.validate_final(missing, [], "贵州茅台2024年毛利率"))

    def test_wrong_value_company_scope_citations(self):
        result = self.margin()
        for patch in ({"value": "95"}, {"unit": "亿元"}, {"company": "600418"}, {"year": 2023},
                      {"citations": ["600519-2024-revenue"]}, {"scope": "parent"}):
            with self.subTest(patch=patch):
                self.assertIsNotNone(self.tools.validate_final({**final(), **patch}, result["citations"], "贵州茅台2024年合并毛利率"))

    def run_model(self, responses):
        class Model:
            def generate(self, messages):
                return responses.pop(0)
        resolver = CompanyResolver([{"code": "600519", "name": "贵州茅台"}], {"600519": []})
        with redirect_stdout(io.StringIO()):
            return GuardedAgent(Model(), self.raw, resolver).run("贵州茅台2024年毛利率")

    def test_scripted_model_completes_with_real_tools(self):
        trace = self.run_model([action("table_read", company="600519", year=2024, metric="gross_margin"),
                                "<final_result>" + json.dumps(final()) + "</final_result>"])
        self.assertEqual(trace["status"], "completed")
        self.assertTrue(trace["answer_validation"]["passed"])
        self.assertEqual(len(trace["tool_audit"][0]["observation"]["subcalls"]), 3)
        self.assertIn("本入口新增派生指标", trace["effective_system_prompt"])

    def test_scripted_original_bad_sequence_rejected(self):
        trace = self.run_model([action("table_read", company="600519", year=2024, metric="gross_margin"),
                                action("table_read", company="600418", year=2024, metric="gross_margin"),
                                "<final_result>" + json.dumps(final(None, "insufficient_evidence")) + "</final_result>"])
        self.assertEqual(trace["status"], "answer_validation_failed")
        self.assertIsNone(trace["final"])
        self.assertIn("COMPANY_MISMATCH", trace["events"][1]["error"])
        self.assertIn("FALSE_MISSING", trace["answer_validation"]["error"])
        self.assertIn("rejected_final", trace)

    def test_period_and_margin_change_checks(self):
        for q in ["贵州茅台2024H1毛利率", "中兴通讯2025年一季报归母净利润是多少"]:
            self.assertEqual(inspect_chat_request(q)["status"], "unsupported_period")
        self.assertEqual(inspect_chat_request("贵州茅台2024年毛利率同比增长率")["status"], "needs_decomposition")
        self.assertEqual(inspect_chat_request("贵州茅台2024年毛利率")["status"], "accepted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
