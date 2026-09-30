"""Regression coverage for per-metric YoY semantics; no model downloads."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent / "metrics"))
from engine import load_registry, MetricError
from planner import metric_mentions, model_plan, rules_plan
from contract import ambiguous_ratio_yoy


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry()

    def ambiguous(self, question):
        return ambiguous_ratio_yoy(question, metric_mentions(question, self.registry), self.registry)

    def test_every_registered_ratio_requires_explicit_yoy(self):
        for metric, spec in self.registry.items():
            if spec["unit"] == "%" and spec["aliases"]:
                name = spec["aliases"][0]
                with self.subTest(metric=metric):
                    self.assertEqual(self.ambiguous(name + "同比"), [metric])
                    self.assertEqual(self.ambiguous(name + "同比增长率"), [])
                    self.assertEqual(self.ambiguous(name + "同比变化多少个百分点"), [])

    def test_other_metric_growth_does_not_resolve_ambiguity(self):
        self.assertEqual(self.ambiguous("毛利率同比，营业收入同比增长率"), ["gross_margin"])
        self.assertEqual(self.ambiguous("营业收入同比增长率，资产负债率同比"), ["debt_ratio"])
        self.assertEqual(self.ambiguous("毛利率同比增长率，资产负债率同比"), ["debt_ratio"])

    def test_group_and_prefix(self):
        self.assertEqual(set(self.ambiguous("毛利率、资产负债率及它们的同比")), {"gross_margin", "debt_ratio"})
        self.assertEqual(self.ambiguous("毛利率、资产负债率及它们的同比百分点变化"), [])
        self.assertEqual(self.ambiguous("同比毛利率"), ["gross_margin"])
        self.assertEqual(self.ambiguous("同比增长率：毛利率"), [])
        self.assertEqual(self.ambiguous("营业收入同比"), [])

    def test_guard_runs_before_model_for_both_policies(self):
        context = {"question": "中兴通讯2024年毛利率同比"}
        for policy in ("baseline", "budgeted"):
            class NeverCalled:
                def generate(self, messages):
                    raise AssertionError("Ambiguous query must not call model: " + policy)
            with self.assertRaises(MetricError) as caught:
                model_plan(context, self.registry, NeverCalled())
            self.assertEqual(caught.exception.code, "needs_clarification")
            self.assertEqual(caught.exception.details["guard"], "ratio_yoy_semantics_v1")
        with self.assertRaises(MetricError) as caught:
            rules_plan(context, self.registry)
        self.assertEqual(caught.exception.code, "needs_clarification")

    def test_unambiguous_model_plan_still_runs(self):
        context = {"question": "中兴通讯2024年毛利率、资产负债率", "year": 2024, "years": [2024],
                   "year_spans": [(4, 8)], "resolution": {"matched_names": [{"alias": "中兴通讯"}]}}
        requests = [{"metric": m, "transform": "value"} for m in ("gross_margin", "debt_ratio")]
        class Model:
            def generate(self, messages):
                return json.dumps({"requests": requests, "clarification": None})
        self.assertEqual(model_plan(context, self.registry, Model())["requests"], requests)


if __name__ == "__main__":
    unittest.main()
