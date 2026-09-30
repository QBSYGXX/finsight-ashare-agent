import copy
import unittest
from scoring import assess, canonical_rows, summarize


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.case = {"expect": {"kind": "task", "company": "000063", "years": [2024], "scope": "consolidated",
                               "money_unit": "亿元", "requests": [{"metric": "gross_margin", "transform": "value"}]}}
        self.result = {"status": "completed", "plan": {k: v for k, v in self.case["expect"].items() if k != "kind"},
                       "results": [{"company": "000063", "year": 2024, "scope": "consolidated", "metric": "gross_margin", "transform": "value",
                                    "status": "ok", "value": "37.9127", "unit": "%", "citations": ["f1"],
                                    "sources": [{"id": "f1", "page": 111}]}]}
        self.reference = {"status": "completed", "canonical_rows": canonical_rows(self.result["results"])}

    def test_correct_task(self):
        self.assertTrue(assess(self.case, self.reference, self.result, 1)["passed"])

    def test_wrong_value_unit_or_source_fails(self):
        for key, value in (("value", "0.3791"), ("unit", "亿元"), ("citations", []), ("sources", [{"id": "f1", "page": 110}])):
            changed = copy.deepcopy(self.result)
            changed["results"][0][key] = value
            self.assertFalse(assess(self.case, self.reference, changed, 1)["passed"])

    def test_duplicate_or_missing_plan_fails(self):
        for requests in ([], self.result["plan"]["requests"] * 2):
            changed = copy.deepcopy(self.result)
            changed["plan"]["requests"] = requests
            self.assertFalse(assess(self.case, self.reference, changed, 1)["passed"])

    def test_missing_data_not_a_generic_refusal_pass(self):
        changed = {"status": "needs_clarification", "results": []}
        self.assertFalse(assess(self.case, self.reference, changed, 1)["passed"])

    def test_preflight_must_not_claim_model_success(self):
        case = {"expect": {"kind": "boundary", "statuses": ["unsupported_period"], "must_not_call_model": True}}
        result = {"status": "unsupported_period", "results": []}
        self.assertTrue(assess(case, {}, result, 0)["passed"])
        self.assertFalse(assess(case, {}, result, 1)["passed"])
        result["results"] = self.result["results"]
        self.assertFalse(assess(case, {}, result, 0)["passed"])

    def test_invalid_json_does_not_count_as_safe_abstention(self):
        case = {"expect": {"kind": "boundary", "statuses": ["needs_clarification", "unsupported_metric"], "must_not_call_model": False}}
        self.assertFalse(assess(case, {}, {"status": "invalid_plan", "results": []}, 1)["passed"])

    def test_summary_separates_unrun_and_preflight(self):
        rows = [{"model_calls": 0, "category": "quarter", "assessment": {"passed": True}},
                {"model_calls": 1, "category": "query", "assessment": {"passed": False}}]
        summary = summarize(rows, 3)
        self.assertEqual(summary["not_run"], 1)
        self.assertEqual(summary["groups"]["preflight_only"], {"executed": 1, "passed": 1})
        self.assertEqual(summary["groups"]["model_called"]["passed"], 0)


if __name__ == "__main__":
    unittest.main()
