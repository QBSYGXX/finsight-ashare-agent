"""Contract regressions, including the user's unsupported H1 question."""
import unittest

from preflight import dispatch, inspect_request


class PreflightTests(unittest.TestCase):
    def test_subannual_periods_never_call_model(self):
        periods = ["2024H1", "2024h2", "2024 H 1", "２０２４Ｈ１", "2024年上半年",
                   "2024年半年度", "2024中报", "2024年中期", "2024Q1", "2024年第三季度",
                   "2024年三季报", "2024年6月30日", "2024-06-30", "2024/06/30",
                   "2024年1-6月", "2024年六月末", "2024 TTM", "2024 YTD",
                   "2024 first half", "2024 semiannual"]
        for period in periods:
            with self.subTest(period=period):
                trace = dispatch("贵州茅台" + period + "归母净利润、毛利率以及他们的同比增幅",
                                 lambda: self.fail("Unsupported request loaded agent"))
                self.assertEqual(trace["status"], "unsupported_period")
                self.assertFalse(trace["model_invoked"])
                self.assertEqual(trace["events"], [])
                self.assertIsNone(trace["final"])

    def test_separate_requested_outputs(self):
        for question in ["贵州茅台2024年归母净利润、毛利率以及他们的同比增幅",
                         "贵州茅台2024年营业收入和归母净利润", "贵州茅台2024年归母净利润及其同比"]:
            with self.subTest(question=question):
                self.assertEqual(inspect_request(question)["status"], "needs_decomposition")

    def test_scalar_formulas_are_allowed(self):
        for question in ["贵州茅台2024年归母净利润同比增长率是多少？",
                         "贵州茅台2024年毛利率，用营业收入和营业成本计算",
                         "美的集团2024年经营现金流净额相当于归母净利润的百分之几？",
                         "中兴通讯2024年底每100元资产对应多少元负债？",
                         "贵州茅台2024年全年归母净利润多少亿元？"]:
            with self.subTest(question=question):
                self.assertEqual(inspect_request(question)["status"], "accepted")

    def test_snapshot_boundary(self):
        self.assertEqual(inspect_request("贵州茅台2026年收入")["status"], "unsupported_year")
        self.assertEqual(inspect_request("贵州茅台去年收入")["status"], "needs_clarification")

    def test_accepted_question_unchanged(self):
        received = []

        class Agent:
            def run(self, question):
                received.append(question)
                return {"status": "completed", "final": {"value": "example"}}

        question = "贵州茅台2024年归母净利润同比增长率是多少？"
        trace = dispatch(question, Agent)
        self.assertEqual(received, [question])
        self.assertEqual(trace["final"], {"value": "example"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
