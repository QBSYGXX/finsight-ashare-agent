import unittest
from budget import reserve_context, run_stages

EOS = 99999
enc = lambda s: list(map(ord, s))
dec = lambda ids: "".join(chr(i) for i in ids if i != EOS)


class BudgetTests(unittest.TestCase):
    def run_case(self, first, second=None, cap=None):
        calls = []
        def generate(prefix, budget, stop, phase):
            calls.append((prefix, budget, stop, phase))
            return first if phase == "thinking" else second
        result, meta = run_stages(enc("prompt<think>"), generate, dec,
            enc("\n</think>\n\n"), EOS, cap if cap is not None else len(first), 512)
        return result, meta, calls

    def test_forced_boundary_preserves_original_tokens_and_reserves_answer(self):
        first = enc('draft {"requests": ["wrong"]}')
        final = enc('{"requests": [], "clarification": "which year?"}')+[EOS]
        ids, m, calls = self.run_case(first, final)
        self.assertTrue(m["forced_close"])
        self.assertEqual(m["stop_reason"], "eos")
        self.assertEqual(calls[1], (enc("prompt<think>")+first+enc("\n</think>\n\n"), 512, False, "answer"))
        self.assertEqual(dec(ids).split("</think>")[1].strip(), dec(final))
        self.assertEqual(m["thinking_tokens"], len(first))
        self.assertEqual(m["answer_tokens"], len(final))

    def test_natural_close_does_not_inject_another_marker(self):
        first = enc("short thought</think>")
        ids, m, calls = self.run_case(first, enc("{}")+[EOS], cap=1024)
        self.assertFalse(m["forced_close"])
        self.assertEqual(dec(ids).count("</think>"), 1)
        self.assertEqual(calls[1][0], enc("prompt<think>")+first)

    def test_answer_budget_exhaustion_is_not_success_even_if_json_exists(self):
        _, m, _ = self.run_case(enc("think"), enc("{}"))
        self.assertEqual(m["stop_reason"], "max_new_tokens")

    def test_early_eos_is_not_repaired_into_a_final_answer(self):
        ids, m, calls = self.run_case(enc("unfinished")+[EOS], cap=1024)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("</think>", dec(ids))
        self.assertTrue(m["early_eos"])

    def test_eos_exactly_at_answer_boundary_counts_as_finished(self):
        _, m, _ = self.run_case(enc("thought"), [65]*511+[EOS])
        self.assertEqual(m["answer_tokens"], 512)
        self.assertEqual(m["stop_reason"], "eos")

    def test_think_close_at_budget_boundary_is_natural(self):
        _, m, _ = self.run_case(enc("thought</think>"), enc("{}")+[EOS])
        self.assertFalse(m["forced_close"])

    def test_unexpected_early_stop_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_case(enc("unfinished"), enc("{}"), cap=1024)

    def test_context_reserves_thought_bridge_and_answer_without_truncating(self):
        reserve_context(745, 2287, 1024, 512, 6)
        with self.assertRaises(ValueError):
            reserve_context(745, 2286, 1024, 512, 6)
        with self.assertRaises(ValueError):
            reserve_context(745, 16384, 0, 512, 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
