"""Exercise the actual adapter with a fake tensor runtime; no CUDA claims."""
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        row, col = key
        value = self.rows[row][col]
        return SimpleNamespace(tolist=lambda: value)


class AdapterTests(unittest.TestCase):
    def exercise(self, natural):
        class Error(ValueError):
            def __init__(self, code, message):
                super().__init__(message)
        fake = {"engine": SimpleNamespace(MetricError=Error),
                "planner_model": SimpleNamespace(PlannerModel=object),
                "transformers": SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list)}
        with patch.dict(sys.modules, fake):
            spec = importlib.util.spec_from_file_location("test_budget_adapter_impl", Path(__file__).with_name("budget_model.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            m = module.BudgetPlanner.__new__(module.BudgetPlanner)
            m.policy, m.think_budget, m.answer_budget, m.seed = "budgeted", 64, 512, 42
            m.options = {"context_length": 16384}
            enc = lambda text: list(map(ord, text))
            dec = lambda ids, **kw: "".join(chr(i) for i in ids if i != 99999)
            prompt = "prompt<|im_start|>assistant\n<think>\n"
            m.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: prompt, encode=lambda s, **k: enc(s),
                decode=dec, convert_tokens_to_ids=lambda s: 99999, unk_token_id=None, pad_token_id=0)
            m.torch = SimpleNamespace(tensor=lambda rows, **k: Tensor(rows), ones_like=lambda x: Tensor([[1]*x.shape[1]]),
                long="long", cuda=SimpleNamespace(synchronize=MagicMock()), inference_mode=nullcontext, manual_seed=MagicMock())
            first = enc("done</think>") if natural else enc("x"*64)
            final = '{"requests":[],"clarification":"which year?"}'
            seen = []
            def generate(**kw):
                seen.append(kw)
                ids = kw["input_ids"].rows[0]
                new = first if len(seen) == 1 else enc(final)+[99999]
                if len(seen) == 1:
                    self.assertEqual(kw["stopping_criteria"][0](Tensor([ids+new]), None), natural)
                return Tensor([ids+new])
            m.model = SimpleNamespace(device="fake", generate=generate)
            with redirect_stdout(io.StringIO()):
                output = m.generate([])
            self.assertEqual(output.split("</think>",1)[1].strip(), final)
            self.assertEqual(m.last_generation["forced_close"], not natural)
            self.assertEqual(m.last_generation["stop_reason"], "eos")
            self.assertEqual([c["max_new_tokens"] for c in seen], [64,512])
            self.assertNotIn("stopping_criteria", seen[1])
            self.assertTrue(all(c["use_cache"] for c in seen))
            bridge = [] if natural else enc("\n</think>\n\n")
            self.assertEqual(seen[1]["input_ids"].rows[0], enc(prompt)+first+bridge)
            m.torch.manual_seed.assert_called_once_with(42)

    def test_forced_thinking_continues_same_turn_with_answer_reserve(self):
        self.exercise(False)

    def test_natural_thinking_boundary_uses_no_injected_marker(self):
        self.exercise(True)


if __name__ == "__main__":
    unittest.main()
