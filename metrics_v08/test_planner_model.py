"""No GPU needed: template contract, generation limits and fail-closed execution."""
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from test_metrics import fixtures, CompanyResolver, MetricEngine, MetricError
from chat import execute_question
from planner_model import PlannerModel


SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
QUESTION = "贵州茅台2024年毛利率、资产负债率"
PLAN = json.dumps({"requests": [{"metric": m, "transform": "value"}
                                for m in ("gross_margin", "debt_ratio")], "clarification": None})


def model_fixture(output=PLAN, ids=None, input_tokens=100):
    model = PlannerModel.__new__(PlannerModel)
    model.options = {"context_length": 4096, "max_new_tokens": 512}
    model.torch = SimpleNamespace(inference_mode=nullcontext, manual_seed=MagicMock())
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template.return_value = "prompt" + SUFFIX
    model.tokenizer.convert_tokens_to_ids.return_value = 999
    model.tokenizer.unk_token_id = None
    model.tokenizer.pad_token_id = 0
    model.tokenizer.decode.return_value = output
    batch = MagicMock()
    batch.__getitem__.return_value.shape = (1, input_tokens)
    batch.to.return_value = {"input_ids": "fake_inputs"}
    model.tokenizer.return_value = batch
    model.model = MagicMock()
    model.model.device = "fake_device"
    model.model.generate.return_value.__getitem__.return_value.tolist.return_value = ids if ids is not None else [1, 2, 999]
    return model


class PlannerGenerationTests(unittest.TestCase):
    def setUp(self):
        self.engine = MetricEngine(fixtures())
        self.resolver = CompanyResolver([{"code": "600519", "name": "贵州茅台"}], {"600519": ["茅台"]})

    def execute(self, model):
        return execute_question(QUESTION, self.engine, self.resolver, "model", model)

    def thinking_model(self, output, ids=None):
        model = model_fixture(output, ids)
        model.enable_thinking = True
        model.options = {"context_length": 8192, "max_new_tokens": 2048}
        model.tokenizer.apply_chat_template.return_value = "prompt<|im_start|>assistant\n<think>\n"
        return model

    def test_thinking_parses_only_final_json_after_closed_segment(self):
        for prefix in ("", "<think>\n"):
            model = self.thinking_model(prefix + 'Draft: {"metric":"invented"}\n</think>\n' + PLAN)
            result = self.execute(model)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["plan"]["planner_final"], PLAN)
            self.assertIn("invented", result["plan"]["planner_output"])
            self.assertTrue(model.tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])
            self.assertEqual(model.model.generate.call_args.kwargs["max_new_tokens"], 2048)

    def test_thinking_must_finish_before_any_plan_executes(self):
        for raw in (PLAN, 'draft ' + PLAN, 'draft</think>', 'draft</think>' + PLAN + '</think>',
                    '<think><think>draft</think>' + PLAN, 'draft</think>{"requests":'):
            with self.subTest(raw=raw):
                result = self.execute(self.thinking_model(raw))
                self.assertEqual(result["status"], "invalid_plan")
                self.assertEqual(result["results"], [])

    def test_thinking_token_cap_remains_a_failure(self):
        result = self.execute(self.thinking_model('draft</think>' + PLAN, [1] * 2048))
        self.assertEqual(result["status"], "generation_limit")
        self.assertEqual(result["details"]["generation"]["generated_tokens"], 2048)

    def test_context_reduction_is_visible(self):
        model = self.thinking_model('draft</think>' + PLAN)
        model.tokenizer.return_value.__getitem__.return_value.shape = (1, 7000)
        result = self.execute(model)
        generation = result["plan"]["generation"]
        self.assertTrue(generation["context_limited"])
        self.assertEqual((generation["requested_max_new_tokens"], generation["max_new_tokens"]), (2048, 1192))

    def test_constructor_defaults_and_override_do_not_mutate_shared_config(self):
        from finsight_ashare.models import LocalModel
        options = {"context_length": 4096, "max_new_tokens": 512}

        def fake_init(instance, root, adapter):
            instance.options = options
            instance.model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=262144)))

        with patch.object(LocalModel, "__init__", fake_init):
            model = PlannerModel(Path("."), thinking=True)
            self.assertEqual((model.options["max_new_tokens"], model.options["context_length"]), (2048, 8192))
            model = PlannerModel(Path("."), thinking=True, max_new_tokens=4096, context_length=16384)
            self.assertEqual((model.options["max_new_tokens"], model.options["context_length"]), (4096, 16384))
        self.assertEqual(options, {"context_length": 4096, "max_new_tokens": 512})

    def test_cli_passes_thinking_limits_to_model_and_writes_trace(self):
        import chat
        fake = self.thinking_model('draft</think>' + PLAN)
        argv = ["chat.py", "--planner", "model", "--model", "base", "--thinking", "on",
                "--max-new-tokens", "2048", "--context-length", "8192", "--question", QUESTION]
        with patch("sys.argv", argv), patch.object(chat, "verify_inputs", return_value=(fixtures(), {})), \
                patch.object(chat.CompanyResolver, "load", return_value=self.resolver), \
                patch("planner_model.PlannerModel", return_value=fake) as constructor, \
                patch.object(chat, "write_json") as save, patch.object(chat, "write_report"), redirect_stdout(io.StringIO()):
            chat.main()
        constructor.assert_called_once_with(chat.ROOT, None, thinking=True, max_new_tokens=2048, context_length=8192,
                                            decoding="auto", seed=42)
        self.assertEqual(save.call_args.args[1]["status"], "completed")
        self.assertIn("metrics_v085", str(save.call_args.args[0]))

    def test_user_extra_yoy_trace_stays_rejected_without_execution(self):
        raw = json.dumps({"requests": [{"metric": "parent_net_profit", "transform": t}
                                       for t in ("value", "yoy_growth")], "clarification": None})
        self.engine.execute_years = MagicMock()
        result = execute_question("贵州茅台可查询到的历年归母净利润", self.engine, self.resolver,
                                  "model", model_fixture(raw))
        self.assertEqual(result["status"], "invalid_plan")
        self.assertEqual(result["details"]["expected_requests"], [{"metric": "parent_net_profit", "transform": "value"}])
        self.assertEqual(result["details"]["extra_requests"], [{"metric": "parent_net_profit", "transform": "yoy_growth"}])
        self.assertEqual(result["details"]["missing_requests"], [])
        self.assertEqual(result["details"]["model_output"], raw)
        self.engine.execute_years.assert_not_called()

    def test_historical_values_growth_and_both_are_distinct(self):
        for suffix, transforms in (("", ["value"]), ("同比增长率", ["yoy_growth"]),
                                   ("及其同比增长率", ["value", "yoy_growth"])):
            with self.subTest(suffix=suffix):
                raw = json.dumps({"requests": [{"metric": "parent_net_profit", "transform": t}
                                               for t in transforms], "clarification": None})
                result = execute_question("贵州茅台可查询到的历年归母净利润" + suffix,
                                          self.engine, self.resolver, "model", model_fixture(raw))
                self.assertIn(result["status"], ("completed", "partial"))
                self.assertEqual(result["plan"]["years"], [2023, 2024])
                self.assertEqual([r["transform"] for r in result["plan"]["requests"]], transforms)
                self.assertEqual(len(result["results"]), 2 * len(transforms))

    def test_missing_requested_yoy_is_still_rejected(self):
        raw = json.dumps({"requests": [{"metric": "parent_net_profit", "transform": "value"}], "clarification": None})
        result = execute_question("贵州茅台历年归母净利润及其同比增长率", self.engine, self.resolver,
                                  "model", model_fixture(raw))
        self.assertEqual(result["status"], "invalid_plan")
        self.assertEqual(result["details"]["missing_requests"], [{"metric": "parent_net_profit", "transform": "yoy_growth"}])

    def test_thinking_defaults_to_documented_sampling_profile(self):
        model = self.thinking_model('draft</think>' + PLAN)
        result = self.execute(model)
        kwargs = model.model.generate.call_args.kwargs
        self.assertEqual({key: kwargs[key] for key in ("do_sample", "temperature", "top_p", "top_k", "min_p", "repetition_penalty")},
                         {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0})
        model.torch.manual_seed.assert_called_once_with(42)
        self.assertEqual(result["plan"]["generation"]["decoding"], "sample")

    def test_explicit_greedy_retains_thinking_control(self):
        model = self.thinking_model('draft</think>' + PLAN)
        model.decoding = "greedy"
        self.assertEqual(self.execute(model)["status"], "completed")
        self.assertFalse(model.model.generate.call_args.kwargs["do_sample"])
        self.assertNotIn("temperature", model.model.generate.call_args.kwargs)
        model.torch.manual_seed.assert_not_called()

    def test_sampling_seed_resets_for_each_query_and_is_logged(self):
        model = model_fixture()
        model.decoding, model.seed = "sample", 7
        for _ in range(2):
            result = self.execute(model)
            self.assertEqual(result["plan"]["generation"]["seed"], 7)
        self.assertEqual(model.torch.manual_seed.call_count, 2)
        model.torch.manual_seed.assert_called_with(7)

    def test_cli_rejects_invalid_or_ignored_budget_options(self):
        import chat
        for args in (["--thinking", "on"], ["--planner", "model", "--max-new-tokens", "0"],
                     ["--planner", "model", "--context-length", "100"],
                     ["--planner", "model", "--max-new-tokens", "8192", "--context-length", "4096"]):
            with self.subTest(args=args), patch("sys.argv", ["chat.py", *args]), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                chat.main()
            self.assertEqual(error.exception.code, 2)

    def test_native_no_thinking_flag_and_generation_telemetry(self):
        model = model_fixture()
        result = self.execute(model)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([r["value"] for r in result["results"]], ["40.0000", "40.0000"])
        kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual(kwargs, {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})
        kwargs = model.model.generate.call_args.kwargs
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual((kwargs["max_new_tokens"], kwargs["eos_token_id"], kwargs["pad_token_id"]), (512, 999, 0))
        telemetry = result["plan"]["generation"]
        self.assertEqual((telemetry["stop_reason"], telemetry["generated_tokens"]), ("eos", 3))
        self.assertEqual(result["plan"]["planner_output"], PLAN)

    def test_unfinished_thinking_is_never_executed(self):
        raw = '<think>Thinking Process: "metric": "debt_'
        model = model_fixture(raw, [1] * 512)
        self.engine.execute_years = MagicMock()
        result = self.execute(model)
        self.assertEqual(result["status"], "generation_limit")
        self.assertEqual(result["details"]["model_output"], raw)
        self.assertEqual(result["details"]["generation"]["stop_reason"], "max_new_tokens")
        self.engine.execute_years.assert_not_called()

    def test_truncated_generation_rejected_even_if_json_parses(self):
        self.assertEqual(self.execute(model_fixture(PLAN, [1] * 512))["status"], "generation_limit")

    def test_eos_at_budget_boundary_is_not_called_truncation(self):
        self.assertEqual(self.execute(model_fixture(PLAN, [1] * 511 + [999]))["status"], "completed")

    def test_strict_json_and_schema_errors_keep_raw_output(self):
        for raw in ('<think>unfinished ' + PLAN, '```json\n' + PLAN + '\n```', '{"requests": []}',
                    '{"requests": [{"metric":"gross_margin","transform":"value"}],"clarification":null}'):
            with self.subTest(raw=raw):
                result = self.execute(model_fixture(raw))
                self.assertEqual(result["status"], "invalid_plan")
                self.assertEqual(result["details"]["model_output"], raw)
                self.assertEqual(result["details"]["generation"]["stop_reason"], "eos")

    def test_template_must_honor_thinking_flag(self):
        model = model_fixture()
        model.tokenizer.apply_chat_template.return_value = "<|im_start|>assistant\n<think>\n"
        self.assertEqual(self.execute(model)["status"], "template_error")
        model.model.generate.assert_not_called()

    def test_missing_template_is_recorded(self):
        model = model_fixture()
        model.tokenizer.apply_chat_template.side_effect = ValueError("No chat template")
        self.assertEqual(self.execute(model)["status"], "template_error")
        model.model.generate.assert_not_called()

    def test_context_budget_does_not_truncate_input(self):
        model = model_fixture(input_tokens=4060)
        self.assertEqual(self.execute(model)["status"], "context_limit")
        model.model.generate.assert_not_called()
        model = model_fixture(input_tokens=4000)
        self.assertEqual(self.execute(model)["status"], "completed")
        self.assertEqual(model.model.generate.call_args.kwargs["max_new_tokens"], 96)

    def test_native_snapshot_template_closes_thinking(self):
        snapshot = Path(__file__).resolve().parents[1] / "models/Qwen3.5-9B"
        if not (snapshot / "tokenizer_config.json").exists():
            self.skipTest("Local model template is not installed")
        import jinja2
        template = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))["chat_template"]
        self.assertEqual(template, (snapshot / "chat_template.jinja").read_text(encoding="utf-8"))
        renderer = jinja2.Environment().from_string(template)
        messages = [{"role": "system", "content": "Only JSON"}, {"role": "user", "content": QUESTION}]
        disabled = renderer.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
        enabled = renderer.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
        self.assertTrue(disabled.endswith(SUFFIX))
        self.assertTrue(enabled.endswith("<|im_start|>assistant\n<think>\n"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
