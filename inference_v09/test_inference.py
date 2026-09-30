"""CPU-only orchestration/adapter tests; real numerical kernel gates run on the server."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent / "metrics"), str(ROOT / "metrics_v08"), str(ROOT / "src")]
from driver import summarize, verify_package
from runtime_check import save, validate_profile
from kernels import conv_prefill, conv_decode
from fast_model import CachedGeneration


class InferenceTests(unittest.TestCase):
    def test_cached_wrapper_sets_only_missing_options(self):
        raw = MagicMock()
        proxy = CachedGeneration(raw)
        proxy.generate(input_ids="ids", max_new_tokens=128)
        self.assertTrue(raw.generate.call_args.kwargs["use_cache"])
        self.assertEqual(raw.generate.call_args.kwargs["logits_to_keep"], 1)
        self.assertEqual(raw.generate.call_args.kwargs["max_new_tokens"], 128)
        proxy.generate(use_cache=False, logits_to_keep=2)
        self.assertFalse(raw.generate.call_args.kwargs["use_cache"])
        self.assertEqual(raw.generate.call_args.kwargs["logits_to_keep"], 2)

    def test_profile_rejects_unmatched_binary_environment(self):
        expected = {"machine": "x86_64", "cxx11_abi": True, "python": "3.11", "torch": "2.8.0+cu126", "transformers": "5.8.0", "triton": "3.4.0"}
        report = {"platform": "Linux", "python": "3.11.16", "machine": "x86_64", "cxx11_abi": True,
                  "cuda_available": True, "versions": {k: expected[k] for k in ("torch", "transformers", "triton")}}
        validate_profile(report, expected)
        for changed in ({"platform": "Windows"}, {"python": "3.12.1"}, {"cuda_available": False}, {"cxx11_abi": False}):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                validate_profile({**report, **changed}, expected)

    def test_convolution_adapter_transposes_axes_and_preserves_cache_object(self):
        x, state, weight, y = [MagicMock() for _ in range(4)]
        x.ndim, x.shape, state.shape, weight.shape = 3, (2, 32, 1), (2, 32, 4), (32, 4)
        state.ndim, weight.ndim = 3, 2
        state.is_contiguous.return_value = True
        prefill, decode = MagicMock(return_value=(y, None)), MagicMock(return_value=(y, state))
        modules = {"fla": MagicMock(), "fla.modules": MagicMock(),
                   "fla.modules.conv": SimpleNamespace(causal_conv1d=prefill),
                   "fla.modules.conv.triton": MagicMock(),
                   "fla.modules.conv.triton.ops": SimpleNamespace(causal_conv1d_update=decode)}
        with patch.dict(sys.modules, modules):
            conv_prefill(x, weight, activation="silu")
            x.transpose.assert_called_with(1, 2)
            self.assertEqual(prefill.call_args.kwargs["backend"], "triton")
            y.transpose.assert_called_with(1, 2)
            conv_decode(x, state, weight, activation="silu")
            self.assertIs(decode.call_args.kwargs["cache"], state)
            self.assertIs(decode.call_args.kwargs["weight"], weight)
            state.copy_.assert_not_called()

    def test_noncontiguous_cache_writeback_across_decode_steps(self):
        # A dependency-free storage double: checks cache ownership and history,
        # not Triton numerical correctness (that requires the server GPU probe).
        class Cache:
            ndim, shape, dtype, device = 3, (1, 1, 4), "bf16", "test"

            def __init__(self, values, contiguous=False):
                self.values, self.aligned = list(values), contiguous

            def is_contiguous(self):
                return self.aligned

            def contiguous(self):
                return Cache(self.values, True)

            def copy_(self, other):
                self.values[:] = other.values

        state = Cache([0, 1, 2, 3])
        alias = state.values
        x, weight, y = MagicMock(), MagicMock(), MagicMock()
        x.ndim, x.shape, weight.ndim, weight.shape = 3, (1, 1, 1), 2, (1, 4)
        observed = []

        def update(**kwargs):
            cache = kwargs["cache"]
            self.assertIsNot(cache, state)
            self.assertTrue(cache.is_contiguous())
            observed.append(list(cache.values))
            cache.values[:] = cache.values[1:] + [x.value]
            return y, cache

        module = SimpleNamespace(causal_conv1d_update=update)
        with patch.dict(sys.modules, {"fla.modules.conv.triton.ops": module}):
            for value in range(4, 12):
                x.value = value
                conv_decode(x, state, weight)
                self.assertEqual(alias, list(range(value - 3, value + 1)))
                self.assertIs(alias, state.values)
        self.assertEqual(observed[-1], [7, 8, 9, 10])

    def test_shape_mismatch_reports_layout_before_kernel_import(self):
        x, state, weight = MagicMock(), MagicMock(), MagicMock()
        x.ndim, x.shape = 3, (1, 32, 1)
        state.ndim, state.shape = 3, (1, 32, 3)
        weight.ndim, weight.shape = 2, (32, 4)
        state.stride.return_value = (128, 1, 32)
        with self.assertRaisesRegex(ValueError, "dimensions do not match.*stride.*128"):
            conv_decode(x, state, weight)

    def test_unsupported_cache_layout_fails_before_kernel_call(self):
        x, state, weight = MagicMock(), MagicMock(), MagicMock()
        x.ndim, x.shape = 3, (1, 32, 2)
        with self.assertRaises(ValueError):
            conv_decode(x, state, weight)
        with self.assertRaises(ValueError):
            conv_prefill(x, weight, seq_idx="packed")

    def write_measurements(self, directory):
        for backend, seconds in (("baseline", 8), ("fast", 2)):
            folder = directory / backend
            save(folder / "fixed_work.json", {"prompt_sha256": "same", "median_seconds": seconds})
            save(folder / "runtime.json", {"backend": backend})
            save(folder / "trace.json", {"status": "completed", "plan": {"requests": [{"metric": "gross_margin", "transform": "value"}]},
                                       "results": [{"metric": "gross_margin", "year": 2024, "value": "40.0000", "unit": "%"}]})

    def test_summary_requires_equal_completed_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_measurements(root)
            result = summarize(root)
            self.assertEqual(result["fixed_work_speedup"], 4)
            self.assertTrue(result["same_completed_answer"])
            path = root / "fast/trace.json"
            trace = json.loads(path.read_text())
            trace["results"][0]["value"] = "41.0000"
            save(path, trace)
            self.assertFalse(summarize(root)["same_completed_answer"])
            trace["status"] = "invalid_plan"
            save(path, trace)
            self.assertFalse(summarize(root)["same_completed_answer"])

    def test_different_prompt_invalidates_speed_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_measurements(root)
            save(root / "fast/fixed_work.json", {"prompt_sha256": "different", "median_seconds": 1})
            with self.assertRaisesRegex(RuntimeError, "prompt mismatch"):
                summarize(root)

    def test_package_members_cannot_escape_addon(self):
        import driver
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            addon = root / "inference_v09"
            addon.mkdir()
            save(addon / "PACKAGE_MANIFEST.json", {"files": {"../outside": "invalid"}})
            with patch.object(driver, "ROOT", root), patch.object(driver, "ADDON", addon), self.assertRaises(RuntimeError):
                verify_package()


if __name__ == "__main__":
    unittest.main(verbosity=2)
