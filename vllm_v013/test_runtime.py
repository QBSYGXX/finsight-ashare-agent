"""CPU tests for isolation, evidence gates, and fair comparison; no pretend GPU success."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import runtime
from adapter import timing_record
from worker import financial_matches
from launcher import comparison


class RuntimeTests(unittest.TestCase):
    def test_base_does_not_load_compat_or_foreign_python(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/foreign", "LD_PRELOAD": "/foreign.so", "CUDA_HOME": "/cuda13"}):
            env = runtime.process_env(1, False)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertNotIn("compat", env["LD_LIBRARY_PATH"])
        for key in ("PYTHONPATH", "LD_PRELOAD", "CUDA_HOME"):
            self.assertNotIn(key, env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_accelerated_only_prefixes_own_compat(self):
        env = runtime.process_env(2, True)
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith(str(runtime.COMPAT)))
        self.assertNotEqual(runtime.BASE, runtime.ENV)

    def test_manifest_rejects_changed_file_and_dependency(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            addon = root / "addon"
            addon.mkdir()
            (addon / "run.py").write_bytes(b"pass")
            (root / "dependency").write_bytes(b"pinned")
            data = {"files": {"run.py": hashlib.sha256(b"pass").hexdigest()},
                    "dependencies": {"dependency": hashlib.sha256(b"pinned").hexdigest()}}
            (addon / "PACKAGE_MANIFEST.json").write_text(json.dumps(data))
            with patch.object(runtime, "ADDON", addon), patch.object(runtime, "ROOT", root):
                runtime.verify_package()
                (root / "dependency").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Required earlier package changed"):
                    runtime.verify_package()
                (root / "dependency").write_bytes(b"pinned")
                (addon / "run.py").write_bytes(b"different")
                with self.assertRaisesRegex(ValueError, "Package file changed"):
                    runtime.verify_package()

    def test_missing_engine_timestamps_are_not_decode_speed(self):
        self.assertIsNone(timing_record(100, 5)["decode_tokens_per_second_estimate"])
        self.assertIsNone(timing_record(100, 5, 0, 5)["decode_tokens_per_second_estimate"])
        self.assertEqual(timing_record(101, 6, 10, 15)["decode_tokens_per_second_estimate"], 20)

    def test_smoke_must_match_value_unit_and_evidence(self):
        good = {"status": "completed", "results": [{"year": 2024, "metric": "gross_margin", "transform": "value",
                 "status": "ok", "value": "37.9127", "unit": "%", "citations": ["fact-id"]}]}
        self.assertTrue(financial_matches(good, good))
        for field, value in (("value", "0.3791"), ("unit", "亿元"), ("citations", []), ("metric", "debt_ratio")):
            bad = copy.deepcopy(good)
            bad["results"][0][field] = value
            self.assertFalse(financial_matches(bad, good))
        bad = {**good, "status": "generation_limit"}
        self.assertFalse(financial_matches(bad, good))

    def test_comparison_rejects_different_prompt_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i, backend in enumerate(("hf", "vllm", "vllm-mtp")):
                p = root / backend
                p.mkdir()
                runtime.save(p / "benchmark.json", {"samples": [{"input_ids_sha256": str(i)}], "median_tokens_per_second": 20})
                runtime.save(p / "trace.json", {"matches_deterministic_reference": True})
            with self.assertRaisesRegex(RuntimeError, "Prompt token IDs differ"):
                comparison(root)

    def test_install_refuses_unowned_environment_before_venv(self):
        import install
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / "existing"
            env.mkdir()
            with patch.object(install, "ENV", env), patch.object(install, "verify_package", return_value={}), \
                 patch.object(install, "sha256", return_value="lock"), patch.object(install.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "not owned"):
                    install.main(1)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
