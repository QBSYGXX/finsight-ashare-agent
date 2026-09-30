import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import doctor


class DoctorTests(unittest.TestCase):
    def fixture(self, root, shard="part.safetensors"):
        (root/"config.json").write_text(json.dumps({"text_config":{"mtp_num_hidden_layers":1}}))
        (root/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"mtp.fc.weight":shard,"model.weight":shard}}))

    def test_metadata_reports_missing_shards_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root)
            report=doctor.inspect_model(root)
            self.assertEqual(report["mtp_tensor_count"],1)
            self.assertFalse(report["shards"][0]["exists"])
            (root/"part.safetensors").write_bytes(b"not a tensor file; must never load")
            report=doctor.inspect_model(root)
            self.assertTrue(report["shards"][0]["exists"])
            self.assertEqual(report["mtp_num_hidden_layers"],1)

    def test_shard_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ("../private", "/etc/passwd", "C:/private", "..\\private"):
                self.fixture(root,name)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    doctor.inspect_model(root)

    def test_unavailable_tool_is_a_record_not_an_exception(self):
        with patch.object(doctor.shutil,"which",return_value=None):
            self.assertEqual(doctor.command(["nvcc","--version"]),{"available":False})

    def test_timeout_is_bounded_and_recorded(self):
        with patch.object(doctor.shutil,"which",return_value="/usr/bin/docker"), \
             patch.object(doctor.subprocess,"run",side_effect=subprocess.TimeoutExpired("docker",12)) as run:
            result=doctor.command(["docker","info"])
            self.assertEqual(result["timeout_seconds"],12)
            self.assertEqual(run.call_args.kwargs["timeout"],12)

    def test_package_tamper_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); addon=root/"addon";addon.mkdir()
            file=addon/"file.py";file.write_bytes(b"original")
            manifest={"files":{"addon/file.py":hashlib.sha256(file.read_bytes()).hexdigest()}}
            (addon/"PACKAGE_MANIFEST.json").write_text(json.dumps(manifest))
            with patch.object(doctor,"ROOT",root),patch.object(doctor,"ADDON",addon):
                doctor.verify_package()
                file.write_bytes(b"changed")
                with self.assertRaises(ValueError):doctor.verify_package()
                manifest["files"]={"../outside":"fake"}
                (addon/"PACKAGE_MANIFEST.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):doctor.verify_package()


if __name__ == "__main__":
    unittest.main()
