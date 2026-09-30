import ctypes as C
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import probe
import worker


class ProbeTests(unittest.TestCase):
    def test_native_grammar_error_string_is_not_success(self):
        fake = SimpleNamespace(LLMatcher=SimpleNamespace(grammar_from_json_schema=lambda s: "grammar",
                                                        validate_grammar=lambda s: "bad grammar"))
        with patch.dict(sys.modules, {"llguidance": fake}):
            with self.assertRaisesRegex(RuntimeError, "bad grammar"):
                worker.check_llguidance()

    def test_chained_library_aliases_target_first(self):
        self.assertEqual(probe.alias_order({"libcuda.so": "libcuda.so.1", "libcuda.so.1": "libcuda.so.575"}),
                         ["libcuda.so.1", "libcuda.so"])
        with self.assertRaises(ValueError):
            probe.alias_order({"a": "b", "b": "a"})
        with self.assertRaises(ValueError):
            probe.alias_order({"a": "../outside"})

    def test_physical_gpu_selection_uses_pci(self):
        text = "0, 00000000:19:00.0, A800, 535.86.10, 20000\n1, 00000000:1A:00.0, A800, 535.86.10, 4\n"
        self.assertEqual(probe.gpu_bus(text, 1)[0], "0000:1A:00.0")
        with self.assertRaises(ValueError):
            probe.gpu_bus(text, 9)

    def test_corrupt_payload_is_rejected_before_links(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "binary").write_bytes(b"changed")
            (root / "PACKAGE_MANIFEST.json").write_text(json.dumps({"files": {"binary": hashlib.sha256(b"original").hexdigest()}, "links": {}}))
            with self.assertRaises(ValueError):
                probe.verify_payload(root)

    def test_existing_file_is_not_overwritten_by_library_alias(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "payload/compat"
            folder.mkdir(parents=True)
            (folder / "lib.so.1").write_bytes(b"existing")
            (folder / "lib.so.2").write_bytes(b"binary")
            (root / "PACKAGE_MANIFEST.json").write_text(json.dumps({"files": {}, "links": {"lib.so.1": "lib.so.2"}}))
            with self.assertRaises(ValueError):
                probe.verify_payload(root)
            self.assertEqual((folder / "lib.so.1").read_bytes(), b"existing")

    def test_driver_probe_executes_and_cleans_up(self):
        self.exercise_driver(False)

    def test_ptx_failure_is_not_reported_as_pass(self):
        self.exercise_driver(True)

    def exercise_driver(self, fail):
        calls = []
        class Function:
            def __init__(self, name):
                self.name = name
            def __call__(self, *args):
                calls.append(self.name)
                if self.name == "cuDriverGetVersion":
                    C.cast(args[0], C.POINTER(C.c_int))[0] = 12090
                elif self.name == "cuDeviceGetByPCIBusId":
                    self.assert_bus = args[1]
                    C.cast(args[0], C.POINTER(C.c_int))[0] = 1
                elif self.name == "cuDeviceGetName":
                    args[0].value = b"A800"
                elif self.name in {"cuCtxCreate_v2", "cuModuleLoadDataEx", "cuModuleGetFunction"}:
                    if fail and self.name == "cuModuleLoadDataEx":
                        return 222
                    C.cast(args[0], C.POINTER(C.c_void_p))[0] = C.c_void_p(123)
                elif self.name == "cuMemAlloc_v2":
                    C.cast(args[0], C.POINTER(C.c_uint64))[0] = 256
                elif self.name == "cuMemcpyDtoH_v2":
                    C.cast(args[0], C.POINTER(C.c_int))[0] = 42
                elif self.name == "cuGetErrorString":
                    C.cast(args[1], C.POINTER(C.c_char_p))[0] = b"unsupported PTX"
                return 0
        class Library:
            def __init__(self):
                self.functions = {}
            def __getattr__(self, key):
                return self.functions.setdefault(key, Function(key))
        library = Library()
        with patch.object(worker.C, "CDLL", return_value=library), patch.object(Path, "read_text", return_value="mapped libcuda.so"):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "unsupported PTX"):
                    worker.check_cuda("0000:1A:00.0", "test.so")
                self.assertNotIn("cuLaunchKernel", calls)
            else:
                report = worker.check_cuda("0000:1A:00.0", "test.so")
                self.assertEqual(report["kernel_output"], 42)
                self.assertIn("cuLaunchKernel", calls)
                self.assertIn("cuMemFree_v2", calls)
                self.assertIn("cuModuleUnload", calls)
        self.assertEqual(library.functions["cuDeviceGetByPCIBusId"].assert_bus, b"0000:1A:00.0")
        self.assertEqual(calls[-1], "cuCtxDestroy_v2")


if __name__ == "__main__":
    unittest.main()
