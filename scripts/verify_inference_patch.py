"""Check archive provenance and run its CPU tests in a fresh installed layout."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent / "_release_work"


def main():
    package = RELEASE / "finsight-inference-v0.9.1.zip"
    checksum = package.with_suffix(".zip.sha256").read_bytes()
    assert b"\r" not in checksum
    assert checksum.decode().split()[0] == hashlib.sha256(package.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="finsight-speed-verify-") as temporary:
        target = Path(temporary)
        shutil.copytree(ROOT / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__"))
        with zipfile.ZipFile(package) as archive:
            assert archive.testzip() is None
            manifest = json.loads(archive.read("inference_v091/PACKAGE_MANIFEST.json"))
            assert set(archive.namelist()) == {*manifest["files"], "inference_v091/PACKAGE_MANIFEST.json"}
            for name, expected in manifest["files"].items():
                path = PurePosixPath(name)
                assert name.startswith("inference_v091/") and not path.is_absolute() and ".." not in path.parts and "\\" not in name
                data = archive.read(name)
                assert hashlib.sha256(data).hexdigest() == expected
                if name.endswith(".py"):
                    ast.parse(data, filename=name)
                if name.endswith(".sh"):
                    assert b"\r" not in data
            lock = json.loads(archive.read("inference_v091/dependencies.json"))
            for item in lock["wheels"]:
                data = archive.read("inference_v091/wheels/" + item["file"])
                assert hashlib.sha256(data).hexdigest() == item["sha256"]
                with zipfile.ZipFile(io.BytesIO(data)) as wheel:
                    assert not any(n.endswith((".so", ".dll", ".pyd")) for n in wheel.namelist())
            with zipfile.ZipFile(RELEASE / "finsight-metrics-v0.8.4.zip") as original:
                for name in archive.namelist():
                    if name.startswith("inference_v091/metrics/"):
                        assert archive.read(name) == original.read("metrics_v084/" + PurePosixPath(name).name)
            archive.extractall(target)
        env = {**os.environ, "FINSIGHT_ROOT": str(target), "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        for command in ([sys.executable, "-X", "utf8", "-B", "-m", "unittest", "discover", "-s", "inference_v091", "-v"],
                        [sys.executable, "-X", "utf8", "-B", "inference_v091/driver.py", "--help"],
                        [sys.executable, "-X", "utf8", "-B", "inference_v091/worker.py", "--help"]):
            result = subprocess.run(command, cwd=target, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
            print(result.stdout[-800:] + result.stderr[-1200:])
            print("EXIT=" + str(result.returncode))
            assert result.returncode == 0
    print("HASH_MANIFEST_PINNED_PLANNER_PURE_WHEELS_ISOLATED_TESTS=PASS; GPU_SPEEDUP=NOT_MEASURED")


if __name__ == "__main__":
    main()
