"""Verify the upload archive and run its CLI in an isolated offline layout."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile

from build_metrics_patch import ROOT, RELEASE, PACKAGE_DIR


def main():
    package = RELEASE / "finsight-metrics-v0.8.5.zip"
    checksum_line = package.with_suffix(".zip.sha256").read_bytes()
    assert b"\r" not in checksum_line
    assert checksum_line.decode("ascii").split()[0] == hashlib.sha256(package.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="finsight-metrics-verify-") as directory:
        target = Path(directory)
        for folder in ("src", "configs"):
            shutil.copytree(ROOT / folder, target / folder, ignore=shutil.ignore_patterns("__pycache__"))
        (target / "data/verified").mkdir(parents=True)
        shutil.copy2(ROOT / "data/verified/facts.json", target / "data/verified/facts.json")
        snapshot = "models/Qwen3.5-9B"
        (target / snapshot).mkdir(parents=True)
        for name in ("tokenizer_config.json", "chat_template.jinja"):
            shutil.copy2(ROOT / snapshot / name, target / snapshot / name)
        for name in ("finsight-evaluation-v0.7.zip", "finsight-chat-v0.7.1.zip", package.name):
            with zipfile.ZipFile(RELEASE / name) as archive:
                assert archive.testzip() is None
                for member in archive.namelist():
                    path = PurePosixPath(member)
                    assert not path.is_absolute() and ".." not in path.parts and "\\" not in member and ":" not in member
                manifest_name = next(n for n in archive.namelist() if n.endswith("/PACKAGE_MANIFEST.json"))
                manifest = json.loads(archive.read(manifest_name))
                for member, checksum in manifest["files"].items():
                    assert hashlib.sha256(archive.read(member)).hexdigest() == checksum, member
                if name == package.name:
                    assert set(archive.namelist()) == {*manifest["files"], manifest_name}
                    assert all(n.startswith(PACKAGE_DIR + "/") for n in archive.namelist())
                    assert b"\r" not in archive.read(PACKAGE_DIR + "/run.sh")
                archive.extractall(target)
        env = {**os.environ, "FINSIGHT_ROOT": str(target), "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        commands = [
            [sys.executable, "-X", "utf8", "-B", "-m", "unittest", "discover", "-s", PACKAGE_DIR, "-p", "test_*.py"],
            [sys.executable, "-X", "utf8", "-B", f"{PACKAGE_DIR}/chat.py", "--question", "贵州茅台2024年毛利率、资产负债率"],
        ]
        for command in commands:
            run = subprocess.run(command, cwd=target, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
            print(run.stdout[-1200:], end="")
            print(run.stderr[-1200:], end="")
            print(f"EXIT={run.returncode}")
            assert run.returncode == 0
        trace_path = next((target / "outputs" / PACKAGE_DIR).glob("*/trace.json"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        assert trace["protocol"] == "metrics-v0.8.5" and trace["status"] == "completed"
        assert [r["value"] for r in trace["results"]] == ["91.9312", "19.0448"]
        report = trace_path.with_name("report.html").read_text(encoding="utf-8")
        assert "91.9312" in report and "19.0448" in report
    print("PACKAGE_HASH_MANIFEST_TESTS_CLI_REPORT=PASS; GPU_INFERENCE=NOT_RUN")


if __name__ == "__main__":
    main()
