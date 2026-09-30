"""Build an isolated v0.10.1 derivative; the original v0.10 ZIP is immutable."""
import argparse
import ast
import hashlib
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
PREFIX = "thinking_v0101"
OUTPUT = RELEASE / "finsight-thinking-v0.10.1.zip"


# 模块：发布包派生；作用：校验基线并只注入共同语义检查；核心业务：否。
def payload():
    source = RELEASE / "finsight-thinking-v0.10.zip"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source.with_suffix(".zip.sha256").read_text().split()[0]
    files = {}
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        original = json.loads(z.read("thinking_v010/PACKAGE_MANIFEST.json"))
        for name, checksum in original["files"].items():
            b = z.read(name)
            assert hashlib.sha256(b).hexdigest() == checksum
            dest = name.replace("thinking_v010/", PREFIX + "/", 1)
            if name.endswith(("/run.py", "/run.sh")):
                b = b.replace(b"thinking_v010", b"thinking_v0101").replace(b"thinking-v010", b"thinking-v0101").replace(b"thinking-v0.10", b"thinking-v0.10.1")
            files[dest] = b
    planner = PREFIX + "/metrics/planner.py"
    code = files[planner].decode()
    guard = '''    from contract import ambiguous_ratio_yoy
    ambiguous = ambiguous_ratio_yoy(context["question"], metric_mentions(context["question"], registry), registry)
    if ambiguous:
        raise MetricError("needs_clarification", "比率指标的同比请明确为同比增长率，或同比变化多少个百分点",
                          {"guard": "ratio_yoy_semantics_v1", "metrics": ambiguous})
'''
    for signature in ("def rules_plan(context, registry):\n", "def model_plan(context, registry, model):\n"):
        assert code.count(signature) == 1
        code = code.replace(signature, signature + guard)
    files[planner] = code.encode()
    for file in (ROOT / PREFIX).iterdir():
        if file.suffix in {".py", ".md"}:
            files[PREFIX + "/" + file.name] = file.read_bytes()
    return files, hashlib.sha256(source.read_bytes()).hexdigest()


# 模块：离线验证；作用：在临时目录跑真实入口依赖与回归；核心业务：否。
def verify(files):
    with tempfile.TemporaryDirectory(prefix="finsight-contract-") as temp:
        target = Path(temp)
        for name, content in files.items():
            part = PurePosixPath(name)
            assert name.startswith(PREFIX + "/") and not part.is_absolute() and ".." not in part.parts
            if name.endswith(".py"):
                ast.parse(content)
            if name.endswith(".sh"):
                assert b"\r" not in content
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        shutil.copytree(ROOT / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__"))
        for old in ("finsight-evaluation-v0.7.zip", "finsight-chat-v0.7.1.zip"):
            with zipfile.ZipFile(RELEASE / old) as z:
                for name in z.namelist():
                    p = PurePosixPath(name)
                    assert not p.is_absolute() and ".." not in p.parts and "\\" not in name
                z.extractall(target)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "FINSIGHT_ROOT": str(target),
               "PYTHONPATH": os.pathsep.join(str(target / p) for p in (PREFIX, "src", "evaluation_v07", "chat_v071"))}
        for folder in (PREFIX, PREFIX + "/metrics"):
            result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", folder, "-p", "test_*.py"],
                                    cwd=target, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
            print(result.stdout[-1200:] + result.stderr[-4000:])
            print("EXIT=" + str(result.returncode))
            assert result.returncode == 0
    print("CONTRACT_AND_PINNED_PROTOCOL_TESTS=PASS; GPU_INFERENCE=NOT_RUN")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    files, source_hash = payload()
    if args.check:
        verify(files)
    else:
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit before publishing"
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        manifest = {"version": "0.10.1", "commit": revision, "base_archive_sha256": source_hash,
                    "files": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}
        files[PREFIX + "/PACKAGE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
        with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as z:
            for name, b in files.items():
                z.writestr(name, b)
        with zipfile.ZipFile(OUTPUT) as z:
            assert z.testzip() is None
            assert all(z.read(n) == b for n, b in files.items())
        digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
        OUTPUT.with_suffix(".zip.sha256").write_bytes(f"{digest}  {OUTPUT.name}\n".encode())
        verify(files)
        print(f"PACKAGE={OUTPUT}\nSHA256={digest}\nCOMMIT={revision}")
