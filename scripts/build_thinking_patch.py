"""Build or independently verify the isolated, offline thinking-budget add-on."""
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
PREFIX = "thinking_v010"
PACKAGE = RELEASE / "finsight-thinking-v0.10.zip"


def build():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit changes before packaging")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = {PREFIX+"/"+p.name: p.read_bytes() for p in (ROOT / PREFIX).iterdir()
             if p.is_file() and p.suffix in {".py", ".md", ".sh"}}
    with zipfile.ZipFile(RELEASE / "finsight-metrics-v0.8.4.zip") as z:
        original = json.loads(z.read("metrics_v084/PACKAGE_MANIFEST.json"))
        for name, checksum in original["files"].items():
            content = z.read(name)
            assert hashlib.sha256(content).hexdigest() == checksum
            files[PREFIX+"/metrics/"+Path(name).name] = content
    manifest = {"version": "0.10", "commit": commit, "planner_version": "0.8.4",
                "files": {n: hashlib.sha256(b).hexdigest() for n,b in files.items()}}
    files[PREFIX+"/PACKAGE_MANIFEST.json"] = (json.dumps(manifest,indent=2)+"\n").encode()
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    checksum = hashlib.sha256(PACKAGE.read_bytes()).hexdigest()
    PACKAGE.with_suffix(".zip.sha256").write_bytes(f"{checksum}  {PACKAGE.name}\n".encode())
    print(f"PACKAGE={PACKAGE}\nBYTES={PACKAGE.stat().st_size}\nSHA256={checksum}\nCOMMIT={commit}")


def verify():
    checksum = PACKAGE.with_suffix(".zip.sha256").read_bytes()
    assert b"\r" not in checksum
    assert hashlib.sha256(PACKAGE.read_bytes()).hexdigest() == checksum.decode().split()[0]
    with tempfile.TemporaryDirectory(prefix="thinking-verify-") as tmp:
        target = Path(tmp)
        with zipfile.ZipFile(PACKAGE) as z, zipfile.ZipFile(RELEASE / "finsight-metrics-v0.8.4.zip") as old:
            assert z.testzip() is None
            manifest = json.loads(z.read(PREFIX+"/PACKAGE_MANIFEST.json"))
            assert set(z.namelist()) == {*manifest["files"], PREFIX+"/PACKAGE_MANIFEST.json"}
            for n,h in manifest["files"].items():
                path = PurePosixPath(n)
                assert n.startswith(PREFIX+"/") and not path.is_absolute() and ".." not in path.parts and "\\" not in n
                b = z.read(n)
                assert hashlib.sha256(b).hexdigest() == h
                if n.endswith(".py"):
                    ast.parse(b)
                if n.endswith(".sh"):
                    assert b"\r" not in b
                if n.startswith(PREFIX+"/metrics/"):
                    assert b == old.read("metrics_v084/"+path.name)
            z.extractall(target)
        shutil.copytree(ROOT/"src",target/"src",ignore=shutil.ignore_patterns("__pycache__"))
        for archive, directory in (("finsight-evaluation-v0.7.zip", "evaluation_v07"), ("finsight-chat-v0.7.1.zip", "chat_v071")):
            with zipfile.ZipFile(RELEASE/archive) as z:
                for n in z.namelist():
                    path = PurePosixPath(n)
                    assert not path.is_absolute() and ".." not in path.parts and "\\" not in n
                z.extractall(target)
        env={**os.environ,"PYTHONUTF8":"1","PYTHONDONTWRITEBYTECODE":"1", "FINSIGHT_ROOT":str(target), "PYTHONPATH":os.pathsep.join(str(target/x) for x in ("src", "evaluation_v07", "chat_v071"))}
        for command in ([sys.executable,"-B","-m","unittest","discover","-s",PREFIX,"-p","test_*.py","-v"],
                        [sys.executable,"-B","-m","unittest","discover","-s",PREFIX+"/metrics","-p","test_*.py"],
                        [sys.executable,"-B",PREFIX+"/run.py","--help"]):
            run=subprocess.run(command,cwd=target,env=env,capture_output=True,text=True,encoding="utf-8",timeout=60)
            print(run.stdout[-1500:]+run.stderr[-3500:])
            print("EXIT="+str(run.returncode))
            assert run.returncode==0
    print("HASH_MANIFEST_PINNED_PLANNER_PROTOCOL_TESTS=PASS; GPU_INFERENCE=NOT_RUN")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--verify",action="store_true")
    args=p.parse_args()
    verify() if args.verify else build()
