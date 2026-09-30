"""Build and verify a small dependency-free Linux diagnostic handoff."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PREFIX="vllm_probe_v011"
RELEASE=ROOT.parent/"_release_work"
PACKAGE=RELEASE/"finsight-vllm-probe-v0.11.zip"


def build():
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        raise RuntimeError("Commit first")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    payload={PREFIX+"/"+p.name:p.read_bytes() for p in (ROOT/PREFIX).iterdir() if p.suffix in {".py",".md",".sh"}}
    manifest={"version":"0.11","commit":commit,"files":{n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}}
    payload[PREFIX+"/PACKAGE_MANIFEST.json"]=(json.dumps(manifest,indent=2)+"\n").encode()
    with zipfile.ZipFile(PACKAGE,"w",zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
    h=hashlib.sha256(PACKAGE.read_bytes()).hexdigest()
    PACKAGE.with_suffix(".zip.sha256").write_bytes(f"{h}  {PACKAGE.name}\n".encode())
    print(f"PACKAGE={PACKAGE}\nBYTES={PACKAGE.stat().st_size}\nSHA256={h}\nCOMMIT={commit}")


def verify():
    checksum=PACKAGE.with_suffix(".zip.sha256").read_bytes()
    assert b"\r" not in checksum and checksum.decode().split()[0]==hashlib.sha256(PACKAGE.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="vllm-probe-verify-") as tmp:
        with zipfile.ZipFile(PACKAGE) as z:
            assert z.testzip() is None
            manifest=json.loads(z.read(PREFIX+"/PACKAGE_MANIFEST.json"))
            assert set(z.namelist())=={*manifest["files"],PREFIX+"/PACKAGE_MANIFEST.json"}
            for n,h in manifest["files"].items():
                path=PurePosixPath(n)
                assert n.startswith(PREFIX+"/") and not path.is_absolute() and ".." not in path.parts and "\\" not in n
                b=z.read(n)
                assert hashlib.sha256(b).hexdigest()==h
                if n.endswith(".py"):ast.parse(b)
                if n.endswith(".sh"):assert b"\r" not in b
            z.extractall(tmp)
        env={**os.environ,"PYTHONUTF8":"1","PYTHONDONTWRITEBYTECODE":"1"}
        for cmd in ([sys.executable,"-B","-m","unittest","discover","-s",PREFIX,"-v"],
                    [sys.executable,"-B",PREFIX+"/doctor.py","--help"]):
            result=subprocess.run(cmd,cwd=tmp,env=env,capture_output=True,text=True,encoding="utf-8",timeout=30)
            print(result.stdout+result.stderr)
            print("EXIT="+str(result.returncode))
            assert result.returncode==0
    print("HASH_MANIFEST_ISOLATED_TESTS=PASS; SERVER_DIAGNOSTICS=NOT_RUN")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true")
    verify() if parser.parse_args().verify else build()
