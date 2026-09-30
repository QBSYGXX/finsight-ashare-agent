"""Build isolated upload package with LF SHA256 sidecar and pinned existing dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_v015.storage import save,sha,sha_lf


def main():
    target=ROOT.parent/"_release_work/finsight-agent-v0.15.5.zip"
    if target.exists() or target.with_suffix(".zip.sha256").exists():
        raise FileExistsError("Release artifact already exists; do not overwrite a published package")
    addon=ROOT/"agent_v015"
    files=sorted(p for p in addon.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                 and p.name!="PACKAGE_MANIFEST.json" and p.suffix in {".py",".sh",".md",".json"})
    names=["data/verified/facts.json","data/processed/pages.jsonl","data/manifest.json","configs/project.json",
           "vllm_v013/runtime.py","vllm_v013/adapter.py"]
    pin={"protocol":"agent-v015-1.5","base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
         "files":{p.relative_to(ROOT).as_posix():sha(p) for p in files},
         "dependencies":{name:sha(ROOT/name) for name in names},
         "dependency_lf_sha256":{name:sha_lf(ROOT/name) for name in ["data/verified/facts.json","data/manifest.json"]}}
    save(addon/"PACKAGE_MANIFEST.json",pin)
    with zipfile.ZipFile(target,"x",zipfile.ZIP_DEFLATED) as z:
        for p in files+[addon/"PACKAGE_MANIFEST.json"]:
            z.write(p,p.relative_to(ROOT).as_posix())
    target.with_suffix(".zip.sha256").write_text(sha(target)+"  "+target.name+"\n",encoding="ascii",newline="\n")
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(pin["files"])|{"agent_v015/PACKAGE_MANIFEST.json"}
        import hashlib
        assert all(hashlib.sha256(z.read(name)).hexdigest()==value for name,value in pin["files"].items())
    print(f"ZIP={target}\nBYTES={target.stat().st_size}\nSHA256={sha(target)}")


if __name__=="__main__":main()
