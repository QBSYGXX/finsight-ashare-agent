"""Package a small regression suite against the exact released v0.13 protocol."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "qa_probe_v014"
RELEASE = ROOT.parent / "_release_work"


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_and_reference():
    with tempfile.TemporaryDirectory(prefix="finsight-qa-probe-") as temp:
        stage = Path(temp)
        for filename in ("finsight-vllm-v0.13.zip", "finsight-evaluation-v0.7.zip", "finsight-chat-v0.7.1.zip"):
            path = RELEASE / filename
            if path.with_suffix(".zip.sha256").exists():
                assert hash_file(path) == path.with_suffix(".zip.sha256").read_text(encoding="utf-8-sig").split()[0]
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    assert not Path(name).is_absolute() and ".." not in Path(name).parts and "\\" not in name
                z.extractall(stage)
        shutil.copytree(ADDON, stage / ADDON.name, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("data/verified/facts.json", "configs/project.json"):
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        code = '''
import sys,json
from pathlib import Path
sys.path.insert(0,"qa_probe_v014")
from support import ROOT,save
from cases import CASES
from scoring import canonical_rows,assess
from chat import verify_inputs,execute_question
from engine import MetricEngine
from resolver import CompanyResolver
facts,pin=verify_inputs()
engine=MetricEngine(facts)
resolver=CompanyResolver.load(ROOT,ROOT/"evaluation_v07")
assert len(CASES)==32 and len({c["id"] for c in CASES})==32
references={}
class Stub:
    last_generation=None
    calls=0
    def generate(self,messages):
        self.calls+=1
        return json.dumps({"requests":self.requests,"clarification":None})
for case in CASES:
    e=case["expect"]
    if e["kind"]=="task":
        assert e["company"] in {"000063","000333"}
        reference=engine.execute_years(e["company"],e["years"],e["scope"],e["requests"],e["money_unit"])
        references[case["id"]]={"kind":"task","status":reference["status"],"canonical_rows":canonical_rows(reference["results"])}
        stub=Stub();stub.requests=e["requests"]
        result=execute_question(case["question"],engine,resolver,"model",stub)
        verdict=assess(case,references[case["id"]],result,stub.calls)
        assert verdict["passed"],(case["id"],result,verdict)
    elif e["must_not_call_model"]:
        stub=Stub();stub.requests=[]
        result=execute_question(case["question"],engine,resolver,"model",stub)
        references[case["id"]]={"kind":"boundary","expected_statuses":e["statuses"]}
        verdict=assess(case,references[case["id"]],result,stub.calls)
        assert verdict["passed"],(case["id"],result,verdict)
    else:
        references[case["id"]]={"kind":"boundary","expected_statuses":e["statuses"]}
    print(case["id"],case["category"],references[case["id"]].get("status",e.get("statuses")),flush=True)
save(ROOT/"references.generated.json",references)
print("EXPLICIT_INTENTS_AND_ROOT_IMPORTS_PASS; REAL_MODEL_NOT_RUN")
save(ROOT/"qa_probe_v014/references.json",references)
# Exercise the actual worker end to end with an explicitly fake model in this
# temporary directory. These results are never included in the upload archive.
import importlib.util,unicodedata
from unittest.mock import patch
import adapter
spec=importlib.util.spec_from_file_location("qa_worker_under_test",ROOT/"qa_probe_v014/worker.py")
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
class FakeEngine:
    instances=0
    def __init__(self,backend):
        type(self).instances+=1
        self.load_seconds=0
    def generate(self,messages):
        question=messages[-1]["content"]
        case=next(c for c in CASES if unicodedata.normalize("NFKC",c["question"])==question)
        self.last_generation={"stop_reason":"eos","enable_thinking":True,"test_double":True}
        if case["id"]=="Q29":
            plan={"requests":[],"clarification":"该指标未登记"}
        else:
            plan={"requests":case["expect"]["requests"],"clarification":None}
        if case["id"]=="Q02":
            return "</think>not JSON"  # One intentional model failure must be preserved.
        return "</think>"+json.dumps(plan,ensure_ascii=False)
with patch.object(adapter,"Engine",FakeEngine),patch.object(worker,"verify",return_value={}),patch.object(sys,"argv",["worker","--output",str(ROOT/"mock-output")]):
    worker.main()
summary=json.loads((ROOT/"mock-output/summary.json").read_text(encoding="utf-8"))
assert summary["executed"]==32 and summary["passed"]==31 and summary["failed"]==1 and summary["not_run"]==0,summary
assert summary["groups"]["preflight_only"]["executed"]==10,summary
assert FakeEngine.instances==1
assert json.loads((ROOT/"mock-output/Q21/model_calls.json").read_text())==[]
assert (ROOT/"mock-output/report.html").exists()
print("WORKER_INTEGRATION_FAKE_MODEL=PASS; 31/32 IS SYNTHETIC AND NOT A MODEL RESULT")
'''
        checked = subprocess.run([sys.executable, "-c", code], cwd=stage, env=env,
                                 capture_output=True, text=True, encoding="utf-8", timeout=60)
        if checked.returncode:
            print(checked.stdout[-3000:] + checked.stderr)
            raise RuntimeError("Reference/worker integration validation failed")
        print("\n".join(line for line in checked.stdout.splitlines() if line.startswith(("Q", "EXPLICIT_", "WORKER_INTEGRATION_"))))
        data = (stage / "references.generated.json").read_bytes().replace(b"\r\n", b"\n")
        target = ADDON / "references.json"
        if target.exists():
            assert target.read_bytes() == data, "Reference changed; investigate rather than automatically accepting"
        else:
            target.write_bytes(data)
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "qa_probe_v014", "-p", "test_*.py"], cwd=stage, env=env, check=True)


def main():
    validate_and_reference()
    if "--check" in sys.argv:
        return
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit before packaging"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with zipfile.ZipFile(RELEASE / "finsight-vllm-v0.13.zip") as z:
        dependency = hashlib.sha256(z.read("vllm_v013/PACKAGE_MANIFEST.json")).hexdigest()
    files = {p.name:p.read_bytes() for p in ADDON.iterdir() if p.suffix in {".py", ".sh", ".md", ".json"}}
    manifest = {"version":"0.14-qa-probe", "commit":commit,"v013_manifest_sha256":dependency,
                "files":{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}}
    files["PACKAGE_MANIFEST.json"]=(json.dumps(manifest,indent=2)+"\n").encode()
    archive = RELEASE / "finsight-qa-v0.14.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():
            z.writestr("qa_probe_v014/"+name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(z.read("qa_probe_v014/"+n)==b for n,b in files.items())
    h=hash_file(archive)
    archive.with_suffix(".zip.sha256").write_bytes(f"{h}  {archive.name}\n".encode())
    print(f"PACKAGE={archive}\nSHA256={h}\nBYTES={archive.stat().st_size}\nCOMMIT={commit}")


if __name__ == "__main__":
    main()
