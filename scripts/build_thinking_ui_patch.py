"""Derive isolated v0.10.2 from the released v0.10.1, preserving inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import build_thinking_contract_patch as checks

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent / "_release_work"
PREFIX = "thinking_v0102"


def build(check_only=False):
    source = RELEASE / "finsight-thinking-v0.10.1.zip"
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha == source.with_suffix(".zip.sha256").read_text().split()[0]
    files = {}
    with zipfile.ZipFile(source) as z:
        manifest = json.loads(z.read("thinking_v0101/PACKAGE_MANIFEST.json"))
        for n, h in manifest["files"].items():
            b = z.read(n)
            assert hashlib.sha256(b).hexdigest() == h
            dest = n.replace("thinking_v0101/", PREFIX + "/", 1)
            if n.endswith(("/run.py", "/run.sh", "/README.md")):
                b = b.replace(b"thinking_v0101", b"thinking_v0102").replace(b"thinking-v0101", b"thinking-v0102").replace(b"0.10.1", b"0.10.2")
            files[dest] = b
    key = PREFIX + "/run.py"
    code = files[key].decode()
    start = code.index("    finished = threading.Event()\n")
    end = code.index('    print("STAGE=verify_package_and_import_runtime"', start)
    code = code[:start] + '    from progress import Progress\n    monitor = Progress()\n    monitor.start()\n' + code[end:]
    code = code.replace('question = input("财务指标> ").strip()', 'question = monitor.read_question()')
    code = code.replace('        finished.set()\n        monitor.join()', '        monitor.stop()')
    code = code.replace('            result = execute_question(question, engine, resolver, "model", model)',
                        '            monitor.set_phase("plan_and_execute")\n            result = execute_question(question, engine, resolver, "model", model)')
    anchor = '                    holder[0] = BudgetPlanner(ROOT, think_budget=args.think_budget,'
    code = code.replace(anchor, '                    monitor.set_phase("load_model")\n                    load_started = time.perf_counter()\n' + anchor)
    anchor = '                holder[0].policy = self.policy'
    code = code.replace(anchor, '                    print(f"MODEL_LOAD_COMPLETE seconds={time.perf_counter()-load_started:.3f}", flush=True)\n                monitor.set_phase("generate_plan")\n' + anchor)
    assert 'input("财务指标>' not in code and 'finished.set()' not in code
    files[key] = code.encode()
    for p in (ROOT / PREFIX).iterdir():
        if p.suffix in {".py", ".md"}:
            files[PREFIX + "/" + p.name] = p.read_bytes()
    checks.PREFIX = PREFIX
    checks.verify(files)
    if check_only:
        return
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files[PREFIX + "/PACKAGE_MANIFEST.json"] = (json.dumps({"version": "0.10.2", "commit": commit,
        "base_archive_sha256": sha, "files": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}, indent=2) + "\n").encode()
    output = RELEASE / "finsight-thinking-v0.10.2.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None and all(z.read(n) == b for n, b in files.items())
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_bytes(f"{sha}  {output.name}\n".encode())
    print(f"PACKAGE={output}\nSHA256={sha}\nCOMMIT={commit}")


if __name__ == "__main__":
    build("--check" in sys.argv)
