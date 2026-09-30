"""Capture reproducible local checks without installing/loading the 9B model."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight_ashare.common import now, source_digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-report", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "outputs/verification" / uuid.uuid4().hex[:8]
    directory.mkdir(parents=True)
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONPATH=str(ROOT / "src"))
    checks = []
    commands = [
        ("compile", [sys.executable, "-m", "compileall", "-q", "src", "server", "scripts", "tests"]),
        ("pytest", [sys.executable, "-m", "pytest", "-q", "--tb=short", "--basetemp", str(ROOT / ".cache" / ("pytest-"+uuid.uuid4().hex))]),
    ]
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/missing-bash")
    if bash.exists():
        commands.append(("bash_syntax", [str(bash), "-n", "server/run.sh"]))
    if args.with_report:
        commands.append(("real_pdf_seed_revalidation", [sys.executable, "-c", "from finsight_ashare.cli import bootstrap_example; bootstrap_example() "]))
        commands.append(("ten_real_fact_tool_replays", [sys.executable, "-m", "finsight_ashare", "smoke", "--replay", "--count", "10"]))
    for name, command in commands:
        print("CHECK="+name, flush=True)
        with (directory / (name+".log")).open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
            code = process.wait()
        checks.append({"name": name, "exit_code": code})
        if code:
            write_json(directory / "summary.json", {"status": "failed", "checks": checks})
            return code
    result = {"created": now(), "status": "passed", "source_digest": source_digest(ROOT), "checks": checks,
              "llm_executed": False, "server_training_executed": False, "real_report_replay": args.with_report}
    write_json(directory / "summary.json", result)
    print("VERIFICATION=" + str(directory / "summary.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
