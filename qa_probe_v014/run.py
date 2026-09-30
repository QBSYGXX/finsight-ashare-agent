"""Launch the diagnostic suite offline, with logs, heartbeat and automatic export."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile
from support import ADDON, ROOT, digest, save, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=range(4), type=int, default=1)
    parser.add_argument("--backend", choices=["vllm", "vllm-mtp"], default="vllm-mtp")
    parser.add_argument("--ids", default="", help="Comma-separated case IDs, e.g. Q01,Q07,Q23")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    from cases import CASES
    if args.list:
        for case in CASES:
            print(f"{case['id']} [{case['category']}] {case['question']}")
        return
    selected = set(args.ids.split(",")) if args.ids else {c["id"] for c in CASES}
    if selected - {c["id"] for c in CASES}:
        parser.error("Unknown IDs: " + str(selected))
    from runtime import BASE, ENV, process_env, sha256
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{os.getpid()}"
    if args.background:
        log = ROOT / "runs/qa_probe_v014" / (stamp + "-launcher.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [str(BASE / "bin/python"), str(ADDON / "run.py"), "--gpu", str(args.gpu), "--backend", args.backend]
        if args.ids:
            command.extend(["--ids", args.ids])
        with log.open("w", encoding="utf-8") as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=process_env(args.gpu, False))
        print(f"PID={child.pid}\nLOG={log}\n查看进度：tail -n 60 -f '{log}'", flush=True)
        return
    output = ROOT / "outputs/qa_probe_v014" / stamp
    output.mkdir(parents=True)
    code = 1
    try:
        manifest = verify()
        ready = json.loads((ENV / "finsight-ready.json").read_text())
        if ready.get("lock_sha256") != sha256(ROOT / "vllm_v013/requirements.lock.txt"):
            raise RuntimeError("Matching v0.13 setup required")
        save(output / "request.json", {**vars(args), "source_commit": manifest["commit"],
             "probe_manifest_sha256": digest(ADDON / "PACKAGE_MANIFEST.json"), "v013_manifest_sha256": manifest["v013_manifest_sha256"],
             "cases_sha256": digest(ADDON / "cases.py"), "references_sha256": digest(ADDON / "references.json"),
             "selected_ids": sorted(selected), "suite_kind": "validation-company diagnostics; no independent test companies"})
        from launcher import run_child
        command = [str(ENV / "bin/python"), str(ADDON / "worker.py"), "--output", str(output), "--backend", args.backend]
        if args.ids:
            command.extend(["--ids", args.ids])
        print(f"STAGE=run_suite CASES={len(selected)} BACKEND={args.backend} GPU={args.gpu}\nOUTPUT={output}", flush=True)
        with (output / "run.log").open("w", encoding="utf-8", buffering=1) as log:
            run_child(command, process_env(args.gpu), log, 7200)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if summary["not_run"]:
            raise RuntimeError("Suite exited with cases not run")
        print(f"SUITE_EXECUTED={summary['executed']} PASSED={summary['passed']} FAILED={summary['failed']}", flush=True)
        code = 0
    except Exception as error:
        save(output / "error.json", {"error": repr(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        save(output / "status.json", {"status": "completed" if code == 0 else "failed", "exit_code": code,
                                     "note": "EXIT=0 means suite executed, NOT all questions passed; inspect summary.json."})
        save(output / "FILES.json", {p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*") if p.is_file()})
        archive = ROOT / "exports" / ("finsight-qa-v014-" + stamp + ".zip")
        archive.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in output.rglob("*"):
                if path.is_file():
                    z.write(path, path.relative_to(output))
        checksum = digest(archive)
        archive.with_suffix(".zip.sha256").write_bytes(f"{checksum}  {archive.name}\n".encode())
        print(f"EXPORT={archive}\nSHA256={checksum}\nEXIT={code}", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
