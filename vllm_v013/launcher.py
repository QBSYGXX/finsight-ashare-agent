"""Logged, bounded subprocesses; isolate environments and export evidence even on failure."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from runtime import ADDON, BASE, ENV, ROOT, process_env, save, sha256, verify_package


def run_child(command, env, log, timeout):
    print("COMMAND=" + json.dumps(command), flush=True)
    started = time.monotonic()
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True, text=True, encoding="utf-8", errors="replace", bufsize=1)
    def relay():
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    reader = threading.Thread(target=relay, daemon=True)
    reader.start()
    try:
        while True:
            try:
                code = proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                print(f"PROCESS_RUNNING pid={proc.pid} elapsed_seconds={elapsed:.0f}", flush=True)
                log.write(f"PROCESS_RUNNING pid={proc.pid} elapsed_seconds={elapsed:.0f}\n")
                log.flush()
                if elapsed >= timeout:
                    raise TimeoutError(f"Stage timeout after {timeout}s; see last log lines")
    finally:
        # Only this launcher's own process group, never unrelated GPU processes.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        reader.join(timeout=10)
    print(f"CHILD_EXIT={code}", flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)


def comparison(output):
    rows = []
    fingerprints = set()
    for backend in ("hf", "vllm", "vllm-mtp"):
        data = json.loads((output / backend / "benchmark.json").read_text())
        trace = json.loads((output / backend / "trace.json").read_text())
        fingerprints.update(r["input_ids_sha256"] for r in data["samples"])
        rows.append({"backend": backend, "median_tokens_per_second": data["median_tokens_per_second"],
                     "financial_smoke_pass": trace["matches_deterministic_reference"]})
    if len(fingerprints) != 1:
        raise RuntimeError("Prompt token IDs differ across engines: speed comparison is invalid")
    for row in rows:
        row["speedup_vs_hf"] = row["median_tokens_per_second"] / rows[0]["median_tokens_per_second"]
    save(output / "comparison.json", {"rows": rows, "input_ids_sha256": next(iter(fingerprints)),
        "note": "Observed end-to-end engine improvement includes Torch/runtime changes. One validation question is not an accuracy benchmark. MTP is accepted only if its own smoke succeeds."})
    print("COMPARISON=" + json.dumps(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["setup", "smoke", "compare"])
    parser.add_argument("--gpu", type=int, choices=range(4), default=1)
    parser.add_argument("--backend", choices=["hf", "vllm", "vllm-mtp"], default="vllm")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{args.mode}-{os.getpid()}"
    if args.background:
        dest = ROOT / "runs/vllm_v013" / (stamp + "-launcher.log")
        dest.parent.mkdir(parents=True, exist_ok=True)
        command = [str(BASE / "bin/python"), str(ADDON / "launcher.py"), args.mode, "--gpu", str(args.gpu), "--backend", args.backend]
        with dest.open("w", encoding="utf-8") as stream:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=process_env(args.gpu, False))
        print(f"PID={proc.pid}\nLAUNCHER_LOG={dest}\n查看进度：tail -n 60 -f '{dest}'", flush=True)
        return
    output = ROOT / "outputs/vllm_v013" / stamp
    output.mkdir(parents=True)
    print(f"OUTPUT={output}\nLOG={output / 'run.log'}", flush=True)
    code = 1
    ready = ENV / "finsight-ready.json"
    try:
        manifest = verify_package()
        save(output / "request.json", {**vars(args), "commit": manifest["commit"], "manifest_sha256": sha256(ADDON / "PACKAGE_MANIFEST.json")})
        # Inventory is read-only; an occupied selected GPU is not killed.
        with (output / "run.log").open("w", encoding="utf-8", buffering=1) as log:
            run_child(["nvidia-smi"], process_env(args.gpu, False), log, 60)
            if args.mode == "setup":
                if ready.exists():
                    ready.unlink()
                run_child([str(BASE / "bin/python"), str(ADDON / "install.py"), str(args.gpu)], process_env(args.gpu, False), log, 2400)
                run_child([str(ENV / "bin/python"), str(ADDON / "worker.py"), "doctor", "--output", str(output / "doctor")], process_env(args.gpu), log, 600)
                save(ready, {"status": "doctor_passed", "lock_sha256": sha256(ADDON / "requirements.lock.txt"), "output": str(output)})
            else:
                if args.mode == "compare" or args.backend != "hf":
                    if not ready.exists() or json.loads(ready.read_text()).get("lock_sha256") != sha256(ADDON / "requirements.lock.txt"):
                        raise RuntimeError("Run setup successfully before vLLM execution")
                backends = ("hf", "vllm", "vllm-mtp") if args.mode == "compare" else (args.backend,)
                for backend in backends:
                    prefix = BASE if backend == "hf" else ENV
                    print("STAGE=" + backend, flush=True)
                    run_child([str(prefix / "bin/python"), str(ADDON / "worker.py"),
                               "bench" if args.mode == "compare" else "smoke", "--backend", backend, "--output", str(output / backend)],
                              process_env(args.gpu, backend != "hf"), log, 3600)
                if args.mode == "compare":
                    comparison(output)
        code = 0
    except Exception as error:
        save(output / "error.json", {"error": repr(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        save(output / "status.json", {"exit_code": code, "status": "completed" if code == 0 else "failed"})
        paths = [p for p in output.rglob("*") if p.is_file()]
        save(output / "FILES.json", {str(p.relative_to(output)): sha256(p) for p in paths})
        archive = ROOT / "exports" / ("finsight-vllm-v013-" + stamp + ".zip")
        archive.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in output.rglob("*"):
                if path.is_file():
                    z.write(path, path.relative_to(output))
        archive.with_suffix(".zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
        print(f"EXPORT={archive}\nEXIT={code}", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
