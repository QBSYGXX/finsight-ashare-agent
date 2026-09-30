"""Offline overlay installation, supervised comparison and automatic result export."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile

from runtime_check import save, validate_profile

ADDON = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FINSIGHT_ROOT", ADDON.parent)).resolve()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def supervise(command, env):
    print("RUN=" + " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(list(map(str, command)), env=env, cwd=ROOT)
    started = time.monotonic()
    try:
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f"PROCESS_RUNNING pid={process.pid} elapsed={time.monotonic() - started:.0f}s", flush=True)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    print(f"PROCESS_EXIT={code} elapsed={time.monotonic() - started:.1f}s", flush=True)
    if code:
        raise RuntimeError(f"Subprocess failed with exit code {code}")


def verify_package():
    manifest = json.loads((ADDON / "PACKAGE_MANIFEST.json").read_text())
    for name, checksum in manifest["files"].items():
        path = ROOT / name
        if not path.resolve().is_relative_to(ADDON) or digest(path) != checksum:
            raise RuntimeError("Package hash mismatch: " + name)
    return manifest


def kernel_environment(overlay):
    return {**os.environ, "PYTHONPATH": str(overlay) + os.pathsep + str(ROOT / "src"),
            "TRITON_CACHE_DIR": str(ROOT / ".runtime/inference-v09/triton-cache"),
            "CUDA_CACHE_PATH": str(ROOT / ".runtime/inference-v09/cuda-cache")}


def install(output, gpu, profile):
    lock = json.loads((ADDON / "dependencies.json").read_text())
    validate_profile(profile, lock["profile"])
    wheel_digest = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    directory = ROOT / ".runtime/inference-v09"
    pointer = directory / "ready.json"
    if pointer.exists():
        ready = json.loads(pointer.read_text())
        overlay = Path(ready["overlay"]).resolve()
        if ready["lock_digest"] == wheel_digest and overlay.is_relative_to(directory.resolve()) and overlay.exists():
            print("REUSING_OVERLAY=" + str(overlay), flush=True)
            supervise([sys.executable, ADDON / "runtime_check.py", "probe", "--gpu", gpu,
                       "--output", output / "kernel_probe.json"], kernel_environment(overlay))
            return overlay
    overlay = directory / ("packages-" + uuid.uuid4().hex[:10])
    wheels = []
    for item in lock["wheels"]:
        path = ADDON / "wheels" / item["file"]
        if digest(path) != item["sha256"]:
            raise RuntimeError("Wheel hash mismatch")
        wheels.append(path)
    overlay.mkdir(parents=True)
    supervise([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
               "--target", overlay, *wheels], os.environ.copy())
    supervise([sys.executable, ADDON / "runtime_check.py", "probe", "--gpu", gpu,
               "--output", output / "kernel_probe.json"], kernel_environment(overlay))
    save(pointer, {"overlay": str(overlay), "lock_digest": wheel_digest, "status": "kernel_probe_passed"})
    return overlay


def summarize(output):
    records = {}
    for backend in ("baseline", "fast"):
        records[backend] = {kind: json.loads((output / backend / (kind + ".json")).read_text())
                            for kind in ("fixed_work", "trace", "runtime")}
    a, b = records["baseline"], records["fast"]
    if a["fixed_work"]["prompt_sha256"] != b["fixed_work"]["prompt_sha256"]:
        raise RuntimeError("Comparison prompt mismatch")
    def answer(trace):
        return [{k: row.get(k) for k in ("year", "metric", "transform", "status", "value", "unit", "citations")}
                for row in trace.get("results", [])]
    valid = all(r["trace"]["status"] == "completed" for r in (a, b))
    same = valid and a["trace"]["plan"]["requests"] == b["trace"]["plan"]["requests"] and answer(a["trace"]) == answer(b["trace"])
    return {"status": "comparison_complete", "same_completed_answer": same,
            "fixed_work_speedup": a["fixed_work"]["median_seconds"] / b["fixed_work"]["median_seconds"],
            "fixed_work_output_tokens": 128,
            "baseline": a, "fast": b,
            "note": "One diagnostic question; latency excludes load/JIT. Generated lengths may differ; no general accuracy claim."}


def export(output, run_dir):
    target = ROOT / "exports" / ("finsight-inference-" + output.name + ".zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob("*"):
            if path.is_file():
                archive.write(path, str(path.relative_to(output)))
        log = run_dir / "run.log"
        if log.exists():
            archive.write(log, "run.log")
    checksum = digest(target)
    target.with_suffix(".zip.sha256").write_bytes(f"{checksum}  {target.name}\n".encode())
    print(f"RESULTS={target}\nSHA256={checksum}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["doctor", "install", "benchmark", "chat"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--thinking", choices=["on", "off"], default="on")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--question", default="中兴通讯2024年毛利率、资产负债率")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 5 or not 64 <= args.max_new_tokens <= 8192 or not args.max_new_tokens < args.context_length <= 32768 or args.gpu < 0:
        parser.error("Invalid repeat, generation limit, context limit or GPU")
    run_dir = Path(args.run_dir).resolve()
    output = ROOT / "outputs" / ADDON.name / run_dir.name
    output.mkdir(parents=True, exist_ok=True)
    status, code = "failed", 1
    try:
        manifest = verify_package()
        save(output / "request.json", {**vars(args), "package_commit": manifest["commit"]})
        supervise([sys.executable, ADDON / "runtime_check.py", "doctor", "--output", output / "doctor.json"], os.environ.copy())
        profile = json.loads((output / "doctor.json").read_text())
        if args.command != "doctor":
            overlay = install(output, args.gpu, profile)
            if args.command in {"benchmark", "chat"}:
                backends = ("baseline", "fast") if args.command == "benchmark" else ("fast",)
                for backend in backends:
                    env = os.environ.copy() if backend == "baseline" else kernel_environment(overlay)
                    supervise([sys.executable, ADDON / "worker.py", args.command, "--backend", backend,
                               "--output", output / backend, "--gpu", args.gpu, "--repeat", args.repeat,
                               "--thinking", args.thinking, "--max-new-tokens", args.max_new_tokens,
                               "--context-length", args.context_length, "--question", args.question], env)
                if args.command == "benchmark":
                    summary = summarize(output)
                    save(output / "summary.json", summary)
                    print("FIXED_WORK_SPEEDUP=" + str(summary["fixed_work_speedup"]), flush=True)
                    print("SAME_COMPLETED_ANSWER=" + str(summary["same_completed_answer"]), flush=True)
        status, code = "completed", 0
    except Exception as error:
        save(output / "error.json", {"error": repr(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        save(output / "status.json", {"status": status, "exit_code": code})
        print(f"STATUS={status} EXIT={code}", flush=True)
        export(output, run_dir)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
