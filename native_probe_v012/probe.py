"""Offline user-local compatibility probe with bounded child processes."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
import zipfile

ADDON = Path(__file__).resolve().parent
ROOT = ADDON.parent


# 模块：包完整性；作用：验证二进制与相对链接，拒绝损坏或越界；核心业务：否。
def verify_payload(addon=ADDON):
    manifest = json.loads((addon / "PACKAGE_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = (addon / name).resolve()
        if not path.is_relative_to(addon):
            raise ValueError("Payload mismatch: " + name)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Payload mismatch: " + name)
    for name in alias_order(manifest["links"]):
        target = manifest["links"][name]
        path = addon / "payload/compat" / name
        dest = path.parent / target
        if Path(name).name != name or Path(target).name != target or not dest.is_file():
            raise ValueError("Invalid relative library alias")
        if path.is_symlink():
            if os.readlink(path) != target:
                raise ValueError("Existing library alias differs: " + name)
        elif path.exists():
            raise ValueError("Refusing to replace existing file: " + name)
        else:
            path.symlink_to(target)
    return manifest


def alias_order(links):
    ordered, visiting = [], set()
    def visit(name):
        if name in ordered:
            return
        if name in visiting:
            raise ValueError("Cyclic library alias")
        target = links[name]
        if Path(name).name != name or Path(target).name != target:
            raise ValueError("Invalid relative library alias")
        visiting.add(name)
        if target in links:
            visit(target)
        visiting.remove(name)
        ordered.append(name)
    for name in links:
        visit(name)
    return ordered


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def gpu_bus(text, index):
    rows = [line.split(",") for line in text.splitlines() if line.strip()]
    for row in rows:
        if len(row) >= 3 and row[0].strip() == str(index):
            # NVIDIA reports 8-digit PCI domains; the driver accepts 4-digit domains.
            raw = row[1].strip()
            domain, bus, slot = raw.split(":")
            return f"{int(domain,16):04x}:{bus}:{slot}", ",".join(row[2:]).strip()
    raise ValueError("Selected physical GPU not found")


# 模块：诊断编排；作用：对子进程单独设置库路径，不修改已安装环境；核心业务：否。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    out = ROOT / "outputs/native_probe_v012" / args.run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    report = {"platform": platform.platform(), "glibc": platform.libc_ver(), "python": sys.version,
              "gpu_index": args.gpu, "checks": {}, "status": "running",
              "note": "Native dependency and PTX gates only; no vLLM installation, model inference or speed claim."}
    stop = threading.Event()
    phase = ["verify_payload", time.monotonic()]
    def heartbeat():
        while not stop.wait(20):
            print(f"PROBE_RUNNING phase={phase[0]} seconds={time.monotonic()-phase[1]:.0f}", flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    code = 1
    def run(name, command, env, timeout=180):
        phase[:] = [name, time.monotonic()]
        print("CHECK_START=" + name, flush=True)
        try:
            p = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
            record = {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
            parsed = [s[len("PROBE_RESULT="):] for s in p.stdout.splitlines() if s.startswith("PROBE_RESULT=")]
            if parsed:
                record["result"] = json.loads(parsed[-1])
        except subprocess.TimeoutExpired as error:
            record = {"exit_code": None, "timeout": timeout, "stdout": str(error.stdout or ""), "stderr": str(error.stderr or "")}
        report["checks"][name] = record
        save(out / (name + ".json"), record)
        print(record["stdout"][-8000:] + record["stderr"][-4000:], flush=True)
        print(f"CHECK_END={name} EXIT={record['exit_code']}", flush=True)
        return record
    try:
        if sys.platform != "linux" or platform.machine() != "x86_64":
            raise RuntimeError("Linux x86_64 required")
        manifest = verify_payload()
        report["package_commit"] = manifest["commit"]
        env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
        inventory = run("gpu_inventory", ["nvidia-smi", "--query-gpu=index,pci.bus_id,name,driver_version,memory.used", "--format=csv,noheader"], env, 20)
        if inventory["exit_code"] != 0:
            raise RuntimeError("GPU inventory failed")
        bus, detail = gpu_bus(inventory["stdout"], args.gpu)
        report["selected_gpu"] = {"pci_bus_id": bus, "description": detail}
        base = ROOT / ".runtime/envs/finsight"
        library_path = str(base / "lib")
        env["LD_LIBRARY_PATH"] = library_path
        env["PYTHONPATH"] = str(ADDON / "payload/site-packages")
        worker = str(ADDON / "worker.py")
        llg = run("llguidance", [sys.executable, "-B", worker, "llguidance"], env)
        # The control can fail on PTX8.8 with the old system driver; record it,
        # but require the isolated compatibility-library run to succeed.
        run("system_ptx_control", [sys.executable, "-B", worker, "cuda", bus, "/usr/lib64/libcuda.so.1"], env)
        compat_env = {**env, "LD_LIBRARY_PATH": str(ADDON / "payload/compat") + ":" + library_path}
        compat = run("compat_ptx", [sys.executable, "-B", worker, "cuda", bus,
            str(ADDON / "payload/compat/libcuda.so.575.57.08")], compat_env)
        passed = all(r["exit_code"] == 0 and r.get("result", {}).get("passed") is True for r in (llg, compat))
        report["status"] = "native_gates_passed" if passed else "native_gates_failed"
        code = 0 if passed else 1
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        stop.set()
        thread.join()
        report["exit_code"] = code
        save(out / "report.json", report)
        export = ROOT / "exports" / ("finsight-native-probe-" + args.run_dir.name + ".zip")
        export.parent.mkdir(parents=True, exist_ok=True)
        print(f"STATUS={report['status']} EXIT={code}", flush=True)
        with zipfile.ZipFile(export, "w", zipfile.ZIP_DEFLATED) as z:
            for path in out.iterdir():
                if path.is_file():
                    z.write(path, path.name)
            log = args.run_dir / "run.log"
            if log.exists():
                z.write(log, "run.log")
        checksum = hashlib.sha256(export.read_bytes()).hexdigest()
        export.with_suffix(".zip.sha256").write_bytes(f"{checksum}  {export.name}\n".encode())
        print(f"RESULTS={export}\nSHA256={checksum}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
