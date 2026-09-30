"""Offline read-only host inventory for choosing a separate vLLM runtime."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile

ADDON = Path(__file__).resolve().parent
ROOT = ADDON.parent


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def read_small(path, limit=12000):
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as stream:
            return stream.read(limit)
    except OSError as error:
        return {"unavailable": str(error)}


def command(argv, timeout=12):
    executable = shutil.which(argv[0])
    if not executable:
        print("CHECK="+argv[0]+" unavailable", flush=True)
        return {"available": False}
    print("CHECK_START="+" ".join(argv), flush=True)
    started = time.perf_counter()
    try:
        result = subprocess.run([executable, *argv[1:]], capture_output=True, text=True,
                                errors="replace", timeout=timeout)
        record = {"available": True, "executable": executable, "exit_code": result.returncode,
                  "stdout": result.stdout[-8000:], "stderr": result.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        record = {"available": True, "executable": executable, "timeout_seconds": timeout}
    except OSError as error:
        record = {"available": True, "error": str(error)}
    record["seconds"] = round(time.perf_counter()-started, 3)
    print("CHECK_END="+argv[0]+" "+json.dumps(record, ensure_ascii=False), flush=True)
    return record


def inspect_model(directory):
    config_path = directory / "config.json"
    index_path = directory / "model.safetensors.index.json"
    result = {"path": str(directory), "exists": directory.is_dir()}
    if not config_path.exists() or not index_path.exists():
        result["status"] = "missing_metadata"
        return result
    config_bytes, index_bytes = config_path.read_bytes(), index_path.read_bytes()
    config, index = json.loads(config_bytes), json.loads(index_bytes)
    text = config.get("text_config", config)
    mtp = {k:v for k,v in index["weight_map"].items() if k.startswith("mtp.")}
    shards = []
    for name in sorted(set(index["weight_map"].values())):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe model index shard path")
        file = directory / name
        shards.append({"file": name, "exists": file.is_file(), "bytes": file.stat().st_size if file.is_file() else None})
    result.update(config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        index_sha256=hashlib.sha256(index_bytes).hexdigest(), model_type=config.get("model_type"),
        architectures=config.get("architectures"), mtp_num_hidden_layers=text.get("mtp_num_hidden_layers"),
        mtp_tensor_count=len(mtp), mtp_shards=sorted(set(mtp.values())), shards=shards,
        note="Only metadata, index and shard existence/size inspected; tensor contents not loaded or verified")
    return result


def collect(root):
    print("STAGE=host_inventory; no model loading or installation", flush=True)
    versions = {}
    for name in ("torch", "transformers", "triton", "vllm", "flashinfer-python", "flash-linear-attention"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    disk = shutil.disk_usage(root)
    result = {"timestamp": datetime.now().astimezone().isoformat(), "platform": platform.system(),
        "machine": platform.machine(), "kernel": platform.release(), "glibc": platform.libc_ver(),
        "os_release": read_small("/etc/os-release"), "python": sys.version, "python_executable": sys.executable,
        "versions": versions, "cpu_count": os.cpu_count(), "memory": read_small("/proc/meminfo", 2500),
        "disk": {"path": str(root), "free_bytes": disk.free, "total_bytes": disk.total},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "commands": {}}
    checks = {
        "gpu": ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap", "--format=csv,noheader"],
        "nvidia_smi": ["nvidia-smi"], "nvcc": ["nvcc", "--version"],
        "gcc": ["gcc", "--version"], "gxx": ["g++", "--version"], "cmake": ["cmake", "--version"],
        "docker_version": ["docker", "--version"],
        "docker_daemon": ["docker", "info", "--format", "{{json .ServerVersion}}"],
        "apptainer": ["apptainer", "--version"], "singularity": ["singularity", "--version"],
        "podman": ["podman", "--version"], "filesystem": ["findmnt", "-T", str(root), "-o", "FSTYPE,OPTIONS", "-n"],
        "module_tool": ["bash", "-lc", "type module >/dev/null 2>&1 && module -t avail cuda apptainer singularity 2>&1"],
    }
    for name, argv in checks.items():
        result["commands"][name] = command(argv)
    result["user_namespaces"] = {name: read_small(name, 500) for name in
        ("/proc/sys/user/max_user_namespaces", "/proc/sys/kernel/unprivileged_userns_clone")}
    candidates = {Path("/usr/local/cuda/compat"), Path("/usr/local/cuda/lib64"),
                  Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/lib64")}
    candidates.update(Path("/usr/local").glob("cuda*/compat"))
    for key in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(key)
        if value:
            candidates.update((Path(value)/"compat", Path(value)/"lib64"))
    candidates.update(Path(x) for x in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if x)
    libraries = []
    for directory in sorted(candidates, key=str):
        try:
            for file in sorted(directory.glob("libcuda.so*"))[:12]:
                libraries.append({"path": str(file), "resolved": str(file.resolve())})
        except OSError:
            pass
    result["libcuda_candidates"] = libraries
    result["model"] = inspect_model(root / "models/Qwen3.5-9B")
    result["routing_note"] = ("Availability is not deployment validation. Container executables do not prove GPU access; "
        "libcuda file discovery does not prove driver compatibility. Select and pin dependencies after review.")
    return result


def verify_package():
    manifest = json.loads((ADDON / "PACKAGE_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ADDON) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Package hash mismatch: "+name)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    output = ROOT / "outputs/vllm_probe_v011" / Path(args.run_dir).name
    finished = threading.Event()
    started = time.perf_counter()
    def heartbeat():
        while not finished.wait(20):
            print(f"DOCTOR_RUNNING elapsed_seconds={time.perf_counter()-started:.0f}", flush=True)
    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    code = 1
    try:
        manifest = verify_package()
        report = collect(ROOT)
        report["package_commit"] = manifest["commit"]
        report["status"] = "diagnostics_complete"
        save(output/"doctor.json", report)
        code = 0
    except Exception as error:
        save(output/"error.json", {"error": repr(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        finished.set()
        monitor.join()
        save(output/"status.json", {"exit_code": code, "vllm_installed_by_probe": False,
             "status": "diagnostics_complete" if code == 0 else "failed"})
        print(f"EXIT={code}; diagnostic collection only, not vLLM readiness", flush=True)
        dest = ROOT/"exports"/("finsight-vllm-doctor-"+output.name+".zip")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            for file in output.glob("*.json"):
                z.write(file, file.name)
            log = Path(args.run_dir)/"run.log"
            if log.exists():
                z.write(log, "run.log")
        checksum = hashlib.sha256(dest.read_bytes()).hexdigest()
        dest.with_suffix(".zip.sha256").write_bytes(f"{checksum}  {dest.name}\n".encode())
        print(f"RESULTS={dest}\nSHA256={checksum}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
