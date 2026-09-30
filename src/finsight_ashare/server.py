"""Server-only setup and training orchestration; no SSH operations."""
import csv
from copy import deepcopy
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .common import ROOT, config, digest, file_hash, now, read_json, source_digest, write_json


def run_command(command, root=ROOT, env=None):
    print("RUN " + " ".join(map(str, command)), flush=True)
    result = subprocess.run(list(map(str, command)), cwd=root, env=env, check=False)
    print(f"EXIT={result.returncode}", flush=True)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command[0]}")


def portable_python(root, profile):
    prefix = root / ".runtime/envs/finsight"
    if not prefix.exists():
        return None
    marker = prefix / ".finsight-ready.json"
    python = prefix / "bin/python"
    if not marker.is_file() or not python.is_file():
        raise ValueError("Portable environment is incomplete; run server/install_env.sh first")
    receipt = read_json(marker)
    if (receipt.get("status") != "ready" or receipt.get("profile") != profile
            or Path(receipt.get("prefix", "")).resolve() != prefix.resolve()):
        raise ValueError("Portable environment profile/path mismatch; do not move an unpacked environment")
    return python


# 模块：环境诊断；作用：不安装任何包即可生成可下载诊断；核心业务：否。
def cuda_probe(root, python):
    print("Testing BF16 forward/backward on every visible GPU...", flush=True)
    try:
        result = subprocess.run([str(python), str(root / "server/environment/probe_cuda.py")],
                                capture_output=True, text=True, timeout=180, check=True)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"passed": False, "error": str(error), "stderr": getattr(error, "stderr", None)}


def driver_compatible(driver, profile, probe):
    version = tuple(int(part) for part in driver.split("."))
    if version[0] >= profile["minimum_driver_major"]:
        return True
    # CUDA 12 minor-version compatibility still requires real kernel validation.
    return (version >= (525, 60, 13) and profile["torch_index"].endswith("/cu126")
            and probe.get("passed") is True and probe.get("cuda_runtime") == "12.6"
            and probe.get("gpu_count") == profile["gpus"]
            and len(probe.get("devices", [])) == profile["gpus"]
            and all(device.get("passed") is True for device in probe["devices"]))


def doctor(root=ROOT, offline=False):
    profile = read_json(root / "configs/server-profile.json")
    report = {"created": now(), "platform": platform.platform(), "python": sys.version,
              "glibc": platform.libc_ver(),
              "python_executable": sys.executable, "disk_free_gib": round(shutil.disk_usage(root).free / 1024**3, 2),
              "gpus": [], "connectivity": {}, "warnings": [], "errors": [], "profile": profile,
              "tools": {p: shutil.which(p) for p in ("python3.11", "python3.12", "python3.13", "git", "nvidia-smi", "unzip")}}
    packed_python = None
    report["offline"] = offline
    try:
        packed_python = portable_python(root, profile)
        report["portable_environment"] = str(packed_python) if packed_python else None
    except ValueError as error:
        report["errors"].append(str(error))
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=20, check=True)
        for row in csv.reader(result.stdout.splitlines()):
            report["gpus"].append(dict(zip(("index", "name", "memory_total_mib", "memory_free_mib", "driver"), [x.strip() for x in row])))
    except (OSError, subprocess.SubprocessError) as error:
        report["errors"].append("nvidia-smi failed: " + str(error))
    if len(report["gpus"]) < profile["gpus"]:
        report["errors"].append("Expected at least 4 visible GPUs")
    if platform.system() == "Linux" and packed_python:
        report["cuda_probe"] = cuda_probe(root, packed_python)
        if not report["cuda_probe"].get("passed"):
            report["errors"].append("Fresh CUDA/BF16 runtime probe failed")
    else:
        report["cuda_probe"] = {"passed": False, "reason": "Verified Linux portable environment unavailable"}
    for gpu in report["gpus"]:
        if float(gpu["memory_total_mib"]) < profile["minimum_gpu_memory_mib"]:
            report["errors"].append("GPU " + gpu["index"] + " has less than 38,000 MiB")
        if float(gpu["memory_free_mib"]) < 30000:
            report["warnings"].append("GPU " + gpu["index"] + " has less than 30,000 MiB free; check other jobs")
        if not driver_compatible(gpu["driver"], profile, report["cuda_probe"]):
            report["errors"].append("Driver compatibility not demonstrated by a fresh CUDA/BF16 probe")
        elif int(gpu["driver"].split(".")[0]) < profile["minimum_driver_major"]:
            report["warnings"].append("GPU " + gpu["index"] + ": CUDA minor-version compatibility accepted after BF16 probe; model/NCCL/20-step training checks remain required")
    if report["disk_free_gib"] < profile["minimum_free_disk_gib"]:
        report["errors"].append("Less than 80 GiB free disk space")
    if platform.system() != "Linux":
        report["errors"].append("Training/setup are intended for Linux; Windows doctor is informational")
    for name, url in {"github": "https://raw.githubusercontent.com/RUC-NLPIR/FinSight/8fc5bce28ac201c713219a47d30367c6a84342f4/README.md",
                      "huggingface": "https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json",
                      "pytorch": profile["torch_index"] + "/torch/", "cninfo": "https://www.cninfo.com.cn/"}.items():
        if offline:
            report["connectivity"][name] = {"skipped": True, "reason": "offline mode"}
            continue
        print("Connectivity probe: " + name, flush=True)
        started = time.perf_counter()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=12) as response:
                response.read(128)
                report["connectivity"][name] = {"ok": True, "status": response.status, "seconds": round(time.perf_counter()-started, 2)}
        except Exception as error:
            report["connectivity"][name] = {"ok": False, "error": str(error)}
            report["warnings"].append(name + " download probe failed; existing local assets may still work")
    report["ready_for_candidate_setup"] = not report["errors"]
    path = root / "exports/doctor.json"
    write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Diagnostic written: {path}", flush=True)
    return report


# 模块：独立环境；作用：按候选版本安装，锁定实际依赖和模型revision；核心业务：否。
def setup(root=ROOT, offline=False):
    diagnostic_path = root / "exports/doctor.json"
    if not diagnostic_path.exists():
        raise ValueError("Run bash server/run.sh doctor first")
    diagnostic = read_json(diagnostic_path)
    profile = read_json(root / "configs/server-profile.json")
    if diagnostic["profile"] != profile or not diagnostic["ready_for_candidate_setup"]:
        raise ValueError("Doctor did not pass this dependency profile; send exports/doctor.json back before setup")
    if platform.system() != "Linux":
        raise ValueError("setup requires the Linux GPU server")
    python = portable_python(root, profile)
    if offline and not python:
        raise ValueError("Offline setup requires the verified portable environment; run server/install_env.sh first")
    environment_source = "portable_conda" if python else "online_venv"
    if python:
        print("Using verified portable Conda environment; dependencies are already installed.", flush=True)
        checksum = root / "server/environment/finsight-conda-linux-x86_64-cu126-v0.1.tar.gz.sha256"
        run_command([python, root / "server/environment/finish_install.py", "--root", root,
                     "--sha256", checksum.read_text(encoding="utf-8").split()[0], "--check"], root)
    else:
        python = root / ".venv/bin/python"
        if not python.exists():
            candidates = [shutil.which(x) for x in ("python3.11", "python3.12", "python3.13")]
            if sys.version_info >= (3, 11):
                candidates.append(sys.executable)
            interpreter = next((p for p in candidates if p), None)
            if not interpreter:
                raise ValueError("Python 3.11+ is required. Install the portable Conda archive with server/install_env.sh first.")
            run_command([interpreter, "-m", "venv", root / ".venv"], root)
        runtime = root / ".runtime"
        runtime.mkdir(exist_ok=True)
        constraints = runtime / "constraints.txt"
        constraints.write_text("\n".join(f"{k}=={profile[k]}" for k in ("torch", "torchvision", "torchaudio", "transformers", "peft")) + "\n", encoding="utf-8")
        run_command([python, "-m", "pip", "install", "--upgrade", "pip"], root)
        run_command([python, "-m", "pip", "install", "--index-url", profile["torch_index"],
                     *[f"{k}=={profile[k]}" for k in ("torch", "torchvision", "torchaudio")]], root)
        run_command([python, "-m", "pip", "install", "-c", constraints, "-e", ".",
                     "https://github.com/hiyouga/LlamaFactory/archive/" + profile["llamafactory_commit"] + ".zip",
                     "sentence-transformers>=3,<6"], root)
    run_command([python, "-m", "pip", "check"], root)
    run_command([python, "-m", "finsight_ashare", "_download", *(["--offline"] if offline else [])], root)
    packages = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    destination = root / "outputs/environment"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "requirements.lock.txt").write_text(packages, encoding="utf-8")
    write_json(destination / "setup.json", {"status": "completed", "created": now(), "profile": profile,
                                          "environment_source": environment_source, "python": str(python),
                                          "lock_sha256": file_hash(destination / "requirements.lock.txt")})
    print("Environment installed. Kernel/memory compatibility still requires the 20-step training probe.", flush=True)


def download_models(root=ROOT, offline=False):
    import torch
    from .assets import download_snapshots, verify_snapshots
    from transformers import AutoConfig, AutoTokenizer
    from llamafactory.data.template import TEMPLATES
    if "qwen3_5_nothink" not in TEMPLATES or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Qwen3.5 template/CUDA/BF16 capability check failed")
    (verify_snapshots if offline else download_snapshots)(root)
    for name in ("Qwen3.5-9B", "bge-m3"):
        AutoConfig.from_pretrained(str(root / "models" / name), local_files_only=True)
        AutoTokenizer.from_pretrained(str(root / "models" / name), local_files_only=True)
    print("Model files and tokenizer configuration available locally.", flush=True)


# 模块：微调；作用：20步门禁、DDP有效批量32、成功保存后才标记完成；核心业务：是。
def check_training_lengths(root, dataset_dir):
    """Reject truncation before a pilot enters LlamaFactory, using its real template."""
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES
    tokenizer = AutoTokenizer.from_pretrained(str(root / "models/Qwen3.5-9B"), local_files_only=True)
    template = TEMPLATES["qwen3_5_nothink"]
    template.fix_special_tokens(tokenizer)
    lengths = []
    for item in read_json(dataset_dir / "train.json"):
        messages = [{"role": "user" if m["from"] == "human" else "assistant", "content": m["value"]}
                    for m in item["conversations"]]
        pairs = template.encode_multiturn(tokenizer, messages, system=item["system"])
        length = sum(len(a) + len(b) for a, b in pairs)
        if length > config(root)["context_length"]:
            raise ValueError(f"Pilot example {item['id']} has {length} tokens; refusing training-time truncation")
        lengths.append({"id": item["id"], "tokens": length})
    if not lengths:
        raise ValueError("No pilot examples to train")
    print(f"PILOT_TOKEN_CHECK=PASS examples={len(lengths)} max_tokens={max(x['tokens'] for x in lengths)}", flush=True)
    return lengths


def train(root=ROOT, probe=False, pilot=False):
    import yaml
    from .dataset import validate_freeze
    from .evaluation import adapter_files
    if pilot and not probe:
        raise ValueError("--pilot is only allowed with --probe; it cannot authorize full training")
    receipt = validate_freeze(root, pilot=pilot)
    if receipt.get("pilot") is not pilot:
        raise ValueError("Dataset mode does not match the requested training mode")
    dataset_dir = root / ("data/pilot" if pilot else "data/frozen")
    lengths = check_training_lengths(root, dataset_dir) if pilot else None
    profile = read_json(root / "configs/server-profile.json")
    training_config = yaml.safe_load((root / "configs/lora.yaml").read_text(encoding="utf-8"))
    if training_config["num_train_epochs"] > 2 or training_config["cutoff_len"] != config(root)["context_length"]:
        raise ValueError("Training budget/context differs from the agreed profile")
    snapshot = read_json(root / "models/snapshots.json")
    experiment = {"dataset_digest": digest(receipt), "profile": profile, "training": deepcopy(training_config), "models": snapshot,
                  "source_digest": source_digest(root), "pilot": pilot}
    identity = digest(experiment)
    probe_path = root / "outputs/training" / ("pilot-probe.json" if pilot else "probe.json")
    if not probe:
        prior = read_json(probe_path) if probe_path.exists() else {}
        if prior.get("status") != "completed" or prior.get("experiment_digest") != identity:
            raise ValueError("Run train --probe first; the same data, model and config must pass 20 steps and adapter reload")
    gpu_count = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True, check=True)
    if int(gpu_count.stdout.strip()) != 4:
        raise ValueError("Expose exactly four GPUs with CUDA_VISIBLE_DEVICES; gradient accumulation assumes world_size=4")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = "pilot-probe-" if pilot else ("probe-" if probe else "full-")
    output = root / "outputs/training" / (prefix + stamp)
    if output.exists():
        raise ValueError("Run directory already exists")
    output.mkdir(parents=True)
    training_config.update(model_name_or_path=str(root / "models/Qwen3.5-9B"), dataset="ashare_train",
                           dataset_dir=str(dataset_dir), output_dir=str(output),
                           gradient_accumulation_steps=8, per_device_train_batch_size=1)
    if probe:
        training_config.update(max_steps=20, save_steps=20)
    generated = output / "train.yaml"
    generated.write_text(yaml.safe_dump(training_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(output / "provenance.json", {**experiment, "experiment_digest": identity, "status": "running", "created": now()})
    if pilot:
        write_json(output / "token_lengths.json", lengths)
        print("PILOT_PROBE: 20 steps on repeated pilot cases; environment check only, not a quality evaluation.", flush=True)
    env = os.environ.copy()
    env.update(FORCE_TORCHRUN="1", NPROC_PER_NODE="4", NNODES="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    executable = Path(sys.executable).parent / "llamafactory-cli"
    run_command([executable, "train", generated], root, env)
    hashes = adapter_files(output)
    state = read_json(output / "trainer_state.json")
    step = int(state["global_step"])
    if step <= 0 or (probe and step != 20):
        raise ValueError("Trainer did not complete the expected number of optimizer steps")
    losses = [float(x["loss"]) for x in state["log_history"] if "loss" in x]
    if not losses or not all(math.isfinite(x) for x in losses):
        raise ValueError("Training losses are missing/non-finite")
    if probe:
        from .models import LocalModel
        from .protocol import SYSTEM
        model = LocalModel(root, output)
        response = model.generate([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "当前没有任何财报证据，请说明无法确认贵州茅台2025年合并营业收入。"}])
        if not response.strip():
            raise ValueError("Adapter reload generated an empty response")
        write_json(output / "reload_check.json", {"response": response, "note": "Save/load compatibility probe only; not an accuracy result"})
    completion = {"status": "completed", "finished": now(), "probe": probe, "pilot": pilot, "global_step": step,
                  "experiment_digest": identity, "dataset_digest": digest(receipt), "adapter": output.relative_to(root).as_posix(),
                  "adapter_hashes": hashes, "last_loss": losses[-1]}
    write_json(output / "completion.json", completion)
    write_json(probe_path if probe else root / "outputs/training/production.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
