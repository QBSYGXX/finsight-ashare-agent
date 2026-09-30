"""Offline runtime boundaries shared by install and inference entry points."""
import hashlib
import json
import os
from pathlib import Path

ADDON = Path(__file__).resolve().parent
ROOT = ADDON.parent
BASE = ROOT / ".runtime/envs/finsight"
ENV = ROOT / ".runtime/envs/finsight-vllm-v013"
COMPAT = ROOT / "native_probe_v012/payload/compat"


# 模块：运行边界；作用：验证指定文件，隔离Python和CUDA库搜索路径；核心业务：否。
def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_package(include_wheels=False):
    data = json.loads((ADDON / "PACKAGE_MANIFEST.json").read_text())
    for name, expected in data["files"].items():
        if name.startswith("wheelhouse/") and not include_wheels:
            continue
        path = (ADDON / name).resolve()
        if not path.is_relative_to(ADDON) or sha256(path) != expected:
            raise ValueError("Package file changed: " + name)
    for name, expected in data["dependencies"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise ValueError("Required earlier package changed: " + name)
    return data


def process_env(gpu, accelerated=True):
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "LD_PRELOAD", "CUDA_HOME", "CUDA_PATH", "CUDACXX", "VIRTUAL_ENV"):
        env.pop(key, None)
    prefix = ENV if accelerated else BASE
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONUTF8="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        PIP_NO_INDEX="1", PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
        CUDA_VISIBLE_DEVICES=str(gpu), CONDA_PREFIX=str(prefix),
        VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_CACHE_ROOT=str(ROOT / ".cache/vllm-v013"),
        TRITON_CACHE_DIR=str(ROOT / ".cache/triton-v013"),
        TORCH_EXTENSIONS_DIR=str(ROOT / ".cache/torch-extensions-v013"))
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = (str(COMPAT) + ":" if accelerated else "") + str(BASE / "lib")
    return env


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
