"""Addon integrity and explicit project root, before any financial module import."""
import hashlib
import json
import os
from pathlib import Path
import sys

ADDON = Path(__file__).resolve().parent
ROOT = ADDON.parent
os.environ["FINSIGHT_ROOT"] = str(ROOT)
sys.path[:0] = [str(ROOT / "vllm_v013"), str(ROOT / "vllm_v013/metrics"),
                str(ROOT / "src"), str(ROOT / "evaluation_v07"), str(ROOT / "chat_v071")]


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify():
    manifest = json.loads((ADDON / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, h in manifest["files"].items():
        target = (ADDON / name).resolve()
        if not target.is_relative_to(ADDON) or digest(target) != h:
            raise ValueError("Probe file changed: " + name)
    if digest(ROOT / "vllm_v013/PACKAGE_MANIFEST.json") != manifest["v013_manifest_sha256"]:
        raise ValueError("Different v0.13 package; do not bypass the dependency pin")
    from runtime import verify_package
    verify_package()
    return manifest
