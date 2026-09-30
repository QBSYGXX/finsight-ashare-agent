"""Bind a relocated environment to its project path and dependency profile."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def finish(root, sha256, check=False):
    prefix = (root / ".runtime/envs/finsight").resolve()
    if Path(sys.prefix).resolve() != prefix:
        raise ValueError("Run this command with the extracted environment's Python")
    profile = json.loads((root / "configs/server-profile.json").read_text(encoding="utf-8"))
    packed_profile = json.loads((prefix / "finsight-server-profile.json").read_text(encoding="utf-8"))
    if profile != packed_profile:
        raise ValueError("Environment archive and project dependency profile differ")
    for name in ("torch", "torchvision", "torchaudio", "transformers", "peft"):
        if importlib.metadata.version(name).split("+")[0] != profile[name]:
            raise ValueError("Unexpected installed version of " + name)
    marker = prefix / ".finsight-ready.json"
    receipt = {"status": "ready", "prefix": str(prefix), "archive_sha256": sha256, "profile": profile}
    if check:
        if json.loads(marker.read_text(encoding="utf-8")) != receipt:
            raise ValueError("Environment receipt differs; do not move a relocated Conda environment")
    else:
        marker.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (prefix / ".finsight-installing").unlink(missing_ok=True)
    print("PORTABLE_ENVIRONMENT_READY=" + str(prefix), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    finish(args.root, args.sha256, args.check)
