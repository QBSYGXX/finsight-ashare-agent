"""Create a flat update ZIP for an existing server project, without runtime/data."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight_ashare.common import file_hash, now

FILES = [
    "src/finsight_ashare/assets.py", "src/finsight_ashare/server.py", "src/finsight_ashare/cli.py",
    "server/environment/probe_cuda.py", "scripts/prepare_offline_assets.py",
    "scripts/build_offline_patch.py", "scripts/validate_offline_models.py",
    "tests/test_offline_assets.py", "docs/OFFLINE_NEXT_STEPS.md",
    "docs/IMPLEMENTATION_STATUS.md",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = {"created": now(), "purpose": "Offline assets and conditional CUDA 12 compatibility",
                "files": {name: file_hash(ROOT / name) for name in FILES}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, name)
        archive.writestr("OFFLINE_PATCH_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Patch CRC failure")
    checksum = file_hash(args.output)
    args.output.with_suffix(".zip.sha256").write_bytes((checksum + "  " + args.output.name + "\n").encode("utf-8"))
    print(f"PATCH={args.output}\nBYTES={args.output.stat().st_size}\nSHA256={checksum}", flush=True)


if __name__ == "__main__":
    main()
