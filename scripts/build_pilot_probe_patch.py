"""Small flat patch for a separately recorded, audited-seed training probe."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight_ashare.common import file_hash, now

FILES = ["src/finsight_ashare/cli.py", "src/finsight_ashare/server.py",
         "tests/test_pilot_training.py", "docs/PILOT_PROBE.md",
         "docs/BASELINE_SMOKE_20260924.md", "docs/IMPLEMENTATION_STATUS.md",
         "scripts/build_pilot_probe_patch.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "0.3.3", "created": now(), "requires": "v0.3 offline setup plus v0.3.2 agent patch",
                "purpose": "Separate 20-step pilot training probe; no production-training authorization",
                "files": {name: file_hash(ROOT / name) for name in FILES}}
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, name)
        archive.writestr("PILOT_PROBE_PATCH_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        for name, digest in manifest["files"].items():
            import hashlib
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    checksum = file_hash(args.output)
    args.output.with_suffix(".zip.sha256").write_bytes((checksum + "  " + args.output.name + "\n").encode("utf-8"))
    print(f"PATCH={args.output}\nBYTES={args.output.stat().st_size}\nSHA256={checksum}", flush=True)


if __name__ == "__main__":
    main()
