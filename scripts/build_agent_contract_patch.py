"""Cumulative source patch for observed tool-call and final-envelope failures."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight_ashare.common import file_hash, now

FILES = ["src/finsight_ashare/protocol.py", "src/finsight_ashare/agent.py",
         "src/finsight_ashare/tools.py", "tests/test_agent_tool_contract.py",
         "tests/test_final_envelope.py",
         "scripts/build_agent_contract_patch.py", "docs/AGENT_CONTRACT_PATCH.md"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"created": now(), "requires": "installed v0.3 offline setup",
                "version": "0.3.2",
                "purpose": "Tool-call feedback and audited missing-final-close normalization",
                "files": {name: file_hash(ROOT / name) for name in FILES}}
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, name)
        archive.writestr("AGENT_CONTRACT_PATCH_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
    checksum = file_hash(args.output)
    args.output.with_suffix(".zip.sha256").write_bytes((checksum + "  " + args.output.name + "\n").encode("utf-8"))
    print(f"PATCH={args.output}\nBYTES={args.output.stat().st_size}\nSHA256={checksum}", flush=True)


if __name__ == "__main__":
    main()
