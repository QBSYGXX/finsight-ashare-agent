"""Verify the ZIP, extract into a fresh directory and run its CLI entrypoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    destination = ROOT / ".cache" / ("release-check-" + uuid.uuid4().hex[:10])
    destination.mkdir(parents=True)
    with zipfile.ZipFile(args.zip) as archive:
        manifest = json.loads(archive.read("finsight-ashare-agent/RELEASE_MANIFEST.json"))
        for name in archive.namelist():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "finsight-ashare-agent":
                raise ValueError("Unsafe ZIP member")
        for name, expected in manifest["files"].items():
            actual = hashlib.sha256(archive.read("finsight-ashare-agent/" + name)).hexdigest()
            if actual != expected:
                raise ValueError("Release hash mismatch: " + name)
        archive.extractall(destination)
    project = destination / "finsight-ashare-agent"
    env = os.environ.copy()
    env.update(PYTHONPATH=str(project / "src"), FINSIGHT_ROOT=str(project), PYTHONUTF8="1")
    commands = [[sys.executable, "-m", "finsight_ashare", "--help"]]
    if args.replay:
        commands.append([sys.executable, "-m", "finsight_ashare", "smoke", "--replay", "--count", "10"])
    for command in commands:
        result = subprocess.run(command, cwd=project, env=env, check=False)
        if result.returncode:
            return result.returncode
    print(f"RELEASE_VALIDATED files={len(manifest['files'])} extracted_root={project}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
