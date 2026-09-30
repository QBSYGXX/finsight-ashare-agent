"""Exercise the shipped installer without network or the original Conda prefix."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import uuid

from build_portable_env import ROOT, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    release = args.output_dir.resolve()
    name = "finsight-env-install-check-" + uuid.uuid4().hex[:8]
    log_path = release / ("finsight-env-install-check-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".log")
    print(f"LOG={log_path}\nCONTAINER={name}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        run(["docker", "run", "--name", name, "--init", "--network", "none", "--platform", "linux/amd64",
             "--mount", "type=volume,source=finsight-conda-install-validation,target=/verification",
             "--mount", f"type=bind,source={ROOT},target=/source,readonly",
             "--mount", f"type=bind,source={release},target=/exports", "finsight-conda-builder:0.1",
             "bash", "/source/server/environment/test_offline_install.sh"], log)
    shutil.copyfile(release / "finsight-offline-install-check.json", ROOT / "server/environment/finsight-offline-install-check.json")
    print("OFFLINE_INSTALL_VALIDATED", flush=True)


if __name__ == "__main__":
    main()
