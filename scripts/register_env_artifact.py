"""Verify a completed Linux artifact and register its small public metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "finsight-conda-linux-x86_64-cu126-v0.1.tar.gz"


def register(release):
    archive = release / ARCHIVE
    checksum = release / (ARCHIVE + ".sha256")
    expected = checksum.read_text(encoding="utf-8").split()[0]
    sha = hashlib.sha256()
    read = 0
    last = time.monotonic()
    with archive.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
            read += len(block)
            if time.monotonic() - last >= 15:
                print(f"HASHING={read / 1024**3:.2f} GiB", flush=True)
                last = time.monotonic()
    if sha.hexdigest() != expected:
        raise ValueError("Environment archive SHA256 mismatch")
    report_path = release / "finsight-conda-environment-check.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report["status"] != "passed" or report["platform"] != "Linux" or report["architecture"] != "x86_64"
            or not report["small_qwen35_forward_backward"] or not report["lora_save_reload"]):
        raise ValueError("Linux environment checks did not pass")
    profile = json.loads((ROOT / "configs/server-profile.json").read_text(encoding="utf-8"))
    for name in ("torch", "torchvision", "torchaudio", "transformers", "peft"):
        if report["versions"][name].split("+")[0] != profile[name]:
            raise ValueError("Archive dependency profile mismatch: " + name)
    destination = ROOT / "server/environment"
    for source in (checksum, report_path, release / "finsight-conda-requirements.lock.txt", release / "finsight-conda-explicit.txt"):
        shutil.copyfile(source, destination / source.name)
    metadata = {"filename": ARCHIVE, "bytes": archive.stat().st_size, "sha256": expected,
                "target": "Linux x86_64", "minimum_glibc": "2.28", "profile": profile,
                "base_model_weights_included": False, "a800_gpu_tested": False,
                "verification": report_path.name}
    (destination / "artifact.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    register(parser.parse_args().output_dir.resolve())
