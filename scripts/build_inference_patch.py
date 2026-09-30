"""Fetch hash-pinned public kernels and package an isolated offline speed experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/inference-wheels"
RELEASE = ROOT.parent / "_release_work"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(item):
    path = CACHE / item["file"]
    if path.exists() and sha(path) == item["sha256"]:
        print("VERIFIED_CACHE=" + path.name, flush=True)
        return path
    temporary = path.with_suffix(".partial")
    print("DOWNLOAD=" + path.name, flush=True)
    count, last = 0, time.monotonic()
    with urllib.request.urlopen(item["url"], timeout=45) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
            count += len(block)
            if time.monotonic() - last > 10:
                print(f"DOWNLOADED={path.name} MiB={count / 1024**2:.1f}", flush=True)
                last = time.monotonic()
    if sha(temporary) != item["sha256"]:
        raise ValueError("Downloaded hash mismatch: " + path.name)
    temporary.replace(path)
    print(f"DOWNLOAD_VERIFIED={path.name} bytes={count}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "inference_v09/dependencies.json").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        wheels = list(pool.map(fetch, lock["wheels"]))
    if args.download_only:
        return
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit changes before packaging")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {"inference_v091/" + p.name: p.read_bytes() for p in sorted((ROOT / "inference_v09").iterdir()) if p.suffix in {".py", ".json", ".sh", ".md"}}
    # Use the exact v0.8.4 planner in both arms, matching the recorded 3992-token query.
    with zipfile.ZipFile(RELEASE / "finsight-metrics-v0.8.4.zip") as old:
        manifest = json.loads(old.read("metrics_v084/PACKAGE_MANIFEST.json"))
        for name, checksum in manifest["files"].items():
            data = old.read(name)
            assert hashlib.sha256(data).hexdigest() == checksum
            if not Path(name).name.startswith("test_"):
                payload["inference_v091/metrics/" + Path(name).name] = data
    for wheel in wheels:
        payload["inference_v091/wheels/" + wheel.name] = wheel.read_bytes()
    metadata = {"version": "0.9.1", "commit": commit, "planner_version": "0.8.4", "files": {n: hashlib.sha256(b).hexdigest() for n, b in payload.items()}}
    payload["inference_v091/PACKAGE_MANIFEST.json"] = (json.dumps(metadata, indent=2) + "\n").encode()
    output = RELEASE / "finsight-inference-v0.9.1.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, checksum in metadata["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum
    checksum = sha(output)
    output.with_suffix(".zip.sha256").write_bytes(f"{checksum}  {output.name}\n".encode("ascii"))
    print(f"PACKAGE={output}\nBYTES={output.stat().st_size}\nSHA256={checksum}\nCOMMIT={commit}", flush=True)


if __name__ == "__main__":
    main()
