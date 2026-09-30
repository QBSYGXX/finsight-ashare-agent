"""Download public pinned snapshots and package the first real report for upload."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--transport", choices=["hub", "curl", "modelscope"], default="hub")
    args = parser.parse_args()
    if args.worker:
        from finsight_ashare.assets import download_snapshots, build_assets_bundle
        finished = threading.Event()
        def progress():
            previous = 0
            while not finished.wait(30):
                try:
                    paths = [p for p in (ROOT / "models").rglob("*") if p.is_file()]
                    size = sum(p.stat().st_size for p in paths)
                    print(f"MODEL_CACHE_GiB={size / 1024**3:.3f} DELTA_MiB={(size - previous) / 1024**2:.1f} "
                          "(includes partial downloads, not a completion check)", flush=True)
                    previous = size
                except OSError:
                    pass  # Download workers may rename a file during the scan.
        monitor = threading.Thread(target=progress, daemon=True)
        monitor.start()
        try:
            download_snapshots(ROOT, transport=args.transport)
            build_assets_bundle(args.output, ROOT)
        finally:
            finished.set()
            monitor.join()
    else:
        from build_portable_env import run
        args.output.parent.mkdir(parents=True, exist_ok=True)
        log_path = args.output.parent / ("finsight-assets-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".log")
        print("LOG=" + str(log_path), flush=True)
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            run([sys.executable, "-u", __file__, "--worker", "--transport", args.transport,
                 "--output", args.output.resolve()], log)


if __name__ == "__main__":
    main()
