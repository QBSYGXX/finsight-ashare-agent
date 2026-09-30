"""Resume official report acquisition, retaining per-report errors and a run log."""
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight_ashare.common import file_hash, now, write_json
from finsight_ashare.data import fetch, manifest


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    output = ROOT / "outputs/acquisition" / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    with (output / "run.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print("START=" + now(), flush=True)
            for row in manifest(ROOT):
                try:
                    fetch(ROOT, only=row["id"])
                except Exception as error:
                    failures.append({"id": row["id"], "error": f"{type(error).__name__}: {error}"})
                    print("REPORT_ERROR=" + json.dumps(failures[-1], ensure_ascii=False), flush=True)
                write_json(output / "failures.json", failures)
            verified = []
            for row in manifest(ROOT):
                path = ROOT / "data/raw" / (row["id"] + ".pdf")
                if path.exists() and row.get("sha256") == file_hash(path) and row.get("downloaded_url") == row["url"]:
                    verified.append(row["id"])
            result = {"finished": now(), "expected": len(manifest(ROOT)), "downloaded_and_hashed": verified,
                      "failures": failures, "note": "Downloaded PDFs are not yet audited financial facts."}
            write_json(output / "summary.json", result)
            print(f"DOWNLOADED={len(verified)}/{result['expected']} FAILURES={len(failures)} LOG={output / 'run.log'}", flush=True)
    return 0 if len(verified) == result["expected"] and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
