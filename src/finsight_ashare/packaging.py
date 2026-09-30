"""Explicit release allowlist and separate private experiment exports."""
from collections import deque
import json
from pathlib import Path
import time
import zipfile

from .common import ROOT, file_hash, now, read_json, write_json

RELEASE_DIRS = {"src", "configs", "server", "scripts", "tests", "docs", "examples", "vendor"}
RELEASE_FILES = {"README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", ".gitignore", ".gitattributes"}
FORBIDDEN_PARTS = {"__pycache__", ".git", ".venv", ".runtime", ".cache", ".pytest_cache", "node_modules"}


# 模块：交付包；作用：代码包排除数据/凭据/模型，结果包另行收集；核心业务：否。
def release_files(root=ROOT):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in FORBIDDEN_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.parts[0] not in RELEASE_DIRS and relative.as_posix() not in RELEASE_FILES:
            continue
        if path.name.startswith(".env") or path.suffix.lower() in {".pdf", ".zip", ".gz", ".bz2", ".tgz", ".partial", ".safetensors", ".bin", ".log", ".key", ".pem", ".pyc"}:
            continue
        yield path


def build_release(destination, root=ROOT):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = list(release_files(root))
    manifest = {"version": "0.2.0", "created": now(), "status": "local implementation; server training/evaluation not yet verified",
                "files": {p.relative_to(root).as_posix(): file_hash(p) for p in files}}
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            archive.write(path, "finsight-ashare-agent/" + relative)
        archive.writestr("finsight-ashare-agent/RELEASE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary.replace(destination)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        if any(not name.startswith("finsight-ashare-agent/") for name in archive.namelist()):
            raise ValueError("Incorrect release root")
    print(f"RELEASE={destination}\nFILES={len(files)}\nBYTES={destination.stat().st_size}\nSHA256={file_hash(destination)}", flush=True)
    return destination


def export_results(root=ROOT):
    destination = root / "exports" / ("finsight-results-" + time.strftime("%Y%m%d-%H%M%S") + ".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected = []
    for folder in ("configs", "outputs", "data/frozen", "data/pilot", "data/verified", "data/review_v0.4"):
        for path in (root / folder).rglob("*"):
            if not path.is_file() or path.is_symlink() or any(part.startswith("checkpoint-") for part in path.parts):
                continue
            if path.name.startswith(".env") or path.suffix in {".bin", ".pt", ".log", ".pdf"}:
                continue
            if path.suffix == ".safetensors" and path.name != "adapter_model.safetensors":
                continue
            if path.suffix in {".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".html", ".safetensors", ".png"}:
                selected.append(path)
    for relative in ("exports/doctor.json", "data/manifest.json", "data/audit.csv", "models/snapshots.json"):
        path = root / relative
        if path.exists():
            selected.append(path)
    run_summaries = []
    for status in sorted((root / "runs").glob("*/status.json")):
        state = read_json(status)
        logfile = status.with_name("run.log")
        tail = []
        if logfile.exists():
            with logfile.open(encoding="utf-8", errors="replace") as stream:
                tail = list(deque(stream, maxlen=120))
        run_summaries.append({"run": status.parent.name, "status": state, "last_120_log_lines": tail})
        selected.append(status)
        if status.with_name("gpu.csv").exists():
            selected.append(status.with_name("gpu.csv"))
    manifest = {"created": now(), "contains_base_model": False,
                "notice": "Private experiment handoff. Review paths, excerpts and model/data licenses before publishing.",
                "files": {p.relative_to(root).as_posix(): file_hash(p) for p in selected}}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("run_summaries.json", json.dumps(run_summaries, ensure_ascii=False, indent=2))
        archive.writestr("EXPORT_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"RESULTS={destination}\nSHA256={file_hash(destination)}", flush=True)
    return destination
