"""Shared paths and atomic artifacts; no model imports at CLI startup."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


# 模块：运行产物；作用：统一路径与原子写入；核心业务：否。
ROOT = Path(os.environ.get("FINSIGHT_ROOT", Path(__file__).resolve().parents[2]))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_digest(root=ROOT):
    return digest({p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / "src/finsight_ashare").glob("*.py"))})


def now():
    return datetime.now(timezone.utc).isoformat()


def config(root=ROOT):
    return read_json(root / "configs/project.json")


def companies(root=ROOT):
    return {item["code"]: item for item in config(root)["companies"]}


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
