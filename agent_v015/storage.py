"""模块：产物与来源；作用：哈希校验、原子保存；核心业务：否。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sha_lf(path):
    """仅统一文本的物理CRLF换行；数字、字符串、空格与JSON键顺序均不改。"""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def verify(root=ROOT):
    pin = read(root / "agent_v015/PACKAGE_MANIFEST.json")
    for name, expected in {**pin["files"], **pin["dependencies"]}.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("非法依赖路径：" + name)
        actual = sha(path)
        if actual == expected:
            continue
        # Only these JSON data files may differ by platform line endings.
        # Code/package files always require exact bytes, as before.
        lf_expected = pin.get("dependency_lf_sha256", {}).get(name)
        if name in {"data/verified/facts.json", "data/manifest.json"} and name in pin["dependencies"] and name not in pin["files"] and lf_expected:
            actual_lf = sha_lf(path)
            if actual_lf == lf_expected:
                print("DEPENDENCY_MATCH=line_endings_only FILE=" + name, flush=True)
                continue
            raise ValueError(f"数据内容校验失败：{name}; expected_lf={lf_expected}; actual_lf={actual_lf}; actual_raw={actual}")
        raise ValueError(f"文件校验失败：{name}; expected={expected}; actual={actual}")
    return pin
