"""Package a clean Git commit; never copy environments, weights or working data."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "finsight-ashare-agent/"
BLOCKED_PARTS = {".git", ".runtime", ".venv", ".cache", "__pycache__", "data", "models", "outputs", "runs", "exports"}
BLOCKED_SUFFIXES = (".pdf", ".safetensors", ".bin", ".pt", ".pth", ".pem", ".key", ".log", ".zip", ".tar.gz", ".tgz")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def validate_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe archive path: " + name)
    if any(p in BLOCKED_PARTS for p in path.parts) or name.lower().endswith(BLOCKED_SUFFIXES):
        raise ValueError("Non-source artifact tracked in Git: " + name)
    if path.name.startswith(".env") and path.name != ".env.example":
        raise ValueError("Private environment configuration: " + name)


def verify(archive):
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC failed")
        names = z.namelist()
        manifest_name = PREFIX + "SOURCE_MANIFEST.json"
        manifest = json.loads(z.read(manifest_name))
        expected = {PREFIX + n for n in manifest["files"]} | {manifest_name}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Archive membership mismatch")
        for name, digest in manifest["files"].items():
            validate_name(name)
            if hashlib.sha256(z.read(PREFIX + name)).hexdigest() != digest:
                raise ValueError("File hash mismatch: " + name)
        # Frozen release files must retain exact bytes after Git text normalization.
        for addon in ("agent_v015", "lora_v0161"):
            frozen = json.loads(z.read(PREFIX + addon + "/PACKAGE_MANIFEST.json"))
            for name, digest in frozen["files"].items():
                if "data" in PurePosixPath(name).parts and name not in manifest["files"]:
                    continue  # Training texts stay external; original pins remain in source.
                if manifest["files"].get(name) != digest:
                    raise ValueError("Frozen source mismatch: " + name)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        manifest = verify(args.verify)
        print(json.dumps({"status": "verified", "commit": manifest["commit"], "files": len(manifest["files"])}, indent=2))
        return
    if not args.output:
        parser.error("--output is required when building")
    if git("status", "--porcelain").strip():
        raise SystemExit("Commit or preserve working changes before release; tree must be clean.")
    commit = git("rev-parse", "HEAD").decode().strip()
    content = {}
    with zipfile.ZipFile(io.BytesIO(git("archive", "--format=zip", "HEAD"))) as source:
        for item in source.infolist():
            if item.is_dir():
                continue
            validate_name(item.filename)
            if item.file_size > 5_000_000:
                raise ValueError("Unexpected large source file: " + item.filename)
            payload = source.read(item)
            if re.search(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)", payload):
                raise ValueError("Potential credential in: " + item.filename)
            content[item.filename] = payload
    manifest = {
        "release": "research-v1.0", "commit": commit,
        "scope": "Committed source and concise public research examples; external assets excluded",
        "files": {n: hashlib.sha256(b).hexdigest() for n, b in sorted(content.items())},
    }
    content["SOURCE_MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    target = args.output.resolve()
    if target.exists() or target.with_suffix(target.suffix + ".sha256").exists():
        raise FileExistsError("Choose a new release filename: " + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, payload in sorted(content.items()):
            info = zipfile.ZipInfo(PREFIX + name, date_time=(2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            z.writestr(info, payload)
    verify(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_text(digest + "  " + target.name + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "verified", "archive": str(target), "commit": commit,
                      "files": len(manifest["files"]), "bytes": target.stat().st_size, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
