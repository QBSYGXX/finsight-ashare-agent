"""Build an additive chat-only patch pinned to the delivered v0.7 source."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent / "_release_work"


def main():
    with zipfile.ZipFile(RELEASE / "finsight-evaluation-v0.7.zip") as old:
        files = json.loads(old.read("evaluation_v07/PACKAGE_MANIFEST.json"))["files"]
        assert all(hashlib.sha256(old.read(n)).hexdigest() == h for n, h in files.items())
        pin = {n: files[n] for n in ("evaluation_v07/resolver.py", "evaluation_v07/resolved_agent.py",
                                    "evaluation_v07/company_aliases.json")}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit project changes before packaging")
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / "chat_v071").iterdir())
               if p.is_file() and p.suffix in {".py", ".sh", ".md"}}
    payload["chat_v071/dependency_pin.json"] = (json.dumps(pin, indent=2) + "\n").encode()
    for name, data in payload.items():
        if name.endswith(".sh") and b"\r" in data:
            raise ValueError("Shell script must use LF")
    manifest = {"version": "0.7.1", "git_commit": revision,
                "files": {n: hashlib.sha256(b).hexdigest() for n, b in payload.items()}}
    payload["chat_v071/PACKAGE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output = RELEASE / "finsight-chat-v0.7.1.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert all(archive.read(n) == b for n, b in payload.items())
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_bytes((checksum + "  " + output.name + "\n").encode("ascii"))
    print(f"PACKAGE={output}\nBYTES={output.stat().st_size}\nSHA256={checksum}\nCOMMIT={revision}")


if __name__ == "__main__":
    main()
