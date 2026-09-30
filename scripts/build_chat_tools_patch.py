"""Package additive v0.7.2 chat files with pinned earlier dependencies."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent / "_release_work"


def main():
    pins = {}
    for archive_name, names in [
        ("finsight-evaluation-v0.7.zip", ["evaluation_v07/resolver.py", "evaluation_v07/resolved_agent.py", "evaluation_v07/company_aliases.json"]),
        ("finsight-chat-v0.7.1.zip", ["chat_v071/preflight.py"]),
    ]:
        with zipfile.ZipFile(RELEASE / archive_name) as archive:
            manifest_name = next(n for n in archive.namelist() if n.endswith("/PACKAGE_MANIFEST.json"))
            manifest = json.loads(archive.read(manifest_name))["files"]
            for name in names:
                checksum = hashlib.sha256(archive.read(name)).hexdigest()
                assert checksum == manifest[name]
                pins[name] = checksum
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit changes before packaging")
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT / "chat_v072").iterdir())
               if p.is_file() and p.suffix in {".py", ".md", ".sh"}}
    payload["chat_v072/run.sh"] = (ROOT / "chat_v071/run.sh").read_bytes()
    payload["chat_v072/dependency_pin.json"] = (json.dumps(pins, indent=2) + "\n").encode()
    assert b"\r" not in payload["chat_v072/run.sh"]
    manifest = {"version": "0.7.2", "git_commit": revision,
                "files": {n: hashlib.sha256(data).hexdigest() for n, data in payload.items()}}
    payload["chat_v072/PACKAGE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output = RELEASE / "finsight-chat-v0.7.2.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert all(archive.read(n) == data for n, data in payload.items())
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_bytes((checksum + "  " + output.name + "\n").encode("ascii"))
    print(f"PACKAGE={output}\nSHA256={checksum}\nCOMMIT={revision}")


if __name__ == "__main__":
    main()
