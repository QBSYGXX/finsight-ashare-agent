"""Package the isolated metric engine without changing frozen experiments."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent / "_release_work"
PACKAGE_DIR = "metrics_v085"


def main():
    files = {}
    for package, names in [
        ("finsight-evaluation-v0.7.zip", ["evaluation_v07/resolver.py", "evaluation_v07/company_aliases.json"]),
        ("finsight-chat-v0.7.1.zip", ["chat_v071/preflight.py"]),
    ]:
        with zipfile.ZipFile(RELEASE / package) as archive:
            manifest = json.loads(archive.read(next(n for n in archive.namelist() if n.endswith("/PACKAGE_MANIFEST.json"))))["files"]
            for name in names:
                files[name] = hashlib.sha256(archive.read(name)).hexdigest()
                assert files[name] == manifest[name]
    with zipfile.ZipFile(RELEASE / "finsight-eval-v07-20260926-165706.zip") as archive:
        export = json.loads(archive.read("EXPORT_MANIFEST.json"))["files"]
        frozen_bytes = archive.read("data/frozen/manifest.json")
        assert hashlib.sha256(frozen_bytes).hexdigest() == export["data/frozen/manifest.json"]
        frozen = json.loads(frozen_bytes)
        frozen_digest = json.loads(archive.read("outputs/evaluation_v07/4e94c83dcdc38248/run.json"))["frozen"]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit project changes before packaging")
    payload = {PACKAGE_DIR + "/" + p.relative_to(ROOT / "metrics_v08").as_posix(): p.read_bytes() for p in sorted((ROOT / "metrics_v08").rglob("*"))
               if p.is_file() and p.suffix in {".py", ".sh", ".json", ".md"}}
    pin = {"files": files, "frozen_digest": frozen_digest,
           **{key: frozen[key] for key in ("facts_digest", "config_digest", "source_digest")}}
    payload[PACKAGE_DIR + "/dependency_pin.json"] = (json.dumps(pin, indent=2) + "\n").encode()
    assert all(b"\r" not in b for n, b in payload.items() if n.endswith(".sh"))
    manifest = {"version": "0.8.5", "git_commit": revision,
                "files": {n: hashlib.sha256(b).hexdigest() for n, b in payload.items()}}
    payload[PACKAGE_DIR + "/PACKAGE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output = RELEASE / "finsight-metrics-v0.8.5.zip"
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
