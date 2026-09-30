"""Extract only upstream source and text configuration from the pinned archive."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

COMMIT = "8fc5bce28ac201c713219a47d30367c6a84342f4"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    entries = {}
    with zipfile.ZipFile(args.archive) as archive:
        prefix = "FinSight-" + COMMIT + "/"
        for item in archive.infolist():
            if item.is_dir() or not item.filename.startswith(prefix):
                continue
            relative = Path(item.filename[len(prefix):])
            if ".." in relative.parts or relative.is_absolute():
                raise ValueError("Unsafe archive member")
            if relative.parts[0] in {"assets", ".github"} or relative.name.startswith(".env"):
                continue
            if relative.suffix not in {".py", ".txt", ".yaml", ".yml", ".md", ".json", ".toml", ".sh"} and relative.name != "LICENSE":
                continue
            data = archive.read(item)
            target = ROOT / "vendor/FinSight" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            entries[relative.as_posix()] = hashlib.sha256(data).hexdigest()
        license_bytes = archive.read(prefix + "LICENSE")
        (ROOT / "LICENSE").write_bytes(license_bytes)
    manifest = {"repository": "https://github.com/RUC-NLPIR/FinSight", "commit": COMMIT,
                "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                "omitted": ["assets (media, example reports)", ".github", ".env files", "non-source formats"], "files": entries}
    (ROOT / "vendor/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Vendored {len(entries)} unmodified source/config files at {COMMIT}")


if __name__ == "__main__":
    main()
