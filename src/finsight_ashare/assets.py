"""Pinned public model snapshots and hash-checked, offline server handoff."""
from pathlib import Path, PurePosixPath
import re
import zipfile
import shutil
import subprocess
import time
import json
import os
import urllib.request
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor

from .common import ROOT, config, file_hash, now, read_json, write_json


def model_specs(root=ROOT):
    options = config(root)
    return ((options["model"], "Qwen3.5-9B"), (options["embedding_model"], "bge-m3"))


def model_file_selected(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return False
    if len(path.parts) > 1:
        return name == "1_Pooling/config.json"
    return (path.suffix in {".json", ".txt", ".model", ".jinja", ".safetensors"}
            or name == "pytorch_model.bin" or name.startswith("LICENSE") or name == "README.md")


def curl_snapshot_file(repo, revision, item, directory, mirror_url=None):
    from huggingface_hub import hf_hub_url
    path = directory / item.rfilename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == item.size and item.lfs and file_hash(path) == item.lfs.sha256:
        print("REUSE=" + item.rfilename, flush=True)
        return
    partial = path.with_name(path.name + ".partial")
    aria2 = os.environ.get("FINSIGHT_ARIA2")
    if partial.with_name(partial.name + ".aria2").exists() and not (mirror_url and aria2 and item.lfs):
        raise ValueError("Segmented partial download requires --transport modelscope and FINSIGHT_ARIA2 to resume")
    if mirror_url and aria2 and item.lfs:
        env = {key: value for key, value in os.environ.items() if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
        print("MULTICONNECTION=" + item.rfilename, flush=True)
        result = subprocess.run([aria2, "--continue=true", "--auto-file-renaming=false", "--file-allocation=none",
            "--max-connection-per-server=8", "--split=8", "--min-split-size=16M", "--summary-interval=30",
            "--console-log-level=warn", "--max-tries=5", "--retry-wait=3", "--timeout=60", "--connect-timeout=20",
            "--checksum=sha-256=" + item.lfs.sha256, "--dir=" + str(path.parent), "--out=" + partial.name,
            mirror_url], env=env, check=False)
        if result.returncode:
            raise RuntimeError(f"aria2 exit={result.returncode}: {item.rfilename}; resume data retained")
        if partial.stat().st_size != item.size:
            raise ValueError("Incomplete multi-connection transfer")
        partial.replace(path)
        return
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if not executable:
        raise RuntimeError("curl transport requested but curl was not found")
    print("DOWNLOAD=" + repo + "/" + item.rfilename, flush=True)
    for attempt in range(5):
        if partial.exists() and partial.stat().st_size == item.size:
            break
        result = subprocess.run([executable, "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "20", "--speed-time", "90", "--speed-limit", "1024",
            "--continue-at", "-", "--output", str(partial),
            *(["--noproxy", "*"] if mirror_url else []),
            mirror_url or hf_hub_url(repo, item.rfilename, revision=revision)], check=False)
        if result.returncode == 0:
            break
        if result.returncode not in {18, 28, 35, 55, 56} or attempt == 4:
            raise RuntimeError(f"curl exit={result.returncode}: {item.rfilename}; partial retained for resume")
        print(f"RESUME={item.rfilename} AFTER_EXIT={result.returncode} ATTEMPT={attempt + 2}", flush=True)
        time.sleep(2)
    if partial.stat().st_size != item.size:
        raise ValueError("Incomplete transfer: " + item.rfilename)
    partial.replace(path)


def verified_modelscope_urls(repo, selected):
    """Mirror only weights whose published hashes equal the pinned HF commit."""
    if repo != "Qwen/Qwen3.5-9B":
        return {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("https://modelscope.cn/api/v1/models/" + repo + "/repo/files?Revision=master&Recursive=true", timeout=30) as response:
        rows = {row["Path"]: row for row in json.load(response)["Data"]["Files"]}
    urls = {}
    for item in selected:
        if not item.rfilename.endswith(".safetensors"):
            continue
        match = rows.get(item.rfilename, {})
        if not item.lfs or match.get("Sha256") != item.lfs.sha256 or match.get("Size") != item.size:
            raise ValueError("ModelScope weight does not match pinned HF hash: " + item.rfilename)
        urls[item.rfilename] = "https://modelscope.cn/models/" + repo + "/resolve/" + match["Revision"] + "/" + quote(item.rfilename)
    return urls


def download_snapshots(root=ROOT, transport="hub"):
    from huggingface_hub import HfApi, snapshot_download
    lock_path = root / "models/snapshots.json"
    lock = read_json(lock_path) if lock_path.exists() else {}
    for repo, folder in model_specs(root):
        previous = lock.get(folder, {})
        if previous and previous.get("repo") != repo:
            raise ValueError("Existing snapshot belongs to another model: " + folder)
        revision = previous.get("revision", "main")
        info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)
        if not re.fullmatch(r"[0-9a-f]{40}", info.sha):
            raise ValueError("Model revision is not an immutable commit")
        selected = [s for s in info.siblings if model_file_selected(s.rfilename)]
        entry = {"repo": repo, "revision": info.sha, "status": "downloading", "files": {}}
        lock[folder] = entry
        write_json(lock_path, lock)
        size = sum(s.size for s in selected)
        print(f"MODEL={repo} REVISION={info.sha} FILES={len(selected)} GiB={size / 1024**3:.2f}", flush=True)
        if transport in {"curl", "modelscope"}:
            mirrors = verified_modelscope_urls(repo, selected) if transport == "modelscope" else {}
            entry["weight_mirror_urls"] = mirrors
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda item: curl_snapshot_file(repo, info.sha, item, root / "models" / folder,
                                                             mirrors.get(item.rfilename)), selected))
        elif transport == "hub":
            snapshot_download(repo_id=repo, revision=info.sha, token=False, local_dir=root / "models" / folder,
                              allow_patterns=[s.rfilename for s in selected], max_workers=4)
        else:
            raise ValueError("Unknown download transport")
        for item in selected:
            path = root / "models" / folder / item.rfilename
            print("VERIFY=" + folder + "/" + item.rfilename, flush=True)
            if path.stat().st_size != item.size:
                raise ValueError("Incomplete model file: " + str(path))
            sha256 = file_hash(path)
            if item.lfs and sha256 != item.lfs.sha256:
                raise ValueError("Model file differs from upstream SHA256: " + str(path))
            entry["files"][item.rfilename] = {"bytes": item.size, "sha256": sha256}
        entry["status"] = "downloaded"
        write_json(lock_path, lock)
    return verify_snapshots(root)


def verify_snapshots(root=ROOT):
    """No network calls or heavyweight model imports; verify every shipped byte."""
    lock = read_json(root / "models/snapshots.json")
    for repo, folder in model_specs(root):
        entry = lock.get(folder, {})
        if (entry.get("repo") != repo or entry.get("status") != "downloaded"
                or not re.fullmatch(r"[0-9a-f]{40}", entry.get("revision", "")) or not entry.get("files")):
            raise ValueError("Missing verified snapshot manifest for " + folder)
        directory = root / "models" / folder
        files = entry["files"]
        for required in ("config.json", "tokenizer_config.json"):
            if required not in files:
                raise ValueError("Missing model configuration: " + folder + "/" + required)
        weights = {name for name in files if name.endswith(".safetensors") or name == "pytorch_model.bin"}
        if not weights:
            raise ValueError("No model weights in " + folder)
        for name, expected in files.items():
            if not model_file_selected(name):
                raise ValueError("Unsupported snapshot path: " + name)
            path = directory / name
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Missing/unsafe snapshot file: " + name)
            print("VERIFY=" + folder + "/" + name, flush=True)
            if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
                raise ValueError("Snapshot integrity failure: " + folder + "/" + name)
        index = directory / "model.safetensors.index.json"
        if index.exists():
            references = set(read_json(index)["weight_map"].values())
            if not references or not references <= weights:
                raise ValueError("Snapshot is missing indexed weight shards: " + folder)
        if folder == "bge-m3":
            if not {"modules.json", "1_Pooling/config.json"} <= files.keys():
                raise ValueError("bge-m3 SentenceTransformer modules/pooling configuration missing")
    return lock


def build_assets_bundle(destination, root=ROOT):
    lock = verify_snapshots(root)
    seed = read_json(root / "examples/maotai_2024/seed.json")
    report_id = seed["report"]["id"]
    pdf = root / "data/raw" / (report_id + ".pdf")
    if not pdf.exists() or file_hash(pdf) != seed["report"]["sha256"]:
        raise ValueError("The audited smoke-test PDF is missing or changed")
    rows = read_json(root / "data/manifest.json")
    selected_report = next(r for r in rows if r["id"] == report_id)
    if (selected_report.get("sha256") != seed["report"]["sha256"]
            or selected_report.get("downloaded_url") != seed["report"]["url"]):
        raise ValueError("Smoke PDF cache provenance is incomplete")
    selected = [root / "models/snapshots.json", root / "data/manifest.json", pdf]
    for _, folder in model_specs(root):
        selected.extend(root / "models" / folder / name for name in lock[folder]["files"])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in selected:
            relative = path.relative_to(root).as_posix()
            print("PACK=" + relative, flush=True)
            archive.write(path, relative)
    temporary.replace(destination)
    checksum = file_hash(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_bytes((checksum + "  " + destination.name + "\n").encode("utf-8"))
    write_json(destination.with_suffix(".json"), {"created": now(), "bytes": destination.stat().st_size,
        "sha256": checksum, "models": {name: {"repo": row["repo"], "revision": row["revision"]} for name, row in lock.items()},
        "reports": [report_id], "all_36_reports_included": False, "contains_training_results": False})
    print(f"ASSETS_COMPLETE={destination}\nBYTES={destination.stat().st_size}\nSHA256={checksum}", flush=True)
    return destination
