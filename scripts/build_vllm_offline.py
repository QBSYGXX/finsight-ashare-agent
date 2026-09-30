"""Audit Linux wheel closure and build two upload archives; never install on Windows."""
import ast
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.markers import default_environment

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/vllm-offline"
RELEASE = ROOT.parent / "_release_work"
ADDON = ROOT / "vllm_v013"
ENGINE = "vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
ENGINE_HASH = "8bc2203995d061e6b988916b71b9dee8a5970f5fdc5f37d4445a877a2fab2cc1"
LLG_HASH = "a87ea547df425eb2a75b083b4d0a045d86d9b4c2236362a1c8c073b90d834ba6"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def repack_llguidance(destination):
    """Rebuild wheel RECORD around the unchanged conda-forge ABI3 binary and licenses."""
    original = CACHE / "llguidance-1.7.6-py310hc9716df_0.conda"
    assert digest(original) == "b280507d74e8372a16669e8280858e3951eaf84b8dfb4200d0fa95015392831b"
    extracted = CACHE / "llguidance-conda"
    entries = json.loads((extracted / "info/paths.json").read_text())["paths"]
    files = {}
    for entry in entries:
        name = entry["_path"]
        if not name.startswith("site-packages/"):
            continue
        data = (extracted / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        rel = name[len("site-packages/"):]
        if rel.rsplit("/", 1)[-1] not in {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"}:
            files[rel] = data
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    for name in sorted(files):
        data = files[name]
        h = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        writer.writerow([name, "sha256=" + h, len(data)])
    record_name = "llguidance-1.7.6.dist-info/RECORD"
    writer.writerow([record_name, "", ""])
    files[record_name] = records.getvalue().encode()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, (2026, 9, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, files[name])
    return files


def jobs_and_paths():
    jobs = json.loads((ADDON / "wheel_sources.json").read_text())
    paths = {}
    for i, job in enumerate(jobs):
        path = (CACHE if job["name"] == "vllm" else CACHE / "wheelhouse-v013") / job["filename"]
        if not path.is_file() or Path(str(path) + ".aria2").exists():
            raise RuntimeError("Wheel download unfinished: " + str(path))
        if digest(path) != job["sha256"]:
            raise RuntimeError("Wheel checksum failed: " + str(path))
        paths[job["name"]] = path
        if i % 20 == 0:
            print(f"HASH_VERIFIED={i+1}/{len(jobs)}", flush=True)
    # Independent resolution from full actual vLLM METADATA must agree exactly.
    lock = tomllib.loads((CACHE / "pylock.fullv013.toml").read_text())
    selected = {canonicalize_name(j["name"]): j["version"] for j in jobs}
    locked = {canonicalize_name(p["name"]): p["version"] for p in lock["packages"]}
    assert selected == locked, (selected.items() - locked.items(), locked.items() - selected.items())
    return jobs, paths


def closure(jobs, paths):
    environment = default_environment()
    environment.update(sys_platform="linux", platform_system="Linux", os_name="posix", platform_machine="x86_64",
                       python_version="3.11", python_full_version="3.11.16", implementation_name="cpython", implementation_version="3.11.16")
    versions = {canonicalize_name(j["name"]): j["version"] for j in jobs}
    requirements = {}
    owners = {}
    collisions = []
    for job in jobs:
        with zipfile.ZipFile(paths[job["name"]]) as z:
            meta = BytesParser().parsebytes(z.read(next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))))
            assert canonicalize_name(meta["Name"]) == canonicalize_name(job["name"])
            assert meta["Version"] == job["version"]
            requirements[canonicalize_name(job["name"])] = [Requirement(r) for r in meta.get_all("Requires-Dist", [])]
            record = z.read(next(n for n in z.namelist() if n.endswith(".dist-info/RECORD"))).decode()
            for name, h, size in csv.reader(io.StringIO(record)):
                if ".dist-info/" in name or not h:
                    continue
                if name in owners and owners[name][1] != h:
                    collisions.append((name, owners[name][0], job["name"]))
                owners[name] = (job["name"], h)
    # Both upstream wheels accidentally ship their PEP517 build backend at the
    # site-packages root. Our wheels-only installer never invokes either backend.
    build_only = [("build_backend.py", "flashinfer-python", "torch-c-dlpack-ext")]
    assert all(c in build_only for c in collisions), f"Conflicting runtime files: {collisions[:20]}"
    for package in ("flashinfer-python", "torch-c-dlpack-ext"):
        with zipfile.ZipFile(paths[package]) as z:
            for name in z.namelist():
                if name.endswith(".py") and name != "build_backend.py":
                    assert b"import build_backend" not in z.read(name) and b"from build_backend" not in z.read(name), name
    needed = {"vllm": set(), "llguidance": set(), "transformers": set(), "pip": set()}
    done = {}
    while True:
        pending = [(n, extras.copy()) for n, extras in needed.items() if done.get(n) != extras]
        if not pending:
            break
        for name, extras in pending:
            done[name] = extras.copy()
            for req in requirements[name]:
                if req.marker and not any(req.marker.evaluate({**environment, "extra": extra}) for extra in {"", *extras}):
                    continue
                key = canonicalize_name(req.name)
                assert key in versions, f"Missing {req}, required by {name}"
                assert req.specifier.contains(versions[key], prereleases=True), f"Unsatisfied {name}: {req} vs {versions[key]}"
                needed.setdefault(key, set()).update(req.extras)
    assert set(needed) == set(versions), f"Unnecessary packages: {set(versions)-set(needed)}"
    return {"packages": len(jobs), "closure": "passed", "runtime_file_collisions": 0, "upstream_build_only_collisions": collisions,
            "active_extras": {n: sorted(e) for n, e in needed.items() if e}}


def source_files():
    files = {p.name: p.read_bytes() for p in ADDON.iterdir() if p.suffix in {".py", ".sh", ".md", ".json", ".txt"}}
    earlier = RELEASE / "finsight-thinking-v0.10.2.zip"
    assert digest(earlier) == earlier.with_suffix(".zip.sha256").read_text().split()[0]
    with zipfile.ZipFile(earlier) as z:
        manifest = json.loads(z.read("thinking_v0102/PACKAGE_MANIFEST.json"))
        for name, h in manifest["files"].items():
            if "/metrics/" in name or name.endswith("/contract.py"):
                content = z.read(name)
                assert hashlib.sha256(content).hexdigest() == h
                files[name[len("thinking_v0102/"):]] = content
    deps = {}
    with zipfile.ZipFile(RELEASE / "finsight-native-probe-v0.12.zip") as z:
        manifest = json.loads(z.read("native_probe_v012/PACKAGE_MANIFEST.json"))
        for name, h in manifest["files"].items():
            if name.startswith("payload/compat/"):
                deps["native_probe_v012/" + name] = h
        for alias, target in manifest["links"].items():
            while target in manifest["links"]:
                target = manifest["links"][target]
            deps["native_probe_v012/payload/compat/" + alias] = manifest["files"]["payload/compat/" + target]
    return files, deps


def build():
    jobs, paths = jobs_and_paths()
    audit = closure(jobs, paths)
    print("DEPENDENCY_AUDIT=" + json.dumps(audit), flush=True)
    # Verify the already-built llguidance wheel contains precisely the reproducible recipe's payload.
    repacked = CACHE / "llguidance-recipe-check.whl"
    expected = repack_llguidance(repacked)
    with zipfile.ZipFile(paths["llguidance"]) as z:
        assert set(z.namelist()) == set(expected)
        assert all(z.read(n) == b for n, b in expected.items() if not n.endswith("/RECORD"))
        # Both valid RECORD encodings may order their rows differently.
        actual_record = z.read("llguidance-1.7.6.dist-info/RECORD").decode()
        assert sorted(csv.reader(io.StringIO(actual_record))) == sorted(csv.reader(io.StringIO(expected["llguidance-1.7.6.dist-info/RECORD"].decode())))
    files, deps = source_files()
    lock_bytes = ("\n".join(f"{j['name']}=={j['version']} --hash=sha256:{j['sha256']}" for j in sorted(jobs, key=lambda j:j['name'])) + "\n").encode()
    assert files["requirements.lock.txt"] == lock_bytes
    files["WHEELS.json"] = (json.dumps({"audit": audit, "wheels": jobs, "llguidance": {
        "origin": "conda-forge/linux-64/llguidance-1.7.6-py310hc9716df_0.conda",
        "original_sha256": "b280507d74e8372a16669e8280858e3951eaf84b8dfb4200d0fa95015392831b",
        "repacked_sha256": LLG_HASH, "note": "Local repack of unchanged ABI3 binary, licenses and metadata; regenerated RECORD; not an official PyPI wheel."}}, indent=2) + "\n").encode()
    for name, b in files.items():
        if name.endswith(".py"):
            ast.parse(b)
        if name.endswith(".sh"):
            assert b"\r" not in b
    stage = CACHE / "stage-v013/vllm_v013"
    stage.mkdir(parents=True, exist_ok=True)
    for name, b in files.items():
        p = stage / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    if "--check" in sys.argv:
        import build_thinking_contract_patch as checks
        checks.PREFIX = "vllm_v013"
        checks.verify({"vllm_v013/" + n: b for n, b in files.items()})
        print("STAGED=" + str(stage), flush=True)
        return
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit before building"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
    hashes.update({"wheelhouse/" + j["filename"]: j["sha256"] for j in jobs})
    files["PACKAGE_MANIFEST.json"] = (json.dumps({"version": "0.13", "commit": commit, "files": hashes, "dependencies": deps}, indent=2) + "\n").encode()
    archives = [RELEASE / "finsight-vllm-v0.13.zip", RELEASE / "finsight-vllm-wheels-v0.13.zip"]
    with zipfile.ZipFile(archives[0], "w", zipfile.ZIP_DEFLATED) as z:
        for name, b in files.items():
            z.writestr("vllm_v013/" + name, b)
    with zipfile.ZipFile(archives[1], "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, job in enumerate(jobs):
            z.write(paths[job["name"]], "vllm_v013/wheelhouse/" + job["filename"])
            if i % 20 == 0:
                print(f"PACKED={i+1}/{len(jobs)}", flush=True)
    for path in archives:
        print("VERIFY_ARCHIVE=" + path.name, flush=True)
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith("/PACKAGE_MANIFEST.json"):
                    continue
                with z.open(name) as stream:
                    assert hashlib.file_digest(stream, "sha256").hexdigest() == hashes[name[len("vllm_v013/"):]]
        checksum = digest(path)
        path.with_suffix(".zip.sha256").write_bytes(f"{checksum}  {path.name}\n".encode())
        print(f"PACKAGE={path}\nBYTES={path.stat().st_size}\nSHA256={checksum}\nCOMMIT={commit}", flush=True)


if __name__ == "__main__":
    build()
