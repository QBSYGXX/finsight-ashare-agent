"""Build rootless compatibility gates from hash-verified upstream binaries."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/vllm-offline"
PREFIX = "native_probe_v012"
RELEASE = ROOT.parent / "_release_work"


# 模块：第三方打包；作用：保留来源哈希、许可与符号链接关系；核心业务：否。
def main():
    import rpmfile
    from elftools.elf.elffile import ELFFile
    from packaging.version import Version
    import io
    files = {p.name: p.read_bytes() for p in (ROOT / PREFIX).iterdir() if p.suffix in {".py", ".sh", ".md"}}
    catalog = json.loads((CACHE / "llguidance-conda.json").read_text())
    llg = next(f for f in catalog["files"] if f["basename"] == "linux-64/llguidance-1.7.6-py310hc9716df_0.conda")
    original = CACHE / Path(llg["basename"]).name
    assert hashlib.file_digest(original.open("rb"), "sha256").hexdigest() == llg["sha256"]
    # Verify every packaged byte against the conda package's own content hashes.
    extracted = CACHE / "llguidance-conda"
    paths = json.loads((extracted / "info/paths.json").read_text())["paths"]
    for entry in paths:
        name = entry["_path"]
        p = extracted / name
        if name.startswith("site-packages/"):
            data = p.read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            files["payload/" + name] = data
    for p in (extracted / "info/licenses").rglob("*"):
        if p.is_file():
            files["licenses/llguidance/" + p.name] = p.read_bytes()
    selected = json.loads((CACHE / "compat-selected.json").read_text())
    rpm_path = CACHE / Path(selected["path"]).name
    assert hashlib.file_digest(rpm_path.open("rb"), "sha256").hexdigest() == selected["sha256"]
    links = {}
    with rpmfile.open(rpm_path) as rpm:
        for member in rpm.getmembers():
            if "/compat/" not in member.name:
                continue
            data = rpm.extractfile(member).read()
            name = Path(member.name).name
            if member.issymlink:
                links[name] = data.decode()
            else:
                files["payload/compat/" + name] = data
    elf_checks = []
    for name, data in files.items():
        if data.startswith(b"\x7fELF"):
            elf = ELFFile(io.BytesIO(data))
            assert elf.header["e_machine"] == "EM_X86_64"
            versions = elf.get_section_by_name(".gnu.version_r")
            needed = [aux.name for _, values in versions.iter_versions() for aux in values] if versions else []
            glibc = [Version(v[6:]) for v in needed if v.startswith("GLIBC_")]
            assert not glibc or max(glibc) <= Version("2.28"), name
            elf_checks.append({"path": name, "maximum_glibc_symbol": str(max(glibc)) if glibc else None})
        if name.endswith(".py"):
            ast.parse(data)
        if name.endswith(".sh"):
            assert b"\r" not in data
    provenance = {"llguidance": {"version": "1.7.6", "url": "https://conda.anaconda.org/conda-forge/" + llg["basename"],
        "sha256": llg["sha256"], "license": "MIT", "dependencies": llg["attrs"]["depends"]},
        "cuda_compat": {**selected, "url": "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/" + selected["path"],
            "license": "NVIDIA License", "license_url": "https://docs.nvidia.com/cuda/eula/index.html"},
        "elf_checks": elf_checks, "execution": "Not tested on server; run the included native probe."}
    files["THIRD_PARTY.json"] = (json.dumps(provenance, indent=2) + "\n").encode()
    assert all(t in {Path(n).name for n in files if n.startswith("payload/compat/")} or t in links for t in links.values())
    if "--check" in sys.argv:
        print(json.dumps(provenance, indent=2))
        print("UPSTREAM_HASHES_CONTENT_HASHES_ELF_GLIBC=PASS")
        return
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit before packaging"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files["PACKAGE_MANIFEST.json"] = (json.dumps({"version": "0.12", "commit": commit, "links": links,
        "files": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}, indent=2) + "\n").encode()
    output = RELEASE / "finsight-native-probe-v0.12.zip"
    print("PACKAGING=" + str(output), flush=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(PREFIX + "/" + n, b)
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for name, data in files.items():
            assert z.read(PREFIX + "/" + name) == data
    checksum = hashlib.file_digest(output.open("rb"), "sha256").hexdigest()
    output.with_suffix(".zip.sha256").write_bytes(f"{checksum}  {output.name}\n".encode())
    print(f"PACKAGE={output}\nBYTES={output.stat().st_size}\nSHA256={checksum}\nCOMMIT={commit}", flush=True)
    if "--bundle-ui" in sys.argv:
        ui = RELEASE / "finsight-thinking-v0.10.2.zip"
        assert hashlib.sha256(ui.read_bytes()).hexdigest() == ui.with_suffix(".zip.sha256").read_text().split()[0]
        bundle = RELEASE / "finsight-rootless-step1-v0.12.zip"
        expected = {}
        print("BUNDLING=" + str(bundle), flush=True)
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as target:
            for source, prefix in ((output, PREFIX), (ui, "thinking_v0102")):
                with zipfile.ZipFile(source) as original:
                    assert original.testzip() is None
                    for name in original.namelist():
                        assert name.startswith(prefix + "/") and ".." not in Path(name).parts and name not in expected
                        b = original.read(name)
                        expected[name] = hashlib.sha256(b).hexdigest()
                        target.writestr(name, b)
        with zipfile.ZipFile(bundle) as z:
            assert z.testzip() is None and set(z.namelist()) == set(expected)
            assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in expected.items())
        with bundle.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        bundle.with_suffix(".zip.sha256").write_bytes(f"{checksum}  {bundle.name}\n".encode())
        print(f"BUNDLE={bundle}\nBYTES={bundle.stat().st_size}\nSHA256={checksum}", flush=True)


if __name__ == "__main__":
    main()
