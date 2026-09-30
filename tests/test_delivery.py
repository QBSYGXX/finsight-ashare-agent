import json
from pathlib import Path
import zipfile

from finsight_ashare.common import ROOT, file_hash
from finsight_ashare.packaging import build_release, release_files, export_results


def test_release_excludes_private_artifacts(tmp_path):
    for relative in ["README.md", "server/run.sh", "src/app.py", ".env", "models/model.safetensors", "data/raw/report.pdf",
                     "outputs/private.json", "runs/task/run.log", "src/__pycache__/app.pyc", "vendor/FinSight/.env.example",
                     "server/environment/runtime.tar.gz", "server/.runtime/envs/python", "server/runtime.tar.gz.partial"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    archive = build_release(tmp_path / "release.zip", tmp_path)
    with zipfile.ZipFile(archive) as result:
        assert set(result.namelist()) == {"finsight-ashare-agent/README.md", "finsight-ashare-agent/server/run.sh",
                                        "finsight-ashare-agent/src/app.py", "finsight-ashare-agent/RELEASE_MANIFEST.json"}


def test_vendor_files_match_pinned_source():
    manifest = json.loads((ROOT / "vendor/manifest.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == "8fc5bce28ac201c713219a47d30367c6a84342f4"
    for name, digest in manifest["files"].items():
        assert file_hash(ROOT / "vendor/FinSight" / name) == digest


def test_shell_entrypoint_uses_lf():
    for path in (ROOT / "server").rglob("*.sh"):
        assert b"\r\n" not in path.read_bytes(), path


def test_result_export_omits_base_weights(tmp_path):
    for relative in ["outputs/training/full/adapter_model.safetensors", "outputs/accidental/model.safetensors",
                     "models/Qwen3.5-9B/model.safetensors"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    exported = export_results(tmp_path)
    with zipfile.ZipFile(exported) as archive:
        assert "outputs/training/full/adapter_model.safetensors" in archive.namelist()
        assert not any(name.endswith("/model.safetensors") for name in archive.namelist())
