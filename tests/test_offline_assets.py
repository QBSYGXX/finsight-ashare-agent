import json
from types import SimpleNamespace
import sys

import pytest

from finsight_ashare import assets, server
from finsight_ashare.common import file_hash, read_json, write_json
from test_portable_environment import prepared_prefix


def test_curl_incomplete_download_does_not_publish(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_url=lambda *a, **kw: "https://example.com/weight"))
    monkeypatch.setattr(assets.shutil, "which", lambda *a: "curl")
    def incomplete(*args, **kwargs):
        (tmp_path / "pytorch_model.bin.partial").write_bytes(b"short")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(assets.subprocess, "run", incomplete)
    item = SimpleNamespace(rfilename="pytorch_model.bin", size=100, lfs=None)
    with pytest.raises(ValueError, match="Incomplete transfer"):
        assets.curl_snapshot_file("BAAI/bge-m3", "a" * 40, item, tmp_path)
    assert not (tmp_path / "pytorch_model.bin").exists()
    assert (tmp_path / "pytorch_model.bin.partial").exists()


def test_curl_tls_interruption_resumes_partial(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_url=lambda *a, **kw: "https://example.com/weight"))
    monkeypatch.setattr(assets.shutil, "which", lambda *a: "curl")
    monkeypatch.setattr(assets.time, "sleep", lambda *a: None)
    partial = tmp_path / "pytorch_model.bin.partial"
    calls = []
    def download(command, **kwargs):
        calls.append(command)
        assert "--continue-at" in command
        if len(calls) == 1:
            partial.write_bytes(b"first")
            return SimpleNamespace(returncode=56)
        assert partial.read_bytes() == b"first"
        with partial.open("ab") as stream:
            stream.write(b"second")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(assets.subprocess, "run", download)
    item = SimpleNamespace(rfilename="pytorch_model.bin", size=11, lfs=None)
    assets.curl_snapshot_file("BAAI/bge-m3", "a" * 40, item, tmp_path)
    assert len(calls) == 2
    assert (tmp_path / "pytorch_model.bin").read_bytes() == b"firstsecond"


def test_sparse_segmented_partial_cannot_be_published_as_sequential(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_url=lambda *a, **kw: "https://example.com/weight"))
    monkeypatch.delenv("FINSIGHT_ARIA2", raising=False)
    (tmp_path / "pytorch_model.bin.partial").write_bytes(b"\0" * 100)
    (tmp_path / "pytorch_model.bin.partial.aria2").write_bytes(b"resume state")
    item = SimpleNamespace(rfilename="pytorch_model.bin", size=100, lfs=None)
    with pytest.raises(ValueError, match="Segmented partial"):
        assets.curl_snapshot_file("BAAI/bge-m3", "a" * 40, item, tmp_path)
    assert not (tmp_path / "pytorch_model.bin").exists()


@pytest.fixture
def snapshots(project):
    lock = {}
    for repo, folder in assets.model_specs(project):
        names = {"config.json": {}, "tokenizer_config.json": {}, "pytorch_model.bin": {}}
        if folder == "bge-m3":
            names.update({"modules.json": [], "1_Pooling/config.json": {}})
        entry = {"repo": repo, "revision": "a" * 40, "status": "downloaded", "files": {}}
        for name, content in names.items():
            path = project / "models" / folder / name
            write_json(path, content)
            entry["files"][name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        lock[folder] = entry
    write_json(project / "models/snapshots.json", lock)
    return project


def test_bge_bin_included_and_corruption_rejected(snapshots):
    assert assets.model_file_selected("pytorch_model.bin")
    assert assets.model_file_selected("1_Pooling/config.json")
    assert not assets.model_file_selected("../config.json")
    assets.verify_snapshots(snapshots)
    (snapshots / "models/bge-m3/pytorch_model.bin").write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity"):
        assets.verify_snapshots(snapshots)


def test_indexed_missing_shard_rejected(snapshots):
    path = snapshots / "models/Qwen3.5-9B/model.safetensors.index.json"
    write_json(path, {"weight_map": {"weight": "missing.safetensors"}})
    with pytest.raises(ValueError, match="missing indexed"):
        assets.verify_snapshots(snapshots)


def test_missing_pooling_rejected(snapshots):
    path = snapshots / "models/snapshots.json"
    lock = read_json(path)
    del lock["bge-m3"]["files"]["1_Pooling/config.json"]
    write_json(path, lock)
    with pytest.raises(ValueError, match="pooling"):
        assets.verify_snapshots(snapshots)


def test_old_driver_needs_complete_fresh_probe():
    profile = {"minimum_driver_major": 560, "torch_index": "https://download.pytorch.org/whl/cu126", "gpus": 4}
    probe = {"passed": True, "cuda_runtime": "12.6", "gpu_count": 4, "devices": [{"passed": True}] * 4}
    assert server.driver_compatible("535.86.10", profile, probe)
    assert not server.driver_compatible("525.60.12", profile, probe)
    assert not server.driver_compatible("535.86.10", profile, {})
    assert not server.driver_compatible("535.86.10", profile, {**probe, "passed": False})
    assert not server.driver_compatible("535.86.10", profile, {**probe, "devices": [{"passed": True}]})


def test_offline_setup_requires_portable_environment(project, monkeypatch):
    profile = {"python": "3.11"}
    write_json(project / "configs/server-profile.json", profile)
    write_json(project / "exports/doctor.json", {"profile": profile, "ready_for_candidate_setup": True})
    monkeypatch.setattr(server.platform, "system", lambda: "Linux")
    monkeypatch.setattr(server, "run_command", lambda *a: pytest.fail("Must not install/download"))
    with pytest.raises(ValueError, match="verified portable"):
        server.setup(project, offline=True)


def test_offline_setup_propagates_offline(project, monkeypatch):
    profile = {"python": "3.11"}
    prepared_prefix(project, profile)
    write_json(project / "configs/server-profile.json", profile)
    write_json(project / "exports/doctor.json", {"profile": profile, "ready_for_candidate_setup": True})
    checksum = project / "server/environment/finsight-conda-linux-x86_64-cu126-v0.1.tar.gz.sha256"
    checksum.parent.mkdir(parents=True)
    checksum.write_text("a" * 64)
    commands = []
    monkeypatch.setattr(server.platform, "system", lambda: "Linux")
    monkeypatch.setattr(server, "run_command", lambda command, *a: commands.append(list(map(str, command))))
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="torch==2.8.0\n"))
    server.setup(project, offline=True)
    assert not any("install" in c or "venv" in c for c in commands)
    assert any(c[-2:] == ["_download", "--offline"] for c in commands)


def test_offline_doctor_skips_network_and_uses_current_probe(project, monkeypatch):
    profile = {"gpus": 4, "minimum_driver_major": 560, "torch_index": "https://download.pytorch.org/whl/cu126",
               "minimum_gpu_memory_mib": 38000, "minimum_free_disk_gib": 1}
    write_json(project / "configs/server-profile.json", profile)
    prepared_prefix(project, profile)
    monkeypatch.setattr(server.platform, "system", lambda: "Linux")
    monkeypatch.setattr(server.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Offline network call"))
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout="\n".join(f"{i}, A800, 81920, 81046, 535.86.10" for i in range(4))))
    probe = {"passed": True, "cuda_runtime": "12.6", "gpu_count": 4, "devices": [{"passed": True}] * 4}
    monkeypatch.setattr(server, "cuda_probe", lambda *a: probe)
    assert server.doctor(project, offline=True)["ready_for_candidate_setup"]
    probe["passed"] = False
    assert not server.doctor(project, offline=True)["ready_for_candidate_setup"]
