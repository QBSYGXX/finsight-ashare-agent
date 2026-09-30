import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finsight_ashare import server


def prepared_prefix(root, profile):
    prefix = root / ".runtime/envs/finsight"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").write_text("fixture", encoding="utf-8")
    (prefix / ".finsight-ready.json").write_text(json.dumps({
        "status": "ready", "prefix": str(prefix.resolve()), "profile": profile,
        "archive_sha256": "a" * 64}), encoding="utf-8")
    return prefix / "bin/python"


def test_incomplete_or_moved_environment_cannot_be_used(tmp_path):
    profile = {"python": "3.11"}
    assert server.portable_python(tmp_path, profile) is None
    python = prepared_prefix(tmp_path, profile)
    assert server.portable_python(tmp_path, profile) == python
    marker = python.parent.parent / ".finsight-ready.json"
    receipt = json.loads(marker.read_text())
    receipt["prefix"] = str(tmp_path / "old-location")
    marker.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="profile/path mismatch"):
        server.portable_python(tmp_path, profile)
    marker.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        server.portable_python(tmp_path, profile)


def test_portable_setup_never_creates_venv_or_installs_packages(tmp_path, monkeypatch):
    profile = {"python": "3.11"}
    python = prepared_prefix(tmp_path, profile)
    for directory in ("configs", "exports", "server/environment"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/server-profile.json").write_text(json.dumps(profile))
    (tmp_path / "exports/doctor.json").write_text(json.dumps({"profile": profile, "ready_for_candidate_setup": True}))
    (tmp_path / "server/environment/finsight-conda-linux-x86_64-cu126-v0.1.tar.gz.sha256").write_text("a" * 64 + "  env.tar.gz\n")
    commands = []
    monkeypatch.setattr(server.platform, "system", lambda: "Linux")
    monkeypatch.setattr(server, "run_command", lambda command, *args: commands.append(list(map(str, command))))
    monkeypatch.setattr(server.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="torch==2.8.0\n"))
    server.setup(tmp_path)
    assert all(command[0] == str(python) for command in commands)
    assert not any("install" in command or "venv" in command for command in commands)
    assert any("_download" in command for command in commands)
    assert not (tmp_path / ".venv").exists()
    assert json.loads((tmp_path / "outputs/environment/setup.json").read_text())["environment_source"] == "portable_conda"
