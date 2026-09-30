import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

import pytest

from finsight_ashare.common import ROOT

pytestmark = pytest.mark.skipif(platform.system() != "Linux", reason="Linux shell integration")


def installer_fixture(root, content=b"fixture", checksum=None):
    (root / "server/environment").mkdir(parents=True)
    shutil.copyfile(ROOT / "server/install_env.sh", root / "server/install_env.sh")
    archive = root / "fixture.tar.gz"
    archive.write_bytes(content)
    digest = checksum or hashlib.sha256(content).hexdigest()
    (root / "server/environment/finsight-conda-linux-x86_64-cu126-v0.1.tar.gz.sha256").write_text(digest + "  archive.tar.gz\n")
    return ["bash", str(root / "server/install_env.sh"), str(archive)]


def test_checksum_failure_does_not_extract(tmp_path):
    result = subprocess.run(installer_fixture(tmp_path, checksum="a" * 64), capture_output=True, text=True)
    assert result.returncode == 2
    assert "SHA256 mismatch" in result.stdout
    assert not (tmp_path / ".runtime/envs/finsight").exists()


def test_installer_preserves_unknown_existing_environment(tmp_path):
    command = installer_fixture(tmp_path)
    prefix = tmp_path / ".runtime/envs/finsight"
    prefix.mkdir(parents=True)
    valuable = prefix / "existing-file.txt"
    valuable.write_text("preserve this")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Refusing to overwrite" in result.stdout
    assert valuable.read_text() == "preserve this"


def test_entrypoint_refuses_partial_environment(tmp_path):
    (tmp_path / "server").mkdir()
    shutil.copyfile(ROOT / "server/run.sh", tmp_path / "server/run.sh")
    (tmp_path / ".runtime/envs/finsight").mkdir(parents=True)
    result = subprocess.run(["bash", str(tmp_path / "server/run.sh"), "doctor"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "installation is incomplete" in result.stderr


def test_entrypoint_prefers_portable_python_and_isolates_paths(tmp_path):
    import sys
    (tmp_path / "server").mkdir()
    shutil.copyfile(ROOT / "server/run.sh", tmp_path / "server/run.sh")
    prefix = tmp_path / ".runtime/envs/finsight"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").symlink_to(sys.executable)
    (prefix / ".finsight-ready.json").write_text("{}")
    (tmp_path / "server/runner.py").write_text(
        "import json, os, sys\nprint(json.dumps({'python': sys.executable, 'path': os.environ['PATH'], "
        "'pythonpath': os.environ['PYTHONPATH'], 'no_user_site': os.environ['PYTHONNOUSERSITE']}))\n")
    env = {**os.environ, "PYTHONHOME": "/wrong-python", "PYTHONPATH": "/wrong-packages"}
    result = subprocess.run(["bash", str(tmp_path / "server/run.sh"), "doctor"], env=env, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["python"] == str(prefix / "bin/python")
    assert report["path"].split(":")[0] == str(prefix / "bin")
    assert report["pythonpath"] == str(tmp_path / "src")
    assert report["no_user_site"] == "1"
