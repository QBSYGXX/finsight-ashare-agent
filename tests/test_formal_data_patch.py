import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location('formal_patch', Path(__file__).resolve().parents[1] / 'scripts/apply_formal_data_patch.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def package(tmp_path, content=b'new', expected=None):
    path = tmp_path / 'patch.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('apply_v04.py', '')
        archive.writestr('payload/src/example.py', content)
        archive.writestr('FORMAL_DATA_MANIFEST.json', json.dumps({'version': '0.4', 'files': {
            'src/example.py': expected or hashlib.sha256(content).hexdigest()}}))
    (tmp_path / 'server').mkdir(exist_ok=True)
    (tmp_path / 'server/run.sh').touch()
    return path


def test_install_preserves_old_file_and_repeat_is_noop(tmp_path):
    path = package(tmp_path)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/example.py').write_bytes(b'old')
    receipt = module.install(path, tmp_path)
    assert (Path(receipt['backup']) / 'src/example.py').read_bytes() == b'old'
    assert (tmp_path / 'src/example.py').read_bytes() == b'new'
    assert module.install(path, tmp_path)['changed_files'] == 0


def test_corrupt_payload_does_not_replace_source(tmp_path):
    path = package(tmp_path, expected='0' * 64)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/example.py').write_bytes(b'old')
    with pytest.raises(ValueError, match='checksum mismatch'):
        module.install(path, tmp_path)
    assert (tmp_path / 'src/example.py').read_bytes() == b'old'


def test_existing_formal_freeze_blocks_mutation(tmp_path):
    path = package(tmp_path)
    (tmp_path / 'data/frozen').mkdir(parents=True)
    (tmp_path / 'data/frozen/manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='already frozen'):
        module.install(path, tmp_path)
    assert not (tmp_path / 'src/example.py').exists()


@pytest.mark.parametrize('name', ['../bad', '/absolute', 'data/../../bad', 'models/weights.bin', 'src/C:bad'])
def test_patch_paths_are_confined(tmp_path, name):
    with pytest.raises(ValueError):
        module.target_path(tmp_path, name)
