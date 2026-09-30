"""Install a verified v0.4 handoff ZIP, preserving every replaced file first."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


def checksum(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def target_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive path: ' + name)
    allowed = {'src', 'scripts', 'tests', 'docs', 'configs', 'data'}
    if not relative.parts or relative.parts[0] not in allowed:
        raise ValueError('Unexpected payload path: ' + name)
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Payload escapes project: ' + name)
    return target


def install(package, root):
    root = root.resolve()
    if not (root / 'server/run.sh').exists():
        raise ValueError('Run from the existing FinSight project directory')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    with zipfile.ZipFile(package) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate ZIP members')
        manifest = json.loads(archive.read('FORMAL_DATA_MANIFEST.json'))
        if manifest['version'] != '0.4':
            raise ValueError('Unexpected patch version')
        files = manifest['files']
        expected = {'payload/' + name for name in files} | {'FORMAL_DATA_MANIFEST.json', 'apply_v04.py'}
        if set(archive.namelist()) != expected:
            raise ValueError('ZIP members differ from manifest')
        targets = {name: target_path(root, name) for name in files}
        changed = [name for name, path in targets.items() if not path.exists() or checksum(path) != files[name]]
        if (root / 'data/frozen/manifest.json').exists() and any(n.startswith(('src/', 'data/', 'configs/')) for n in changed):
            raise ValueError('A formal dataset is already frozen. Preserve/version that experiment before replacing its source or data.')
        (root / '.runtime').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='v04-staging-', dir=root / '.runtime') as staging_dir:
            staging = Path(staging_dir)
            for index, (name, expected_hash) in enumerate(files.items(), 1):
                staged = staging / name
                staged.parent.mkdir(parents=True, exist_ok=True)
                with archive.open('payload/' + name) as source, staged.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                if checksum(staged) != expected_hash:
                    raise ValueError('Payload checksum mismatch: ' + name)
                if index % 25 == 0:
                    print(f'PATCH_VERIFY={index}/{len(files)}', flush=True)
            backup = root / 'outputs/patch_backups' / ('pre-v0.4-' + stamp)
            if changed:
                backup.mkdir(parents=True)
                for name in changed:
                    target = targets[name]
                    if target.exists():
                        saved = backup / name
                        saved.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(target, saved)
                if (root / 'data/pilot').exists():
                    shutil.copytree(root / 'data/pilot', backup / 'data/pilot')
                (backup / 'install_plan.json').write_text(json.dumps({'changed': changed, 'manifest': manifest}, indent=2), encoding='utf-8')
                print('BACKUP=' + str(backup), flush=True)
                for name in changed:
                    targets[name].parent.mkdir(parents=True, exist_ok=True)
                    (staging / name).replace(targets[name])
            receipt = {'status': 'installed', 'version': '0.4', 'changed_files': len(changed),
                       'backup': str(backup) if changed else None, 'package_sha256': checksum(package),
                       'files': files, 'notice': 'Old pilot data remain historical and cannot train under the new source/fact digest.'}
            receipt_path = root / 'outputs/installations' / ('v0.4-' + stamp + '.json')
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(f'PATCH_INSTALLED=0.4 CHANGED={len(changed)}\nRECEIPT={receipt_path}', flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=Path)
    args = parser.parse_args()
    import fcntl
    root = Path.cwd()
    (root / '.runtime').mkdir(exist_ok=True)
    with (root / '.runtime/gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        install(args.package, root)


if __name__ == '__main__':
    main()
