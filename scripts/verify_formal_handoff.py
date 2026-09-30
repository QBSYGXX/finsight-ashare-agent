"""Exercise the real ZIP installer and audit in an isolated local project."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from finsight_ashare.common import file_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('patch_installer', ROOT / 'scripts/apply_formal_data_patch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = ROOT / 'outputs/handoff_v0.4'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.log').open('w', encoding='utf-8') as log:
        with tempfile.TemporaryDirectory(prefix='install-', dir=output) as temporary:
            target = Path(temporary)
            (target / 'server').mkdir()
            shutil.copy2(ROOT / 'server/run.sh', target / 'server/run.sh')
            first = module.install(args.package, target)
            print('ISOLATED_INSTALL=PASS', flush=True)
            env = os.environ.copy()
            env.update(FINSIGHT_ROOT=str(target), PYTHONPATH=str(target / 'src'), PYTHONUTF8='1')
            command = [sys.executable, '-X', 'utf8', '-c',
                       'from finsight_ashare.data import audit; from finsight_ashare.common import ROOT; assert len(audit(ROOT)) == 216']
            process = subprocess.run(command, cwd=target, env=env, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f'Isolated audit exit={process.returncode}; see {output / "run.log"}')
            print('ISOLATED_AUDIT=216 PASS', flush=True)
            second = module.install(args.package, target)
            if second['changed_files'] != 0:
                raise ValueError('Repeated installation changed files')
            checksum_text = args.package.with_suffix('.zip.sha256').read_bytes()
            if b'\r' in checksum_text or checksum_text.decode('ascii').split()[0] != file_hash(args.package):
                raise ValueError('Invalid checksum sidecar')
            write_json(output / 'verification.json', {
                'package': args.package.name, 'sha256': file_hash(args.package),
                'first_install_files': first['changed_files'], 'audit_facts': 216,
                'audit_exit_code': process.returncode, 'repeated_install_changes': second['changed_files'],
                'checksum_sidecar_lf': True, 'platform': sys.platform,
                'limits': 'Installer function and installed-code audit tested locally; Linux CLI lock and actual server tokenization not run.'})
    print('HANDOFF_VERIFIED=' + str(output / 'verification.json'), flush=True)


if __name__ == '__main__':
    main()
