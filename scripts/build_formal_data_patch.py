"""Package reviewed PDFs/data and source without models, environment or old runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from finsight_ashare.common import file_hash, now, read_json, source_digest


def build(output):
    receipt = read_json(ROOT / 'data/review_v0.4/review_receipt.json')
    if source_digest(ROOT) != receipt['source_digest']:
        raise ValueError('Source changed after review/replay; rerun the local review verification')
    for name, expected in receipt['files'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Reviewed snapshot changed: ' + name)
    if receipt['facts'] != 216 or receipt['all_task_replays_passed'] != 1243:
        raise ValueError('Review/replay receipt is incomplete')
    files = set()
    for folder, glob in [('src/finsight_ashare', '*.py'), ('tests', '*.py'), ('docs', '*.md'),
                         ('data/raw', '*.pdf'), ('data/review_v0.4', '*.json')]:
        files.update(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).glob(glob))
    files.update(['configs/project.json', 'configs/lora.yaml', 'data/audit.csv',
                  'data/manifest.json', 'data/processed/pages.jsonl', 'data/verified/facts.json',
                  'data/verified/audit_receipt.json', 'data/review_v0.4/audit.csv'])
    files.update('scripts/' + name for name in ['apply_formal_data_patch.py', 'build_formal_data_patch.py',
                 'review_financial_statements.py', 'render_statement_review.py', 'finalize_review_v04.py',
                 'verify_server_export.py', 'fetch_report_corpus.py'])
    files = sorted(files)
    manifest = {'version': '0.4', 'created': now(), 'requires': 'Installed v0.3.3 plus existing offline model/environment',
                'notice': 'Private server handoff; PDFs and experiment data do not belong in the public Git repository.',
                'facts': 216, 'reports': 36, 'server_tokenization_pending': True,
                'files': {name: file_hash(ROOT / name) for name in files}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / 'scripts/apply_formal_data_patch.py', 'apply_v04.py')
        archive.writestr('FORMAL_DATA_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        for index, name in enumerate(files, 1):
            archive.write(ROOT / name, 'payload/' + name)
            if index % 25 == 0:
                print(f'PACK={index}/{len(files)}', flush=True)
    print('Verifying ZIP CRC and all payload SHA256 hashes...', flush=True)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC failed')
        for name, expected in manifest['files'].items():
            if hashlib.sha256(archive.read('payload/' + name)).hexdigest() != expected:
                raise ValueError('ZIP hash failed: ' + name)
    temporary.replace(output)
    checksum = file_hash(output)
    output.with_suffix('.zip.sha256').write_bytes((checksum + '  ' + output.name + '\n').encode('ascii'))
    print(f'PACKAGE={output}\nBYTES={output.stat().st_size}\nPAYLOAD_FILES={len(files)}\nSHA256={checksum}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
