"""Build isolated compatibility repair, retaining the exact v0.16 dataset and dependency pins."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lora_v0161.common import ADDON,read,sha,save


def main():
    original=read(ROOT/'lora_v016/PACKAGE_MANIFEST.json')
    for n,h in {**original['files'],**original['dependencies']}.items():
        if sha(ROOT/n)!=h:raise ValueError('Baseline changed: '+n)
    for n in ('config.json','data/train.jsonl','data/trajectories.json','data/audit.json'):
        assert (ADDON/n).read_bytes()==(ROOT/'lora_v016'/n).read_bytes(),n
    files=sorted(p for p in ADDON.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
                 p.name!='PACKAGE_MANIFEST.json' and p.suffix in {'.py','.sh','.md','.json','.jsonl'})
    for p in files:
        b=p.read_bytes()
        if p.suffix=='.py':ast.parse(b)
        if p.suffix=='.sh':assert b'\r' not in b
    if '--check' in sys.argv:
        print('BASELINE_SOURCE_DEPENDENCIES_DATA_RECIPE_AST_LF=PASS')
        return
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit all source before release')
    target=ROOT.parent/'_release_work/finsight-lora-v0.16.1.zip'
    if target.exists() or target.with_suffix('.zip.sha256').exists():raise FileExistsError('Published release exists')
    pin={**original,'protocol':'lora-v0161-1',
         'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'derived_from_manifest_sha256':sha(ROOT/'lora_v016/PACKAGE_MANIFEST.json'),
         'files':{p.relative_to(ROOT).as_posix():sha(p) for p in files}}
    save(ADDON/'PACKAGE_MANIFEST.json',pin)
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ADDON/'PACKAGE_MANIFEST.json']:z.write(p,p.relative_to(ROOT).as_posix())
    target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n',encoding='ascii',newline='\n')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(pin['files'])|{'lora_v0161/PACKAGE_MANIFEST.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in pin['files'].items())
    print(f'ZIP={target}\nBYTES={target.stat().st_size}\nSHA256={sha(target)}\nMEMBERS={len(files)+1}')


if __name__=='__main__':main()
