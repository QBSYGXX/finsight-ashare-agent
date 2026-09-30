"""模块：离线发布；作用：打包新协议代码和训练轨迹，固定原系统依赖；核心业务：否。"""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lora_v016.common import ADDON,sha,save,read


def main():
    target=ROOT.parent/'_release_work/finsight-lora-v0.16.zip'
    if target.exists() or target.with_suffix('.zip.sha256').exists():raise FileExistsError('Published package exists')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit tracked source changes before pinning a release')
    files=sorted(p for p in ADDON.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
                 p.name!='PACKAGE_MANIFEST.json' and p.suffix in {'.py','.sh','.md','.json','.jsonl'})
    required=['data/train.jsonl','data/trajectories.json','data/audit.json','config.json','README.md']
    if any(not (ADDON/n).exists() for n in required):raise ValueError('Missing release artifact')
    # Preserve the exact frozen Agent and the old runtime, including native tokenizer metadata.
    deps=['agent_v015/PACKAGE_MANIFEST.json','vllm_v013/runtime.py','vllm_v013/adapter.py',
          'models/Qwen3.5-9B/config.json','models/Qwen3.5-9B/tokenizer_config.json','models/Qwen3.5-9B/tokenizer.json',
          'models/Qwen3.5-9B/model.safetensors.index.json']
    deps += [p.relative_to(ROOT).as_posix() for p in (ROOT/'models/Qwen3.5-9B').glob('*.jinja')]
    pin={'protocol':'lora-v016-1','base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'files':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
         'dependencies':{n:sha(ROOT/n) for n in deps},
         'base_model_shards':{},'frozen_agent':'agent-v015.5','test_companies_used':False}
    shards=sorted(set(read(ROOT/'models/Qwen3.5-9B/model.safetensors.index.json')['weight_map'].values()))
    for n in shards:
        print('PIN_MODEL='+n,flush=True)
        pin['base_model_shards'][n]=sha(ROOT/'models/Qwen3.5-9B'/n)
    save(ADDON/'PACKAGE_MANIFEST.json',pin)
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ADDON/'PACKAGE_MANIFEST.json']:z.write(p,p.relative_to(ROOT).as_posix())
    target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n',encoding='ascii',newline='\n')
    with zipfile.ZipFile(target) as z:
        import hashlib
        assert z.testzip() is None
        assert set(z.namelist())==set(pin['files'])|{'lora_v016/PACKAGE_MANIFEST.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in pin['files'].items())
    print(f'ZIP={target}\nBYTES={target.stat().st_size}\nSHA256={sha(target)}',flush=True)


if __name__=='__main__':main()
