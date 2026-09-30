"""模块：实验边界；作用：固定输入、路径和产物身份；核心业务：否。"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / 'lora_v0161'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def run_path(run):
    if not isinstance(run, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run):
        raise ValueError('Invalid run ID')
    return ROOT / 'outputs/lora_v0161' / run


# 模块：输入冻结；作用：绑定当前Agent、数据、模板及模型权重；核心业务：否。
def verify():
    from agent_v015.storage import verify as verify_agent
    verify_agent(ROOT)
    pin = read(ADDON/'PACKAGE_MANIFEST.json')
    for name, expected in {**pin['files'], **pin['dependencies']}.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError('LoRA package/dependency mismatch: '+name)
    return pin


def model_fingerprint():
    model = ROOT/'models/Qwen3.5-9B'
    index = read(model/'model.safetensors.index.json')
    names = sorted(set(index['weight_map'].values()) | {
        'config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json'})
    for p in model.glob('*.jinja'):
        names.append(p.name)
    result = {}
    for name in names:
        print('HASH_MODEL='+name, flush=True)
        result[name] = sha(model/name)
    return result


def experiment_identity(pin, model_hashes):
    if any(model_hashes.get(n)!=h for n,h in pin['base_model_shards'].items()):
        raise ValueError('Base model weights differ from the pinned local snapshot')
    return digest({'package': pin, 'model': model_hashes, 'config': read(ADDON/'config.json')})
