"""模块：训练标签；作用：复用推理模板，只监督当前动作和结束符；核心业务：是。"""
import json
from collections import Counter

from .common import ROOT, ADDON, read, save, sha, verify, model_fingerprint, experiment_identity
from .corpus import blocked
from agent_v015.action_contract import parse_envelope


def encode_sample(tokenizer, row, config):
    messages, target = row['messages'], row['target']
    if messages[0]['role']!='system' or messages[-1]['role']!='user':
        raise ValueError('Training state must match the live next-action interface')
    action, notes = parse_envelope(target)
    if notes or set(json.loads(target))!={'tool','arguments'}:
        raise ValueError('Only canonical actions can be supervised')
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'):
        raise ValueError('Native non-thinking template changed')
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt+target+'<|im_end|>', add_special_tokens=False)
    if ids[:len(prefix)]!=prefix:
        raise ValueError('Tokenizer boundary crosses the prompt/target; refusing a shifted mask')
    targets = ids[len(prefix):]
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if not targets or targets[-1]!=eos or tokenizer.decode(targets,skip_special_tokens=False)!=target+'<|im_end|>':
        raise ValueError('Action/EOS roundtrip mismatch')
    if len(ids)>config['max_length'] or len(targets)>config['max_action_tokens']:
        raise ValueError(f"No truncation allowed: {row['id']} input={len(ids)} target={len(targets)}")
    return {'id':row['id'],'category':row['category'],'input_ids':ids,
            'labels':[-100]*len(prefix)+targets,'attention_mask':[1]*len(ids)}


# 模块：GPU训练前准备；作用：逐样本模板、泄漏、长度检查，冻结模型及实验身份；核心业务：否。
def prepare(dest):
    from transformers import AutoTokenizer
    pin=verify()
    if dest.exists(): raise FileExistsError('Run already exists; use a new run ID')
    config=read(ADDON/'config.json')
    from .train import tiny_backend_test
    print('STAGE=tiny_backend_forward_backward',flush=True)
    tiny=tiny_backend_test()
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3.5-9B',local_files_only=True,trust_remote_code=False)
    companies=read(ROOT/'configs/project.json')['companies']
    allowed={c['code'] for c in companies if c['split']=='train'}
    encoded, lengths, ids=[],[],set()
    with (ADDON/'data/train.jsonl').open(encoding='utf-8') as stream:
        for i,line in enumerate(stream,1):
            row=json.loads(line)
            if row['id'] in ids or row['company'] not in allowed or blocked(line,companies):
                raise ValueError('Duplicate or held-out company in training sample')
            ids.add(row['id'])
            sample=encode_sample(tokenizer,row,config)
            encoded.append(sample)
            lengths.append({'id':row['id'],'category':row['category'],'tokens':len(sample['input_ids']),
                            'supervised_tokens':sum(t!=-100 for t in sample['labels'])})
            if i%200==0: print(f'TOKEN_CHECK={i}',flush=True)
    if len(encoded)!=read(ADDON/'data/audit.json')['actions']: raise ValueError('Corpus count mismatch')
    hashes=model_fingerprint()
    dest.mkdir(parents=True,exist_ok=False)
    with (dest/'encoded.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for row in encoded: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    save(dest/'token_lengths.json',lengths)
    receipt={'status':'prepared','experiment_digest':experiment_identity(pin,hashes),'model_hashes':hashes,
             'package_sha256':sha(ADDON/'PACKAGE_MANIFEST.json'),'encoded_sha256':sha(dest/'encoded.jsonl'),
             'config':config,'examples':len(encoded),'max_tokens':max(r['tokens'] for r in lengths),
             'min_supervised_tokens':min(r['supervised_tokens'] for r in lengths),
             'categories':dict(Counter(r['category'] for r in encoded)),
             'loss_mask':'last action + EOS only; system/user/tools/history/empty-thinking all -100',
             'truncated_samples':0,'model_run':False}
    receipt['tiny_random_model_test']=tiny
    save(dest/'prepare.json',receipt)
    print('PREPARED='+json.dumps({k:v for k,v in receipt.items() if k!='model_hashes'},ensure_ascii=False),flush=True)


def verify_prepared(dest, rehash_model=True):
    pin=verify()
    receipt=read(dest/'prepare.json')
    if sha(dest/'encoded.jsonl')!=receipt['encoded_sha256']:
        raise ValueError('Prepared examples changed')
    hashes=model_fingerprint() if rehash_model else receipt['model_hashes']
    if experiment_identity(pin,hashes)!=receipt['experiment_digest']:
        raise ValueError('Dataset/config/model changed since prepare')
    return receipt
