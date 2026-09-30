"""模块：独立权重合并；作用：仅叠加语言LoRA，逐张量保留原始视觉/MTP权重；核心业务：否。"""
import json
import shutil

from .common import ROOT, read, save, sha
from .tokenization import verify_prepared


def adapter_pairs(keys, base_keys):
    pairs={}
    for key in keys:
        if key.endswith('.lora_A.weight'):
            module=key.removeprefix('base_model.model.').removesuffix('.lora_A.weight')
            base=module+'.weight'
            if not module.startswith('model.language_model.layers.') or base not in base_keys:
                raise ValueError('Adapter target outside original language weights: '+key)
            b=key.removesuffix('.lora_A.weight')+'.lora_B.weight'
            if b not in keys: raise ValueError('Missing LoRA B')
            pairs[base]=(key,b)
    if not pairs or {k for pair in pairs.values() for k in pair}!=set(keys):
        raise ValueError('Unexpected adapter keys; no silent tensor dropping')
    return pairs


def merge(dest):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    receipt=verify_prepared(dest)
    complete=read(dest/'full/completion.json')
    if complete.get('status')!='completed' or complete.get('probe') or not complete.get('adapter_reload'):
        raise ValueError('Full training and reload required')
    if complete['experiment_digest']!=receipt['experiment_digest']: raise ValueError('Training identity changed')
    adapter=dest/'full/adapter'
    for n,h in complete['adapter_hashes'].items():
        if sha(adapter/n)!=h: raise ValueError('Adapter changed after training: '+n)
    config=read(adapter/'adapter_config.json')
    if config.get('use_dora') or config.get('use_rslora') or config.get('fan_in_fan_out') or config.get('modules_to_save') or config.get('rank_pattern') or config.get('alpha_pattern') or config.get('bias','none')!='none':
        raise ValueError('Only standard constant-rank alpha/r LoRA is supported')
    base=ROOT/'models/Qwen3.5-9B'
    index=read(base/'model.safetensors.index.json')
    weights=load_file(adapter/'adapter_model.safetensors',device='cpu')
    pairs=adapter_pairs(weights,index['weight_map'])
    out=dest/'merged'
    if out.exists(): raise FileExistsError('Merged snapshot already exists; never overwrite')
    size=sum((base/n).stat().st_size for n in set(index['weight_map'].values()))
    if shutil.disk_usage(dest).free<size+6*1024**3: raise OSError('Insufficient free space for isolated merged snapshot')
    out.mkdir()
    hashes={}; changed=[]; preserved_mtp=0; changed_values=0
    scale=config['lora_alpha']/config['r']
    for shard in sorted(set(index['weight_map'].values())):
        print('MERGE_SHARD='+shard,flush=True)
        with safe_open(base/shard,framework='pt',device='cpu') as source:
            metadata=source.metadata()
        tensors=load_file(base/shard,device='cpu')
        original_keys=set(tensors)
        for name in sorted(set(tensors)&set(pairs)):
            a,b=(weights[k].float() for k in pairs[name])
            original=tensors[name]
            if a.shape[0]!=config['r'] or b.shape[1]!=config['r'] or (b.shape[0],a.shape[1])!=original.shape:
                raise ValueError('LoRA dimensions do not match '+name)
            delta=(b@a)*scale
            updated=(original.float()+delta).to(original.dtype)
            if not torch.isfinite(updated).all():raise ValueError('Non-finite merged tensor')
            if not torch.equal(updated,original):changed_values+=1
            tensors[name]=updated.contiguous();changed.append(name)
        target=out/shard
        save_file(tensors,target,metadata=metadata)
        # Full equality check for every serialized tensor; untouched MTP is explicitly counted.
        with safe_open(target,framework='pt',device='cpu') as saved:
            if set(saved.keys())!=original_keys:raise ValueError('Tensor inventory changed')
            for name in tensors:
                if not torch.equal(saved.get_tensor(name),tensors[name]):raise ValueError('Saved tensor mismatch')
                if name.startswith('mtp.'):preserved_mtp+=1
        hashes[shard]=sha(target)
        del tensors
    if set(changed)!=set(pairs) or not changed_values or not preserved_mtp:
        raise ValueError('Merge incomplete, numerically unchanged, or MTP lost')
    # Preserve all tokenizer/config metadata; exclude original weights already written above.
    for p in sorted(base.iterdir()):
        if p.is_file() and (p.suffix in {'.json','.jinja','.txt','.model'}):
            shutil.copyfile(p,out/p.name);hashes[p.name]=sha(out/p.name)
    save(dest/'merge.json',{'status':'completed','experiment_digest':receipt['experiment_digest'],
         'adapter_hashes':complete['adapter_hashes'],'files':hashes,'updated_tensors':len(changed),
         'numerically_changed_tensors':changed_values,'preserved_mtp_tensors':preserved_mtp,
         'note':'Original shard inventory retained; visual/MTP weights not trained or dropped; base snapshot unchanged'})
    print(f'MERGE=completed updated={len(changed)} mtp_preserved={preserved_mtp}',flush=True)
