"""模块：新协议SFT；作用：BF16 LoRA、20步探针、保存重载和完成凭证；核心业务：是。"""
import gc
import inspect
import json
import math
import os
from pathlib import Path

from .common import ROOT, read, save, sha, digest
from .tokenization import verify_prepared


def training_arguments(arguments_class, config, output, world, probe):
    """Validate the entire pinned Trainer interface before loading model weights."""
    if world not in (1,2,4) or config['effective_batch_size'] % world:
        raise ValueError('Use 1, 2 or 4 GPUs with an integral accumulation count')
    options=dict(output_dir=str(output),
        per_device_train_batch_size=1,gradient_accumulation_steps=config['effective_batch_size']//world,
        num_train_epochs=config['epochs'],max_steps=config['probe_steps'] if probe else -1,
        learning_rate=config['learning_rate'],lr_scheduler_type='cosine',warmup_steps=0.05,
        bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        logging_steps=1,logging_nan_inf_filter=False,save_strategy='no',report_to='none',
        ddp_find_unused_parameters=False,remove_unused_columns=False,seed=config['seed'],data_seed=config['seed'],
        dataloader_num_workers=0,dataloader_pin_memory=True,optim='adamw_torch')
    # Do not silently filter unsupported options: they can alter the experiment.
    inspect.signature(arguments_class).bind(**options)
    args=arguments_class(**options)
    if args.get_warmup_steps(20)!=1 or args.get_warmup_steps(143)!=8:
        raise ValueError('TrainingArguments must interpret warmup_steps=0.05 as a 5 percent ratio')
    print('TRAINING_ARGUMENTS=PASS; warmup=5%; output overwrite guarded by filesystem check',flush=True)
    return args


def require_fresh_output(output):
    if output.exists():
        raise FileExistsError('Training output exists; never overwrite or auto-resume')


def language_targets(module_names):
    return sorted(n for n in module_names if n.startswith('model.language_model.layers.')
                  and not any(part in n.split('.') for part in ['mtp','visual','vision']))


def probe_rows(rows):
    chosen={}
    for row in rows:
        key=row['category']
        if key not in chosen or len(row['input_ids'])>len(chosen[key]['input_ids']): chosen[key]=row
    result=list(chosen.values())
    selected={r['id'] for r in result}
    for row in sorted(rows,key=lambda r:r['id']):
        if row['id'] not in selected:
            result.append(row);selected.add(row['id'])
        if len(result)>=32:break
    return result


# 模块：动作损失；作用：仅计算最后一个动作的logits，避免为长历史生成巨大词表张量；核心业务：否。
def action_loss(model, inputs):
    import torch
    labels=inputs['labels']
    if labels.shape[0]!=1: raise ValueError('This tail-loss implementation requires per-device batch size 1')
    n=int((labels[0]!=-100).sum())
    if n<1 or (labels[0,-n:]==-100).any() or (labels[0,:-n]!=-100).any():
        raise ValueError('Labels must be a contiguous supervised suffix')
    result=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],
                 use_cache=False,logits_to_keep=n+1)
    logits=result.logits
    if logits.shape[1]!=n+1: raise ValueError('Model did not honor logits_to_keep; do not risk full-vocabulary history allocation')
    return torch.nn.functional.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),labels[:,-n:].reshape(-1))


def tiny_backend_test():
    """先用随机小模型比较完整损失和尾部损失，验证当前Transformers/PEFT接口。"""
    import torch
    from transformers import Qwen3_5Config,Qwen3_5TextConfig,Qwen3_5ForConditionalGeneration,TrainingArguments
    from .common import ADDON
    training_arguments(TrainingArguments,read(ADDON/'config.json'),ROOT/'.cache/lora-v0161-api-check',1,True)
    from peft import LoraConfig,get_peft_model
    text=Qwen3_5TextConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,full_attention_interval=2,
        linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=16,linear_value_head_dim=16,
        linear_conv_kernel_dim=4,mtp_num_hidden_layers=0,
        rope_parameters={'rope_type':'default','rope_theta':10000.0,'partial_rotary_factor':0.25,
                         'mrope_section':[1,1,0],'mrope_interleaved':True})
    config=Qwen3_5Config(text_config=text.to_dict(),vision_config={'depth':1,'hidden_size':32,
        'intermediate_size':64,'num_heads':4,'out_hidden_size':64,'patch_size':2,
        'spatial_merge_size':1,'temporal_patch_size':1,'num_position_embeddings':16})
    config._attn_implementation='sdpa'
    model=get_peft_model(Qwen3_5ForConditionalGeneration(config),LoraConfig(task_type='CAUSAL_LM',r=2,
        lora_alpha=4,lora_dropout=0,target_modules=['q_proj','v_proj'])).eval()
    ids=torch.tensor([[1,3,5,7,9,11,13,15]])
    labels=torch.tensor([[-100,-100,-100,-100,-100,11,13,15]])
    inputs={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels}
    tail=action_loss(model,inputs)
    full=model(**inputs,use_cache=False).loss
    if not torch.allclose(tail,full,atol=1e-5,rtol=1e-5):raise ValueError('Tail loss differs from masked full loss')
    tail.backward()
    gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Tiny LoRA backward failed')
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=0.01).step()
    for module in model.modules():
        if hasattr(module,'lora_A') and 'default' in module.lora_A:
            manual=(module.lora_B['default'].weight@module.lora_A['default'].weight)*2
            if not torch.allclose(manual,module.get_delta_weight('default'),atol=1e-7,rtol=1e-6):
                raise ValueError('Manual merge formula differs from PEFT')
    return {'tail_loss':float(tail.detach()),'full_masked_loss':float(full.detach()),
            'backward':'passed','merge_formula_vs_peft':'passed'}


def train(dest, probe=False):
    import torch
    import torch.distributed as dist
    import importlib.metadata
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model
    rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA BF16 required')
    receipt=verify_prepared(dest,rehash_model=rank==0)
    config=receipt['config']
    if world not in [1,2,4] or config['effective_batch_size']%world: raise ValueError('Use 1, 2 or 4 GPUs')
    run_identity=digest({'experiment':receipt['experiment_digest'],'world_size':world})
    if not probe:
        prior=read(dest/'probe/completion.json')
        if prior.get('status')!='completed' or prior.get('training_identity')!=run_identity or not prior.get('adapter_reload'):
            raise ValueError('Same experiment/GPU count must pass 20-step probe and adapter reload first')
    output=dest/('probe' if probe else 'full')
    # All ranks check before Trainer initializes DDP; only rank0 creates this output below.
    require_fresh_output(output)
    if world>1:
        torch.cuda.set_device(local)
        if not dist.is_initialized():dist.init_process_group(backend='nccl')
        dist.barrier() # every rank checked absence before any Trainer can create the directory
    rows=[json.loads(line) for line in (dest/'encoded.jsonl').open(encoding='utf-8')]
    if probe: rows=probe_rows(rows)
    set_seed(config['seed'])
    torch.cuda.set_device(local)
    arguments=training_arguments(TrainingArguments,config,output,world,probe)
    print(f'MODEL_LOAD rank={rank} world={world} probe={probe}',flush=True)
    model=AutoModelForImageTextToText.from_pretrained(ROOT/'models/Qwen3.5-9B',local_files_only=True,
          dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':local})
    targets=language_targets([n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)])
    if not targets: raise ValueError('No language linear layers found')
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=config['rank'],lora_alpha=config['alpha'],
          lora_dropout=config['dropout'],target_modules=targets,bias='none'))
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n or '.language_model.layers.' not in n for n in trainable):
        raise ValueError('Unexpected non-language/non-LoRA trainable parameters')
    model.config.use_cache=False
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    class ActionTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            if return_outputs: raise ValueError('Use the separate Agent evaluator, not Trainer prediction')
            return action_loss(model,inputs)

    class FiniteLog(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            for key in ['loss','grad_norm']:
                if key in (logs or {}) and not math.isfinite(float(logs[key])):
                    raise FloatingPointError('Non-finite '+key)
            if state.is_world_process_zero:
                print('TRAIN_PROGRESS='+json.dumps({'step':state.global_step,**(logs or {})}),flush=True)

    def collate(batch):
        if len(batch)!=1:raise ValueError('Expected microbatch=1')
        return {k:torch.tensor([batch[0][k]],dtype=torch.long) for k in ['input_ids','attention_mask','labels']}

    trainer=ActionTrainer(model=model,args=arguments,train_dataset=rows,data_collator=collate,callbacks=[FiniteLog()])
    trainer.model_accepts_loss_kwargs=False
    if rank==0:
        save(output/'provenance.json',{'status':'running','training_identity':run_identity,'targets':targets,
             'world_size':world,'microbatch':1,'gradient_accumulation':config['effective_batch_size']//world,
             'experiment_digest':receipt['experiment_digest'],'examples':len(rows),'config':config,
             'versions':{m:importlib.metadata.version(m) for m in ['torch','transformers','peft','accelerate']}})
    trainer.train()
    state=trainer.state
    if state.global_step!=state.max_steps or state.global_step<1 or (probe and state.global_step!=config['probe_steps']):
        raise ValueError('Optimizer steps did not complete')
    if not probe and state.epoch<config['epochs']-0.01: raise ValueError('Full epochs did not complete')
    losses=[float(r['loss']) for r in state.log_history if 'loss' in r]
    if not losses or not all(math.isfinite(x) for x in losses): raise ValueError('Loss missing/non-finite')
    trainer.save_model(str(output/'adapter'))
    trainer.save_state()
    if dist.is_initialized(): dist.barrier()
    if rank==0:
        save(output/'trained.json',{'global_step':state.global_step,'epoch':state.epoch,'losses':losses,
                                   'training_identity':run_identity})
    # GPU memory release before saved-adapter reload in a fresh subprocess.
    del trainer,model
    gc.collect(); torch.cuda.empty_cache()
    if dist.is_initialized(): dist.destroy_process_group()
    print(f'TRAIN_WORKER_DONE rank={rank}; reload gate still pending',flush=True)


# 模块：保存后重载；作用：新进程确认适配器可加载、动作损失有限且生成非空；核心业务：否。
def reload_check(dest,probe=False):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    receipt=verify_prepared(dest)
    output=dest/('probe' if probe else 'full')
    trained=read(output/'trained.json')
    weights=load_file(output/'adapter/adapter_model.safetensors')
    if not weights or not all(torch.isfinite(w).all() for w in weights.values()): raise ValueError('Invalid adapter tensors')
    if not any('lora_B' in k and torch.count_nonzero(v)>0 for k,v in weights.items()): raise ValueError('Adapter did not update')
    base=AutoModelForImageTextToText.from_pretrained(ROOT/'models/Qwen3.5-9B',local_files_only=True,
         dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0})
    model=PeftModel.from_pretrained(base,output/'adapter',local_files_only=True).eval()
    loaded=get_peft_model_state_dict(model)
    if set(loaded)!=set(weights) or any(not torch.equal(loaded[k].detach().cpu().to(v.dtype),v) for k,v in weights.items()):
        raise ValueError('Reload did not restore every saved adapter tensor exactly')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3.5-9B',local_files_only=True)
    with (dest/'encoded.jsonl').open(encoding='utf-8') as stream: sample=json.loads(next(stream))
    inputs={k:torch.tensor([sample[k]],device='cuda:0') for k in ['input_ids','labels','attention_mask']}
    with torch.inference_mode():
        loss=action_loss(model,inputs)
        if not torch.isfinite(loss): raise ValueError('Reload loss non-finite')
        prefix=next(i for i,v in enumerate(sample['labels']) if v!=-100)
        ids=inputs['input_ids'][:,:prefix]
        generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
             max_new_tokens=128,use_cache=True,eos_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'),
             pad_token_id=tokenizer.pad_token_id)
    text=tokenizer.decode(generated[0,prefix:],skip_special_tokens=False)
    if not text.strip(): raise ValueError('Empty reload output')
    hashes={p.name:sha(p) for p in (output/'adapter').iterdir() if p.is_file()}
    save(output/'reload.json',{'loss':float(loss),'output':text,'quality_claim':False})
    save(output/'completion.json',{'status':'completed','probe':probe,'adapter_reload':True,
         'training_identity':trained['training_identity'],'experiment_digest':receipt['experiment_digest'],
         'global_step':trained['global_step'],'epoch':trained['epoch'],'adapter_hashes':hashes,
         'quality_evaluation_pending':True})
    print('ADAPTER_RELOAD=PASS; this is serialization compatibility, not task accuracy',flush=True)
