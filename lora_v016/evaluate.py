"""模块：公平对照；作用：相同Agent/检索/采样下比较基础与新LoRA合并模型；核心业务：否。"""
import hashlib
import json
import time

from .common import ROOT, read, save, sha
from .tokenization import verify_prepared


class PairedModel:
    def __init__(self, model_path):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3.5-9B',local_files_only=True)
        self.llm=LLM(model=str(model_path),tokenizer=str(ROOT/'models/Qwen3.5-9B'),dtype='bfloat16',
             tensor_parallel_size=1,max_model_len=16384,max_num_seqs=1,gpu_memory_utilization=0.65,
             seed=42,language_model_only=True,trust_remote_code=False,enable_prefix_caching=False,
             disable_log_stats=False,speculative_config={'method':'mtp','num_speculative_tokens':1})
        self.last_generation=None

    def generate(self,messages):
        from vllm import SamplingParams
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=self.tokenizer.encode(prompt,add_special_tokens=False)
        if len(ids)+1024>16384: raise ValueError('Context limit exceeded; no truncation')
        started=time.perf_counter()
        out=self.llm.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0.6,top_p=0.95,
             top_k=20,seed=42,max_tokens=1024,skip_special_tokens=False),use_tqdm=False)[0].outputs[0]
        seconds=time.perf_counter()-started
        text=self.tokenizer.decode(out.token_ids,skip_special_tokens=False).strip()
        self.last_generation={'backend':'vllm-mtp','thinking':False,'input_tokens':len(ids),
             'generated_tokens':len(out.token_ids),'seconds':seconds,'stop_reason':out.finish_reason,
             'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'max_tokens':1024,
             'tokens_per_second':len(out.token_ids)/seconds,'raw_output':text}
        print('GENERATION_END='+json.dumps({k:v for k,v in self.last_generation.items() if k!='raw_output'}),flush=True)
        if out.finish_reason=='length':raise ValueError('generation_limit: action not executed')
        return text


def evaluate(dest,arm):
    from agent_v015.retrieval import Index
    from agent_v015.evaluation import run as diagnostic
    from agent_v015.natural_evaluation import run as natural
    receipt=verify_prepared(dest)
    if arm not in {'base','tuned'}:raise ValueError('Unknown comparison arm')
    completion=read(dest/'full/completion.json')
    if completion['status']!='completed':raise ValueError('Training not completed')
    model_path=ROOT/'models/Qwen3.5-9B'
    if arm=='tuned':
        merged=read(dest/'merge.json')
        if merged['status']!='completed' or merged['experiment_digest']!=receipt['experiment_digest']:
            raise ValueError('Merged snapshot identity mismatch')
        model_path=dest/'merged'
        for n,h in merged['files'].items():
            if sha(model_path/n)!=h:raise ValueError('Merged file changed: '+n)
    output=dest/'evaluation'/arm
    if output.exists():raise FileExistsError('Comparison result already exists; do not overwrite')
    index=Index(ROOT,ROOT/'outputs/agent_v015/index/validation-hybrid')
    save(output/'run_config.json',{'arm':arm,'experiment_digest':receipt['experiment_digest'],
        'model_path':str(model_path),'backend':'vllm-mtp','thinking':False,'seed':42,'max_tokens':1024,
        'index':index.meta,'scorers_unchanged':True,'independent_test_used':False,
        'adapter':None if arm=='base' else completion['adapter_hashes']})
    # Lazily load exactly once, so first-question cold start remains visible.
    class Lazy:
        instance=None
        last_generation=None
        def generate(self,messages):
            if self.instance is None:self.instance=PairedModel(model_path)
            try:return self.instance.generate(messages)
            finally:self.last_generation=self.instance.last_generation
    model=Lazy()
    summaries={}
    for name,runner in [('diagnostic',diagnostic),('natural',natural)]:
        summaries[name]=runner(ROOT,index,model,output/name)
        if summaries[name]['execution_status']!='completed' or summaries[name]['not_run']:
            raise RuntimeError('Evaluation incomplete; do not publish as full comparison')
    save(output/'summary.json',summaries)


def compare(dest):
    arms={arm:read(dest/'evaluation'/arm/'summary.json') for arm in ['base','tuned']}
    configs={arm:read(dest/'evaluation'/arm/'run_config.json') for arm in arms}
    for key in ['experiment_digest','backend','thinking','seed','max_tokens','index']:
        if configs['base'][key]!=configs['tuned'][key]:raise ValueError('Unfair comparison: '+key)
    groups={}
    for group in ['diagnostic','natural']:
        old={r['id']:r for r in arms['base'][group]['records']}
        new={r['id']:r for r in arms['tuned'][group]['records']}
        if set(old)!=set(new):raise ValueError('Comparison cases differ')
        groups[group]={'base':arms['base'][group]['diagnostic_pass'],'tuned':arms['tuned'][group]['diagnostic_pass'],
            'gained':[k for k in old if not old[k]['verdict']['passed'] and new[k]['verdict']['passed']],
            'regressed':[k for k in old if old[k]['verdict']['passed'] and not new[k]['verdict']['passed']],
            'both_failed':[k for k in old if not old[k]['verdict']['passed'] and not new[k]['verdict']['passed']]}
    for arm in arms:
        groups['natural'][arm+'_model']=arms[arm]['natural']['model_invoking_pass']
        groups['natural'][arm+'_preflight']=arms[arm]['natural']['preflight_pass']
    result={'status':'completed','protocol':'lora-v016-comparison','groups':groups,
        'limitations':'Development diagnostics, one training seed, no held-out company test. MTP uses unchanged draft weights with target verification; do not claim blind/general accuracy.'}
    save(dest/'comparison.json',result)
    print('COMPARISON='+json.dumps(result,ensure_ascii=False),flush=True)
