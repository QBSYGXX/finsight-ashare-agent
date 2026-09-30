"""模块：推理适配；作用：复用已验证vLLM/MTP环境，显式记录新协议参数；核心业务：否。"""
import hashlib
import sys
import time

from .storage import ROOT


class Model:
    def __init__(self, backend="vllm-mtp", thinking=False, max_tokens=None):
        sys.path.insert(0,str(ROOT/"vllm_v013"))
        from adapter import Engine
        self.engine=Engine(backend)
        self.thinking=thinking
        self.budget=max_tokens or (8192 if thinking else 1024)
        self.last_generation=None

    def generate(self,messages):
        from vllm import SamplingParams
        engine=self.engine
        text=engine.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=self.thinking)
        ids=engine.tokenizer.encode(text,add_special_tokens=False)
        if len(ids)+self.budget>16384:
            raise ValueError("Context limit exceeded; evidence was not silently truncated")
        start=time.perf_counter()
        out=engine.llm.generate([{"prompt_token_ids":ids}],SamplingParams(temperature=0.6,top_p=0.95,top_k=20,
             seed=42,max_tokens=self.budget,skip_special_tokens=False),use_tqdm=False)[0].outputs[0]
        elapsed=time.perf_counter()-start
        self.last_generation={"backend":engine.backend,"thinking":self.thinking,"input_tokens":len(ids),
            "generated_tokens":len(out.token_ids),"seconds":elapsed,"stop_reason":out.finish_reason,
            "prompt_sha256":hashlib.sha256(text.encode()).hexdigest(),"max_tokens":self.budget,
            "tokens_per_second":len(out.token_ids)/elapsed,
            "raw_output":engine.tokenizer.decode(out.token_ids,skip_special_tokens=False).strip()}
        print("GENERATION_END="+str({k:v for k,v in self.last_generation.items() if k!="raw_output"}),flush=True)
        if out.finish_reason=="length":
            raise ValueError("generation_limit: model did not finish; incomplete output not executed")
        return engine.tokenizer.decode(out.token_ids,skip_special_tokens=False).strip()
