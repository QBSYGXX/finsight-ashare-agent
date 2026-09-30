"""Pinned single-GPU BF16 engines with identical prompt IDs and sampling."""
import hashlib
import json
import time
from pathlib import Path
from runtime import ROOT

SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0)


def timing_record(tokens, seconds, first=None, last=None):
    record = {"generated_tokens": tokens, "seconds": round(seconds, 6),
              "tokens_per_generation_second": tokens / seconds if seconds > 0 else None,
              "decode_tokens_per_second_estimate": None}
    if first is not None and last is not None and first > 0 and last > first and tokens > 1:
        record["decode_tokens_per_second_estimate"] = (tokens - 1) / (last - first)
    return record


# 模块：模型适配；作用：复用同一模型与原生思考模板，区别仅为推理引擎；核心业务：否。
class Engine:
    def __init__(self, backend):
        self.backend = backend
        self.last_generation = None
        started = time.perf_counter()
        print("MODEL_LOAD_START backend=" + backend, flush=True)
        if backend == "hf":
            from finsight_ashare.models import LocalModel
            self.hf = LocalModel(ROOT)
            self.tokenizer, self.torch = self.hf.tokenizer, self.hf.torch
            devices = {str(p.device) for p in self.hf.model.parameters()}
            if devices != {"cuda:0"}:
                raise RuntimeError("HF baseline must be fully resident on one visible GPU")
        else:
            from transformers import AutoTokenizer
            from vllm import LLM
            model = ROOT / "models/Qwen3.5-9B"
            self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
            options = dict(model=str(model), tokenizer=str(model), dtype="bfloat16", tensor_parallel_size=1,
                max_model_len=16384, max_num_seqs=1, gpu_memory_utilization=0.65, seed=42,
                language_model_only=True, trust_remote_code=False, enable_prefix_caching=False,
                disable_log_stats=False)
            if backend == "vllm-mtp":
                options["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 1}
            self.llm = LLM(**options)
        self.load_seconds = time.perf_counter() - started
        print(f"MODEL_LOAD_END backend={backend} seconds={self.load_seconds:.3f}", flush=True)

    def encode(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if not prompt.endswith("<|im_start|>assistant\n<think>\n"):
            raise RuntimeError("Native thinking template mismatch")
        return prompt, self.tokenizer.encode(prompt, add_special_tokens=False)

    def generate(self, messages):
        return self.run(messages, 8192, fixed=False)[0]

    def run(self, messages, budget, fixed=False):
        prompt, input_ids = self.encode(messages)
        if len(input_ids) + budget > 16384:
            raise ValueError("Context limit exceeded; input was not truncated")
        eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos is None or eos == self.tokenizer.unk_token_id:
            raise RuntimeError("Missing EOS token")
        first, last, ttft = None, None, None
        started = time.perf_counter()
        if self.backend == "hf":
            from transformers.generation.streamers import BaseStreamer
            class Timer(BaseStreamer):
                prompt_seen = False
                first = None
                last = None
                def put(self, value):
                    if not self.prompt_seen:
                        self.prompt_seen = True
                    else:
                        now = time.perf_counter()
                        if self.first is None:
                            self.first = now
                        self.last = now
                def end(self):
                    pass
            timer = Timer()
            torch = self.torch
            batch = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
            torch.manual_seed(42)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = self.hf.model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
                    do_sample=True, **SAMPLING, max_new_tokens=budget, min_new_tokens=budget if fixed else 0,
                    eos_token_id=eos, pad_token_id=self.tokenizer.pad_token_id or eos, streamer=timer, use_cache=True)
            torch.cuda.synchronize()
            tokens = output[0, len(input_ids):].tolist()
            first, last = timer.first, timer.last
            ttft = first - started if first is not None else None
            stop = "eos" if tokens and tokens[-1] == eos and not fixed else "max_new_tokens" if len(tokens) == budget else "other"
        else:
            from vllm import SamplingParams
            params = SamplingParams(**SAMPLING, seed=42, max_tokens=budget, min_tokens=budget if fixed else 0,
                ignore_eos=fixed, stop_token_ids=None if fixed else [eos], skip_special_tokens=False)
            output = self.llm.generate([{"prompt_token_ids": input_ids}], params, use_tqdm=False)[0]
            completion = output.outputs[0]
            tokens = list(completion.token_ids)
            stop = "max_new_tokens" if completion.finish_reason == "length" else "eos" if completion.finish_reason == "stop" else "other"
            metrics = getattr(output, "metrics", None)
            if metrics is not None:
                first = getattr(metrics, "first_token_ts", None)
                last = getattr(metrics, "last_token_ts", None)
                ttft = getattr(metrics, "first_token_latency", None)
        elapsed = time.perf_counter() - started
        if fixed and len(tokens) != budget:
            raise RuntimeError(f"Fixed-token test produced {len(tokens)} instead of {budget}")
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        self.last_generation = {"backend": self.backend, "enable_thinking": True, "seed": 42,
            "sampling": SAMPLING, "input_tokens": len(input_ids), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_ids_sha256": hashlib.sha256(json.dumps(input_ids).encode()).hexdigest(),
            "output_ids_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "max_new_tokens": budget, "fixed_length": fixed, "stop_reason": stop, "ttft_seconds": ttft,
            "timing_note": "Generation wall time includes prefill. Decode estimate uses engine token timestamps; MTP may emit bursts.",
            **timing_record(len(tokens), elapsed, first, last)}
        print("GENERATION=" + json.dumps(self.last_generation), flush=True)
        return text, dict(self.last_generation)
