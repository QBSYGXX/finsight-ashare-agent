"""Planner-only native thinking controls; frozen agent loading is reused."""
import hashlib
import json
import threading
import time

from engine import MetricError
from finsight_ashare.models import LocalModel


class PlannerModel(LocalModel):
    last_generation = None
    enable_thinking = False
    decoding = "auto"
    seed = 42

    def __init__(self, root, adapter=None, *, thinking=False, max_new_tokens=None, context_length=None,
                 decoding="auto", seed=42):
        super().__init__(root, adapter)
        self.enable_thinking = thinking
        self.decoding, self.seed = decoding, seed
        self.options = dict(self.options)
        self.options["max_new_tokens"] = max_new_tokens if max_new_tokens is not None else (2048 if thinking else 512)
        self.options["context_length"] = context_length if context_length is not None else (8192 if thinking else 4096)
        text_config = getattr(self.model.config, "text_config", self.model.config)
        limit = getattr(text_config, "max_position_embeddings", None)
        if limit is not None and self.options["context_length"] > limit:
            raise MetricError("context_limit", "请求的上下文长度超过模型配置上限")

    def generate(self, messages):
        self.last_generation = {"enable_thinking": self.enable_thinking, "template": "tokenizer.apply_chat_template"}
        sample = self.decoding == "sample" or (self.decoding == "auto" and self.enable_thinking)
        generation_options = {"do_sample": sample}
        if sample:
            # Use the model card's precise-coding profile for structured planning.
            generation_options.update(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0)
        self.last_generation.update(decoding="sample" if sample else "greedy",
                                    sampling=generation_options, seed=self.seed if sample else None)
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        except (ValueError, TypeError) as error:
            raise MetricError("template_error", "无法使用模型原生聊天模板，未生成计划",
                              {"error": str(error), "generation": self.last_generation}) from error
        # Verify the native template actually honored the flag before spending GPU time.
        suffix = "<|im_start|>assistant\n<think>\n" + ("" if self.enable_thinking else "\n</think>\n\n")
        if not prompt.endswith(suffix):
            raise MetricError("template_error", "模型模板未应用所选思考模式，未生成计划",
                              {"generation": self.last_generation})
        batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        length = batch["input_ids"].shape[1]
        budget = min(self.options["max_new_tokens"], self.options["context_length"] - length)
        self.last_generation.update(input_tokens=length, max_new_tokens=budget,
                                    requested_max_new_tokens=self.options["max_new_tokens"],
                                    context_length=self.options["context_length"],
                                    context_limited=budget < self.options["max_new_tokens"],
                                    prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        if budget < 64:
            raise MetricError("context_limit", "查询计划上下文空间不足，未截断输入",
                              {"generation": self.last_generation})
        eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos is None or eos == self.tokenizer.unk_token_id:
            raise MetricError("template_error", "模型缺少有效的结束标记，未生成计划")
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        batch = batch.to(self.model.device)
        started = time.perf_counter()
        finished = threading.Event()

        def progress():
            while not finished.wait(30):
                print(f"PLANNER_GENERATING elapsed_seconds={time.perf_counter() - started:.0f}", flush=True)

        monitor = threading.Thread(target=progress, daemon=True)
        print("PLANNER_GENERATION_START=" + json.dumps(self.last_generation), flush=True)
        monitor.start()
        try:
            if sample:
                self.torch.manual_seed(self.seed)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **batch, **generation_options, max_new_tokens=budget,
                    eos_token_id=eos, pad_token_id=pad, return_dict_in_generate=False)
            tokens = generated[0, length:]
            ids = tokens.tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            reason = "eos" if ids and ids[-1] == eos else "max_new_tokens" if len(ids) >= budget else "other"
            self.last_generation.update(generated_tokens=len(ids), stop_reason=reason,
                                        seconds=round(time.perf_counter() - started, 3))
            print("PLANNER_GENERATION_END=" + json.dumps(self.last_generation), flush=True)
            return text
        finally:
            finished.set()
            monitor.join()
