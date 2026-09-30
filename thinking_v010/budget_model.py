"""Bounded thinking using the original single-GPU PyTorch model and planner."""
import hashlib
import json
import threading
import time
from budget import CLOSE, reserve_context, run_stages
from engine import MetricError
from planner_model import PlannerModel


class BudgetPlanner(PlannerModel):
    def __init__(self, root, *, think_budget=1024, answer_budget=512, context_length=16384, seed=42):
        super().__init__(root, thinking=True, max_new_tokens=8192,
                         context_length=context_length, decoding="sample", seed=seed)
        self.think_budget, self.answer_budget = think_budget, answer_budget
        self.policy = "budgeted"
        devices = {str(p.device) for p in self.model.parameters()}
        if devices != {"cuda:0"}:
            raise RuntimeError(f"Expected one visible GPU with entire model resident; got {devices}")

    def generate(self, messages):
        if self.policy == "baseline":
            return super().generate(messages)
        from transformers import StoppingCriteria, StoppingCriteriaList
        tokenizer, torch = self.tokenizer, self.torch
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        if not prompt.endswith("<|im_start|>assistant\n<think>\n"):
            raise MetricError("template_error", "原生思考模板不匹配，未生成")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        bridge_ids = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
        eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos is None or eos == tokenizer.unk_token_id:
            raise MetricError("template_error", "结束标记无效，未生成")
        try:
            reserve_context(len(prompt_ids), self.options["context_length"],
                            self.think_budget, self.answer_budget, len(bridge_ids))
        except ValueError as error:
            raise MetricError("context_limit", str(error)) from error
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
        sampling = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                        min_p=0.0, repetition_penalty=1.0)
        self.last_generation = dict(enable_thinking=True, policy="budgeted", seed=self.seed,
            sampling=sampling, input_tokens=len(prompt_ids), think_budget=self.think_budget,
            answer_budget=self.answer_budget, context_length=self.options["context_length"],
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), phases=[],
            continuation="same assistant turn; second prefill from original token IDs; no KV reuse")
        finished = threading.Event()
        started = time.perf_counter()

        def heartbeat():
            while not finished.wait(30):
                print(f"BUDGET_GENERATING elapsed_seconds={time.perf_counter()-started:.0f}", flush=True)

        class EndThinking(StoppingCriteria):
            def __init__(self, start):
                self.start = start

            def __call__(self, input_ids, scores, **kwargs):
                # The closing marker is ordinary multi-token text, not an EOS ID.
                tail = input_ids[0, max(self.start, input_ids.shape[1]-32):].tolist()
                return CLOSE in tokenizer.decode(tail, skip_special_tokens=False)

        def generate(ids, budget, stop_thinking, phase):
            print(f"BUDGET_PHASE={phase} max_new_tokens={budget}", flush=True)
            batch = torch.tensor([ids], dtype=torch.long, device=self.model.device)
            kwargs = {"stopping_criteria": StoppingCriteriaList([EndThinking(len(ids))])} if stop_thinking else {}
            torch.cuda.synchronize()
            began = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
                    **sampling, **kwargs, max_new_tokens=budget, eos_token_id=eos,
                    pad_token_id=pad, use_cache=True, logits_to_keep=1, return_dict_in_generate=False)
            torch.cuda.synchronize()
            new_ids = output[0, len(ids):].tolist()
            self.last_generation["phases"].append(dict(phase=phase, generated_tokens=len(new_ids),
                seconds=round(time.perf_counter()-began, 3)))
            return new_ids

        print("BUDGET_START=" + json.dumps(self.last_generation), flush=True)
        monitor = threading.Thread(target=heartbeat, daemon=True)
        monitor.start()
        try:
            torch.manual_seed(self.seed)
            ids, metadata = run_stages(prompt_ids, generate,
                lambda x: tokenizer.decode(x, skip_special_tokens=True), bridge_ids,
                eos, self.think_budget, self.answer_budget)
            self.last_generation.update(metadata, generated_tokens=metadata["thinking_tokens"]+metadata["answer_tokens"],
                seconds=round(time.perf_counter()-started, 3))
            print("BUDGET_END=" + json.dumps(self.last_generation), flush=True)
            return tokenizer.decode(ids, skip_special_tokens=True).strip()
        finally:
            finished.set()
            monitor.join()
