"""Lazy local inference. All GPU work happens on the server."""
from .common import ROOT, config


# 模块：模型接口；作用：加载固定本地快照及可选LoRA；核心业务：否。
class LocalModel:
    def __init__(self, root=ROOT, adapter=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.options = config(root)
        self.torch = torch
        model_path = root / "models/Qwen3.5-9B"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for Qwen3.5-9B inference")
        print("Loading Qwen3.5-9B from local snapshot...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(str(model_path), dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa", local_files_only=True)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter), local_files_only=True)
        self.model.eval()

    def generate(self, messages):
        # Use the exact non-thinking ChatML format registered by the pinned LlamaFactory.
        prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        prompt += "<|im_start|>assistant\n"
        batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        length = batch["input_ids"].shape[1]
        budget = self.options["context_length"] - length
        if budget < 64:
            raise ValueError("Context limit reached; shorten retrieved evidence instead of silently truncating it")
        batch = batch.to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**batch, do_sample=False,
                max_new_tokens=min(self.options["max_new_tokens"], budget),
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return self.tokenizer.decode(generated[0, length:], skip_special_tokens=True).strip()
