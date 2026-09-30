"""Same planner/generation logic, single-GPU placement and verified native kernels."""
import time
from planner_model import PlannerModel


class CachedGeneration:
    def __init__(self, model):
        self.inner = model

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def generate(self, **kwargs):
        kwargs.setdefault("use_cache", True)
        kwargs.setdefault("logits_to_keep", 1)
        return self.inner.generate(**kwargs)


class FastPlannerModel(PlannerModel):
    def __init__(self, root, adapter=None, *, gpu=0, thinking=True, max_new_tokens=8192,
                 context_length=16384, decoding="sample", seed=42):
        import torch
        torch.cuda.set_device(gpu)
        from kernels import enable
        self.kernel_backend = enable()
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from transformers.models.qwen3_5 import modeling_qwen3_5 as module
        from finsight_ashare.common import config
        if not module.is_fast_path_available:
            raise RuntimeError("Fast kernels not active; use install/probe first")
        if not 0 <= gpu < torch.cuda.device_count():
            raise ValueError("Selected GPU is outside CUDA_VISIBLE_DEVICES")
        free, _ = torch.cuda.mem_get_info(gpu)
        if free < 28 * 1024**3:
            raise RuntimeError("Selected GPU has less than 28 GiB free; choose --gpu with enough free memory")
        torch.cuda.set_device(gpu)
        self.torch = torch
        self.options = dict(config(root))
        self.options.update(max_new_tokens=max_new_tokens, context_length=context_length)
        self.enable_thinking, self.decoding, self.seed = thinking, decoding, seed
        path = root / "models/Qwen3.5-9B"
        print(f"FAST_MODEL_LOAD=gpu:{gpu} BF16 SDPA FLA Triton-convolution", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        raw = AutoModelForImageTextToText.from_pretrained(str(path), dtype=torch.bfloat16,
            device_map={"": f"cuda:{gpu}"}, attn_implementation="sdpa", local_files_only=True)
        if adapter is not None:
            from peft import PeftModel
            raw = PeftModel.from_pretrained(raw, str(adapter), local_files_only=True)
        raw.eval()
        raw.config.use_cache = True
        raw.generation_config.use_cache = True
        self.model = CachedGeneration(raw)

    def generate(self, messages):
        devices = sorted({p.device.index for p in self.model.parameters() if p.device.type == "cuda"})
        for device in devices:
            self.torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = super().generate(messages)
        for device in devices:
            self.torch.cuda.synchronize(device)
        self.last_generation.update(inference_backend="fla_triton_single_gpu",
                                    synchronized_seconds=round(time.perf_counter() - started, 6))
        return result
