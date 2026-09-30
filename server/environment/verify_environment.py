"""Linux binary imports, small Qwen forward/backward and LoRA serialization.

No base-model weights are downloaded. This cannot replace the A800 GPU probe.
"""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import tempfile


def verify():
    import torch
    import torchvision
    import torchaudio
    import transformers
    import peft
    import sentence_transformers
    import pymupdf
    import yaml
    from llamafactory.data.template import TEMPLATES
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, PeftModel, get_peft_model

    assert "qwen3_5_nothink" in TEMPLATES
    text = Qwen3_5TextConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, full_attention_interval=2,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, mtp_num_hidden_layers=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config={"depth": 1, "hidden_size": 32,
        "intermediate_size": 64, "num_heads": 4, "out_hidden_size": 64, "patch_size": 2,
        "spatial_merge_size": 1, "temporal_patch_size": 1, "num_position_embeddings": 16})
    config._attn_implementation = "sdpa"
    model = get_peft_model(Qwen3_5ForConditionalGeneration(config), LoraConfig(
        task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
    ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]])
    result = model(input_ids=ids, labels=ids, use_cache=False)
    assert torch.isfinite(result.loss), "Non-finite small-model loss"
    result.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients), "LoRA gradients missing/non-finite"
    with tempfile.TemporaryDirectory(prefix="finsight-peft-") as directory:
        model.save_pretrained(directory)
        reloaded = PeftModel.from_pretrained(Qwen3_5ForConditionalGeneration(config), directory)
        assert torch.isfinite(reloaded(input_ids=ids, labels=ids, use_cache=False).loss)
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "torchaudio", "transformers", "peft", "llamafactory",
        "sentence-transformers", "PyMuPDF", "PyYAML", "conda-pack", "setuptools")}
    return {"status": "passed", "platform": platform.system(), "architecture": platform.machine(),
            "glibc": platform.libc_ver(), "python": sys.version, "prefix": sys.prefix, "versions": versions,
            "small_qwen35_forward_backward": True, "lora_save_reload": True,
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "a800_gpu_tested": False, "base_9b_weights_included": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
