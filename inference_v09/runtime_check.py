"""Read-only environment report and a small GPU equivalence gate for native kernels."""
import argparse
import copy
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def profile():
    import torch
    versions = {}
    for name in ("torch", "transformers", "triton", "einops", "flash-linear-attention", "fla-core", "causal-conv1d"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "platform": platform.system(), "machine": platform.machine(),
              "glibc": platform.libc_ver(), "versions": versions, "cuda": torch.version.cuda,
              "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI), "cuda_available": torch.cuda.is_available(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpus": []}
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        result["gpus"].append({"index": index, "name": torch.cuda.get_device_name(index), "free_bytes": free, "total_bytes": total})
    try:
        result["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        result["nvidia_smi_error"] = str(error)
    return result


def validate_profile(report, expected):
    errors = []
    for key in ("machine", "cxx11_abi"):
        if report[key] != expected[key]:
            errors.append(key)
    if not report["python"].startswith(expected["python"] + ".") or report["platform"] != "Linux":
        errors.append("Linux Python ABI")
    for key in ("torch", "transformers", "triton"):
        if report["versions"].get(key) != expected[key]:
            errors.append(key)
    if not report["cuda_available"]:
        errors.append("CUDA unavailable")
    if errors:
        raise RuntimeError("Environment differs from pinned wheels; no base environment change: " + ", ".join(errors))


def kernel_probe(gpu):
    import torch
    from kernels import enable, conv_prefill, conv_decode, tensor_layout
    torch.cuda.set_device(gpu)
    backend = enable()
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    if not module.is_fast_path_available:
        raise RuntimeError("Qwen3.5 optimized FLA/causal-conv path is not active")
    torch.manual_seed(42)
    # Verify convolution axes and in-place cache updates independently of the model.
    conv_errors = []
    for batch in (1, 2):
        x = torch.randn(batch, 32, 17, device=f"cuda:{gpu}", dtype=torch.bfloat16)
        weight = torch.randn(32, 4, device=x.device, dtype=x.dtype) * 0.1
        actual = conv_prefill(x, weight, activation="silu")
        expected = torch.nn.functional.silu(torch.nn.functional.conv1d(x, weight[:, None], padding=3, groups=32)[..., :17])
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.015, rtol=0.03)
        # Exercise both Qwen-like transposed caches and sliced backing storage.
        states = {
            "contiguous": x[..., -4:].contiguous().clone(),
            "transposed": x[..., -4:].transpose(1, 2).contiguous().transpose(1, 2),
            "strided_slice": x[..., -8::2],
        }
        for layout, state in states.items():
            print("CONV_PROBE=" + json.dumps({"batch": batch, "layout": layout,
                  "cache": tensor_layout(state)}), flush=True)
            reference_state = state.contiguous().clone()
            state_pointer = state.data_ptr()
            for step in range(8):
                one = torch.randn(batch, 32, 1, device=x.device, dtype=x.dtype)
                actual = conv_decode(one, state, weight, activation="silu")
                expected = module.torch_causal_conv1d_update(one, reference_state, weight, activation="silu")
                torch.testing.assert_close(actual.float(), expected.float(), atol=0.015, rtol=0.03)
                torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
                assert state.data_ptr() == state_pointer
            conv_errors.append({"batch": batch, "layout": layout, "decode_steps": 8,
                                "max_abs_error": float((actual.float() - expected.float()).abs().max())})
    text = Qwen3_5TextConfig(vocab_size=128, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, full_attention_interval=2,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, mtp_num_hidden_layers=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 3, 3], "mrope_interleaved": True})
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config={"depth": 1, "hidden_size": 32,
        "intermediate_size": 64, "num_heads": 4, "out_hidden_size": 256, "patch_size": 2,
        "spatial_merge_size": 1, "temporal_patch_size": 1, "num_position_embeddings": 16})
    config._attn_implementation = "sdpa"
    fast = Qwen3_5ForConditionalGeneration(config).to(device=f"cuda:{gpu}", dtype=torch.bfloat16).eval()
    reference = copy.deepcopy(fast)
    count = 0
    for layer in reference.modules():
        if hasattr(layer, "chunk_gated_delta_rule"):
            layer.causal_conv1d_fn = None
            layer.causal_conv1d_update = module.torch_causal_conv1d_update
            layer.chunk_gated_delta_rule = module.torch_chunk_gated_delta_rule
            layer.recurrent_gated_delta_rule = module.torch_recurrent_gated_delta_rule
            norm = module.Qwen3_5RMSNormGated(layer.head_v_dim, eps=layer.layer_norm_epsilon).to(device=f"cuda:{gpu}", dtype=torch.bfloat16)
            norm.load_state_dict(layer.norm.state_dict())
            layer.norm = norm
            count += 1
    assert count > 0
    ids = torch.arange(1, 33, device=f"cuda:{gpu}").unsqueeze(0)
    cache_fast = cache_ref = None
    errors = []
    print("KERNEL_PROBE=prefill_and_cached_decode; first call may compile Triton kernels", flush=True)
    with torch.inference_mode():
        for step in range(9):
            current = ids if step == 0 else ids[:, step:step + 1]
            if cache_fast is not None and step == 1:
                for index, layer in enumerate(cache_fast.layers):
                    state = getattr(layer, "conv_states", None)
                    if state is not None:
                        print("QWEN_CACHE=" + json.dumps({"layer": index, **tensor_layout(state)}), flush=True)
            print(f"MODEL_PROBE_STEP={step}", flush=True)
            a = fast(input_ids=current, past_key_values=cache_fast, use_cache=True)
            b = reference(input_ids=current, past_key_values=cache_ref, use_cache=True)
            assert torch.isfinite(a.logits).all() and torch.isfinite(b.logits).all()
            torch.testing.assert_close(a.logits.float(), b.logits.float(), atol=0.03, rtol=0.03)
            errors.append(float((a.logits.float() - b.logits.float()).abs().max()))
            cache_fast, cache_ref = a.past_key_values, b.past_key_values
    torch.cuda.synchronize(gpu)
    return {"status": "passed", "fast_path_available": True, "backend": backend,
            "conv_max_abs_errors": conv_errors, "max_abs_logit_errors": errors,
            "atol": 0.03, "rtol": 0.03, "dtype": "bfloat16", "note": "small random model, not 9B answer equivalence"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["doctor", "probe"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    try:
        result = profile()
        if args.mode == "probe":
            result["kernel_probe"] = kernel_probe(args.gpu)
        save(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        save(args.output, {"status": "failed", "error": repr(error)})
        raise
