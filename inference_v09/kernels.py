"""Adapt pinned FLA Triton convolution to Qwen's channel-first/cache interface.

Avoids the causal-conv1d wheel whose GLIBC_2.32 requirement exceeds the server's
recorded glibc 2.28. No installed Transformers file is changed.
"""


def conv_prefill(x, weight, bias=None, activation=None, seq_idx=None):
    if seq_idx is not None or x.ndim != 3:
        raise ValueError("Only ordinary channel-first batched sequences are supported")
    from fla.modules.conv import causal_conv1d
    y, _ = causal_conv1d(x=x.transpose(1, 2), weight=weight, bias=bias,
                        activation=activation, backend="triton", output_final_state=False)
    return y.transpose(1, 2)


def tensor_layout(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device),
            "contiguous": tensor.is_contiguous()}


def conv_decode(x, conv_state, weight, bias=None, activation=None):
    reason = None
    if x.ndim != 3 or x.shape[-1] != 1:
        reason = "Expected one channel-first decode token [B,D,1]"
    elif weight.ndim != 2 or conv_state.ndim != 3:
        reason = "Expected weight [D,W] and cache [B,D,W]"
    elif (x.shape[:2] != conv_state.shape[:2] or x.shape[1] != weight.shape[0]
          or conv_state.shape[-1] != weight.shape[-1]):
        reason = "Decode input, weight and cache dimensions do not match"
    if reason:
        raise ValueError(f"{reason}; layouts=" + str({
            "x": tensor_layout(x), "cache": tensor_layout(conv_state),
            "weight": tensor_layout(weight)}))
    from fla.modules.conv.triton.ops import causal_conv1d_update
    # FLA's input_guard otherwise makes a private contiguous copy and its in-place
    # update is lost to Qwen. Keep Qwen's original storage/aliases up to date.
    cache = conv_state if conv_state.is_contiguous() else conv_state.contiguous()
    y, _ = causal_conv1d_update(x=x.transpose(1, 2), cache=cache, weight=weight,
                               bias=bias, activation=activation)
    if cache is not conv_state:
        conv_state.copy_(cache)
    return y.transpose(1, 2)


def enable():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    if module.chunk_gated_delta_rule is None or module.fused_recurrent_gated_delta_rule is None:
        raise RuntimeError("FLA gated-delta kernels did not load")
    module.causal_conv1d_fn = conv_prefill
    module.causal_conv1d_update = conv_decode
    module.is_fast_path_available = all((module.causal_conv1d_fn, module.causal_conv1d_update,
                                       module.chunk_gated_delta_rule, module.fused_recurrent_gated_delta_rule))
    return {"gated_delta": "FLA 0.5.2", "convolution": "FLA 0.5.2 Triton adapter",
            "native_causal_conv1d_wheel": False}
