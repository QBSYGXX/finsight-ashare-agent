"""Two-stage token protocol, independent of Torch; never interprets draft JSON."""
CLOSE = "</think>"


def reserve_context(prompt_tokens, context_length, think_budget, answer_budget, bridge_tokens):
    if min(think_budget, answer_budget) < 1:
        raise ValueError("Budgets must be positive")
    required = prompt_tokens + think_budget + bridge_tokens + answer_budget
    if required > context_length:
        raise ValueError(f"Need {required} context tokens; configured {context_length}; input is not truncated")


def run_stages(prompt_ids, generate, decode, bridge_ids, eos, think_budget, answer_budget):
    # generate returns only NEW tokens; phase 2 consumes the original token IDs.
    first = generate(prompt_ids, think_budget, True, "thinking")
    thought = decode(first)
    closed = CLOSE in thought
    forced = False
    if first and first[-1] == eos:
        return first, {"thinking_tokens": len(first), "answer_tokens": 0,
                       "forced_close": False, "stop_reason": "eos", "early_eos": True,
                       "bridge_tokens": 0}
    if not closed:
        if len(first) != think_budget:
            raise ValueError("Thinking stopped unexpectedly before EOS, delimiter or budget")
        forced = True
    bridge = bridge_ids if forced else []
    second = generate(prompt_ids + first + bridge, answer_budget, False, "answer")
    reason = "eos" if second and second[-1] == eos else "max_new_tokens"
    return first + bridge + second, {"thinking_tokens": len(first), "answer_tokens": len(second),
        "forced_close": forced, "bridge_tokens": len(bridge), "stop_reason": reason, "early_eos": False}
