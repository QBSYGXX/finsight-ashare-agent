# FinSight bounded thinking v0.10

Keeps thinking enabled on the base Qwen3.5-9B model, one visible GPU, original
PyTorch kernels, and the exact v0.8.4 financial planner/prompt/engine. No FLA,
retraining, new weights, data changes, or environment installation.

Default protocol: up to 1024 thinking tokens, followed by up to 512 final-answer
tokens. A natural `</think>` ends phase 1 early; otherwise the program appends
`\n</think>\n\n` at the thinking budget. This is an experimental intervention,
not a claim that the model completed its reasoning. `forced_close` is recorded.
Phase 2 continues the same assistant turn from original token IDs; it performs
a second prefill (does NOT reuse KV state across calls). Both phases use the
existing sampling parameters and one seed reset per question. The second prefill
cost and phase-boundary handling are included in the reported generation time.

Only the final text after the closed thinking segment is parsed. Draft JSON is
never executed. Invalid plans, reopened thinking, missing requested metrics, and
answer-budget exhaustion remain failures. No hidden fallback or automatic budget
increase. This controls generation length, not a guaranteed wall-clock timeout.

## Server operation

Upload ZIP and SHA256 to `/calc/users/geyuboao/workspace/FinSight`, then:

```bash
cd /calc/users/geyuboao/workspace/FinSight &&
sha256sum -c finsight-thinking-v0.10.zip.sha256 &&
unzip -n finsight-thinking-v0.10.zip &&
CUDA_VISIBLE_DEVICES=0 bash thinking_v010/run.sh smoke --background
```

Follow the printed `TAIL=...` command. A 30-second generation heartbeat is emitted.
This smoke question is the same ZTE 2024 gross-margin/debt-ratio question; a
non-completed answer yields exit 1. Result ZIP and SHA256 are exported even on
failure. Upload that result for inspection; do not treat launch as completion.

After checking smoke:

```bash
CUDA_VISIBLE_DEVICES=0 bash thinking_v010/run.sh chat
CUDA_VISIBLE_DEVICES=0 bash thinking_v010/run.sh evaluate --background
```

`evaluate` runs six validation-company diagnostic cases under original 8192-token
thinking and the bounded protocol, using one loaded model. It records plan or
abstention matching, completion statuses, generation tokens and time. Unsupported
period and ambiguous ratio questions may be rejected by preflight without model
inference; these must not be counted as inference speed wins. This is NOT the
held-out test set or a general financial accuracy benchmark. The original v0.8.4
historical-query limitations remain intentionally present in both arms.

Options: `--think-budget 1024 --answer-budget 512 --context-length 16384 --seed 42`.
Default GPU is 0 only when CUDA_VISIBLE_DEVICES is unset; a multi-GPU selection is
rejected. The exact runtime model placement is checked. The base model is used;
the previous tool-calling LoRA is not silently applied to the JSON planner.

Rollback: run the existing `metrics_v084/run.sh` with thinking on; original files
are not overwritten. Local protocol tests and package checks cannot establish GPU
speed, actual EOS behavior or query correctness. These require server results.
