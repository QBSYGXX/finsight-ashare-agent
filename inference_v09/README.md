# FinSight inference acceleration experiment v0.9.1

This add-on keeps the Qwen3.5-9B BF16 model, financial facts and planner unchanged
while changing execution kernels and placement. It installs two hash-pinned FLA
0.5.2 wheels into a separate `.runtime/inference-v09/packages-*` overlay; it never
upgrades or replaces the existing portable Torch/Transformers environment.

The original `metrics_v084` planner is embedded verbatim from its verified archive
so both benchmark arms use the same prompt as the recorded thinking-on query.
This release does not incorporate the different v0.8.5 prompt. No training occurs.

## Acceleration

- FLA chunk/recurrent gated-delta kernels and fused gated normalization.
- FLA's own Triton causal convolution, through a small channel-layout/cache adapter.
  Native causal-conv1d 1.7.0 was inspected and excluded: its official Torch 2.8
  wheel requires GLIBC_2.32, whereas the supplied server diagnostic reports 2.28.
  The included wheels are pure Python/Triton, with no new glibc-linked extension.
- Fixed single visible GPU (default `--gpu 0`) rather than automatic placement.
  CUDA_VISIBLE_DEVICES is respected; no changes to scheduler allocation are made.
- Explicit KV cache and last-position logits; BF16 and SDPA are retained.
  First Triton compilation is warmed up and excluded from measurements.

This is not a vLLM deployment or four-GPU tensor parallelism. A 9B model fits in a
free 80GB A800; more GPUs are not automatically faster for one generated sequence.

## Install and measure

Upload ZIP and SHA256 alongside the existing project, then run:

```bash
cd /calc/users/geyuboao/workspace/FinSight
sha256sum -c finsight-inference-v0.9.1.zip.sha256 && unzip -n finsight-inference-v0.9.1.zip
bash inference_v091/run.sh benchmark --gpu 0 --background
```

The command prints PID and log path. Follow that exact log with `tail -n 60 -f`.
The background process survives disconnect. Ctrl-C in tail only closes the viewer.
Completion is recorded in status.json and printed with EXIT; launch is not success.

The benchmark automatically performs:

1. Package verification, Python/Torch/CUDA/Triton/ABI and GPU-memory diagnostics.
2. Offline `pip --no-index --no-deps --target` install into an isolated overlay.
3. GPU equivalence checks: convolution outputs/cache updates, then the same small
   random Qwen model under reference and accelerated implementations, including
   prefill and eight cached decoding steps (BF16 tolerances recorded in results).
   A failed check stops before loading 9B. This does not prove full-model equivalence.
4. Baseline and fast workers in separate processes. Each warms up then runs two
   fixed-work 128-token measurements and one real question with thinking ON,
   sampling seed 42, output limit 8192 and context limit 16384.
5. A report with placement, model load time, fixed-work latency, real generated
   token counts, final plans/financial answers and evidence. A results ZIP is
   exported automatically even if installation/probing fails.

The fixed-work rate includes prefill; it is not pure decode-only throughput. The
real question can generate different lengths after kernel arithmetic changes,
so its latency alone must not be interpreted as a kernel speedup. Check both
`FIXED_WORK_SPEEDUP` and `SAME_COMPLETED_ANSWER`. One question is a diagnostic,
not an accuracy evaluation. A failed answer does not become successful because
the process completed or its fixed-work speed increased.

Download the exact ZIP printed as `RESULTS=...` for inspection. It contains small
diagnostics, benchmark traces, an HTML answer report and the log, not model weights.
Keep the old entrypoints as the baseline; rollback is simply running those again.

After a successful server check, interactive accelerated use is:

```bash
bash inference_v091/run.sh chat --gpu 0 --thinking on --max-new-tokens 8192 --context-length 16384
```

The model remains loaded across questions. `--thinking off --max-new-tokens 2048`
is available but is not used to claim the thinking-on speedup. `doctor` runs a
read-only report; `install` only prepares and probes the overlay. `--gpu` selects
a logical visible device; select one with enough free memory (28 GiB minimum).

## Provenance and limits

The package embeds FLA/fla-core wheels with their original licenses and metadata;
`dependencies.json` records public source URLs and official PyPI SHA256 hashes.
The wheel versions require existing Torch >=2.7 and Triton >=3.3; this handoff pins
the recorded Python 3.11 / Torch 2.8.0+cu126 / Transformers 5.8.0 / Triton 3.4.0
environment. New GPU kernels still require an actual driver/compiler check.

Local verification covers source logic, kernel-adapter interfaces, package hashes,
entrypoints and preservation of the original metrics source. The local Docker
daemon was unavailable, so native Linux imports and A800 numerical/speed results
remain pending server execution. No speedup factor is claimed before that run.

The old thinking-mode repetition is not fixed by this performance change; the
same output budget and prompt are used to distinguish faster execution from
shorter reasoning. Existing data coverage and financial-evidence restrictions
remain in force.

## v0.9.1 cache compatibility fix

The first v0.9 server probe failed at the decode-cache layout guard, before the
9B benchmark. Qwen prefill can preserve transposed/noncontiguous cache storage.
The adapter now uses a contiguous work buffer when needed and copies updates
back into the original cache, retaining its identity and aliases. Simply removing
the guard would lose updates through FLA's automatic contiguous conversion.
Shape errors include shape, stride, dtype and device diagnostics.

The GPU gate now checks contiguous, transposed and sliced caches for batches 1/2,
with eight consecutive decode updates and exact cache comparisons, then checks a
small Qwen model over prefill plus eight decode steps. Actual Qwen cache layouts
are logged. Existing numerical tolerances are unchanged.

The ZIP installs into `inference_v091/`; previous packages and results are kept.
The `.runtime/inference-v09` wheel/JIT cache location is intentionally reused.
Source remains in `inference_v09/` locally; Git and package manifests identify the
version. CPU interface tests cannot establish CUDA correctness or speedup.
