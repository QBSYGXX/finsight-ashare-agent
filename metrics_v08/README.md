# Financial metric engine v0.8.5

The previous agent treated derived metrics as raw fields, changed issuer codes
after argument errors, and sometimes called those failures missing data. This
add-on introduces one declarative metric registry and one dependency executor.
It is an engineering extension, not a new LoRA result or a replacement score for
the frozen v0.7 evaluation. Existing source, training data and adapters are intact.

Version 0.8.1 added multi-year queries; v0.8.2 introduced non-thinking generation.
Version 0.8.3 adds explicit thinking and token-budget controls.
Version 0.8.4 adds selectable sampling for a controlled repetition diagnostic.
Version 0.8.5 clarifies period selection versus year-over-year transformations.
Local source remains in `metrics_v08/`; the upload package installs into the new
`metrics_v085/` directory. Earlier entrypoints and outputs remain for rollback.

## Unrequested Year-Over-Year Diagnostic

In the supplied v0.8.4 non-thinking trace, the question about all available years
of Moutai parent-attributable net profit produced both `value` and `yoy_growth`.
Only `value` was requested. The model reached EOS, so this was a semantic planning
failure, not a token-limit failure. Existing validation correctly rejected it.

The prompt now explicitly separates year selection from transformations and gives
three company-independent revenue examples: historical values only, growth only,
and values plus growth. Historical coverage does not itself request a comparison.
These examples contain no financial answers or held-out company data. The parser
still rejects extra and missing requests, with no pruning, rule fallback or retry.
Mismatch traces include expected/proposed requests and extra/missing items; the
terminal also prints `PLAN_EXPECTED` and `PLAN_PROPOSED` for immediate inspection.

Prompt changes need real model validation. Tests replay the observed wrong plan
to ensure it stays rejected; scripted correct plans verify that explicitly asked
growth is still allowed. Neither test measures whether the model follows the new
prompt. Thinking repetition is a separate unresolved issue; use non-thinking
mode for this comparison with the same sampling seed and budgets as v0.8.4.

## Repeated Thinking Diagnostic

The supplied v0.8.3 trace exhausted 2048 generated tokens (745 input tokens;
8192 context limit; 117.351 seconds). Its tail repeatedly drafted the same two
correct metric IDs, without closing the thinking segment. This proves repeated
deliberation in that trace, but does not by itself prove the decoding algorithm
caused it or that a larger budget could never finish.

That run used `do_sample=False`. The local Qwen3.5-9B model card recommends
sampling profiles (README Sampling Parameters / Best Practices). For structured
planning, this add-on uses its precise-coding profile: temperature 0.6, top_p 0.95,
top_k 20, min_p 0.0, repetition_penalty 1.0. The profile's presence penalty is zero;
no custom presence-penalty processor is added to Transformers.

`--decoding auto` now selects sampling when thinking is on, otherwise greedy.
`--decoding sample` and `--decoding greedy` make either choice explicit. `--seed 42`
is the default, reset at each sampled query and stored with decoding parameters
in the trace. A seed aids same-environment comparison, not universal bitwise
reproducibility across GPU configurations or library versions.

First rerun the same base-model question with the same 2048/8192 budgets and
`--decoding sample --seed 42`. Only decoding changes in this comparison; the
prompt, parser, engine and facts are retained. No actual GPU success is claimed
by this release. Thinking drafts are still never treated as final plans. If the
run again hits the limit, keep its trace before trying another change.

## Planner Generation Fix

The reported base-model response was an unfinished `<think>` segment with no
final JSON. The former hand-built prompt did not explicitly disable thinking;
the shared generation cap was 512 new tokens. No token telemetry was recorded
for that old response, so its precise stopping reason cannot be confirmed.

`planner_model.py` reuses the frozen model loader but replaces generation only
for this add-on. It calls the snapshot's native `apply_chat_template` with
the selected `enable_thinking` flag and verifies the matching prefix. The default
`--thinking off` uses a 4096-token context and 512-token output cap; `--thinking on`
defaults to an 8192-token context and 2048-token output cap. Override these with
`--context-length` (4096..32768) and `--max-new-tokens` (64..8192). Output must leave
space for the prompt. The actual output budget is capped by remaining context;
`context_limited`, requested/effective output budgets and input tokens are logged.
These are add-on inference settings, not a change to the training context.
Neither the frozen agent prompt nor the LoRA adapter is changed.

Logs print generation start/end metadata and a heartbeat every 30 seconds while
generation is running. Traces preserve raw output, the planning prompt, input
and output token counts, output budget, generation time and stop reason.
`generation_limit` means the cap was reached without an EOS token; no plan is
executed even if a JSON prefix looks usable. `invalid_plan` still means a JSON
or schema failure. In thinking mode the native prompt already opens `<think>`;
only the final text after exactly one closing `</think>` is JSON-decoded. JSON
drafts inside the thinking segment are never executed. Raw output and the final
plan text are both retained. Missing/duplicate closing tags, incomplete JSON and
Markdown fences are rejected, not repaired or converted to rule-mode success.

This fixes the prompt contract, not an evaluated claim of model reliability.
The native template is checked locally with Jinja; fake-generation tests do not
replace an actual server GPU run. Start with the base model. The existing tuned
adapter learned the old tool-call format, not this JSON-only protocol.

## What the Deterministic Tools Do

The model selects registered indicator IDs and transformations. `registry.json`
defines formulas, and `engine.py` reads previously audited facts and computes with
Decimal arithmetic. Gross margin is `(revenue - cost) / revenue * 100`; debt ratio
is `liabilities / assets * 100`. The engine checks units, issuer, period, statement
scope, source metadata and comparative basis before calculating. Missing total
net profit blocks net margin; parent-attributable profit is not substituted.
It retains existing report/page/hash evidence, but it does not re-audit PDF text
or guarantee that upstream human-reviewed facts are correct. No extra installation
or model-generated Python is involved.

## Architecture

`question -> issuer/period/scope checks -> constrained metric plan -> dependency graph -> audited facts -> Decimal calculations -> multiple results with provenance`

- `registry.json` holds raw field mappings, aliases, formulas, input offsets,
  units and domain restrictions. Six original raw fields remain unchanged.
  `total_net_profit` is explicitly distinct from the existing `net_profit`
  (attributable to parent shareholders).
- `engine.py` implements subtract, average, ratio and percentage ratio. Formulas
  compose these primitives and may depend on other metrics. No `eval`, generated
  Python, hardcoded company result, or model-supplied formula is executed.
- Annual value, year-over-year growth, difference and percentage-point change
  share one implementation. Reads are cached per request, including shared
  dependencies. Each result independently succeeds or reports missing fields,
  incompatible reporting basis or a mathematical domain error.
- Same-period inputs require the same report snapshot. Cross-year inputs require
  adjacent years and a verified prior comparative column equal to the old fact.
  Restated or unavailable comparatives block the calculation.
- Every successful result carries the input values and units, exact formula,
  fact IDs, report IDs, PDF pages, source URLs and PDF hashes. The output renderer
  does not ask a model to rewrite or invent the numbers.

## Two Planning Modes

The default `rules` mode uses registered aliases and a deliberately limited query
grammar. It does not load a language model or retrieve PDF passages. Unknown
remaining words cause clarification rather than silently dropping requested work.
For mixed tasks, put the transformation after each indicator and separate items
with commas. `以及它们的同比增长率` applies to all listed indicators. Ratio `同比增幅`
is ambiguous and requires a growth-rate versus percentage-point definition.

The optional `model` mode loads the existing base model (or `--model tuned`) to
generate a JSON plan containing only registered metric IDs and transformation
IDs. It cannot select a different issuer, year, scope, formula, result or source.
Plans are schema-checked; omissions of explicit metric mentions and disagreements
with an unambiguously rule-parsed request are rejected. For novel wording, the
model's semantic interpretation is still not independently proven. Read `PLAN`
and the printed result names. Invalid JSON is a planning failure, not missing data.

The tuned model was not trained for this JSON planner protocol. Real model-mode
reliability has not been measured. The rule executor's successful tests must not
be described as improved neural reasoning or agent accuracy.

## Year Selection

`所有年份`, `全部年份`, `所有年度`, `历年` and `各年` use the years actually
represented by verified facts for the resolved issuer and statement scope. They
do not mean the company's entire reporting history. A year with some available
facts but a missing requested operand remains in the output with a missing-field
status. If the requested scope has no verified years, no other scope is substituted.

Explicit lists such as `2023、2025年` keep exactly those years. Inclusive ranges
such as `2023至2025年` or `2023-2025年` expand to all years in the interval.
Unavailable explicitly requested years remain visible as missing rows. Date and
quarter expressions are still rejected, rather than being read as annual ranges.

One set of metric requests applies to every selected year. Interleaved requests
such as `2023年毛利率、2024年资产负债率` ask for clarification rather than creating
an unintended cross product. A single call is limited to 12 years and 120 result
rows. YoY operates separately for each year using the preceding year; missing
predecessors stay missing. Dependencies are cached across the whole query.

Rule and model planning share the same program-selected year list. The language
model cannot change or invent the year coverage. Terminal rows and HTML tables
include a year column. Repeated report/page labels are deduplicated for display;
both operand fact IDs remain in the trace.

## Definitions and Data Coverage

Direct fields: revenue, cost, parent-attributable net profit, operating cash flow,
assets and liabilities. Derived: gross profit, gross margin, parent profit divided
by revenue, debt-to-assets, total equity, equity ratio, operating cash flow divided
by parent profit, and total asset turnover using beginning/end average assets.

Net profit including non-controlling interests and sales net margin are defined,
but the frozen snapshot lacks their required `total_net_profit` field. They return
the missing field explicitly. Parent profit is never substituted. Total equity
is assets minus liabilities, not equity attributable to parent shareholders.

No ROE/ROIC, segment ratios, weighted-average equity, quarterly/H1/TTM data,
multiple companies or arbitrary cross-year comparison formulas are claimed.
One statement scope is supported. Prior-year operands are added automatically.
YoY of an already averaged indicator can span three fiscal years; this version
rejects that basis instead of assuming comparability. Negative/zero denominators
and nonpositive prior values are rejected under the declared definitions.

Adding a metric supported by existing fields and primitives only needs a reviewed
registry entry and numerical tests. Adding a new primitive needs code review and
tests. Adding missing raw fields needs a separately audited/versioned data snapshot
and a new dependency pin; do not edit the frozen training dataset to make a query pass.

## Server Installation

Requires installed v0.7 resolver and v0.7.1 preflight files. Upload the ZIP and
checksum to `/calc/users/geyuboao/workspace/FinSight`:

```bash
cd /calc/users/geyuboao/workspace/FinSight
sha256sum -c finsight-metrics-v0.8.5.zip.sha256 && unzip -n finsight-metrics-v0.8.5.zip
bash metrics_v085/run.sh chat
```

Try these separately:

```text
告诉我你能查询到的所有年份贵州茅台的毛利率
贵州茅台2023至2025年毛利率、资产负债率
贵州茅台2024年毛利率、净利率和资产负债率
贵州茅台2024年归母净利润及其同比增长率
贵州茅台2024年毛利率同比变化多少个百分点
贵州茅台2024年总资产周转率
贵州茅台2024年母公司营业收入
```

Or one command:

```bash
bash metrics_v085/run.sh --question '告诉我你能查询到的所有年份贵州茅台的毛利率'
```

Optional model planning (requires server GPU):

```bash
bash metrics_v085/run.sh chat --planner model --model base --thinking off --decoding sample --seed 42 --max-new-tokens 2048 --context-length 8192
```

For the v0.8.5 regression, try `贵州茅台可查询到的历年归母净利润`,
`贵州茅台可查询到的历年归母净利润同比增长率`, and
`贵州茅台可查询到的历年归母净利润及其同比增长率` separately. Expected plans
are respectively `value`, `yoy_growth`, and both. A correct growth plan can still
yield missing-comparative results under the audited data contract.

Use `chat --planner model --model base --thinking on` for repeated questions in
one process, keeping the model loaded. Use `--thinking off` for a non-thinking
comparison. A larger token cap is an upper bound, not a promise that generation
will finish or the plan will be correct. No new server model results are claimed.

`--catalog` prints the registry. `exit` closes the session. Execution logs are in
`runs/*-metrics-v085-*/run.log`, with an exit code. Query traces and standalone HTML
reports are in `outputs/metrics_v085/`. A successful process exit does not imply all
requested financial results succeeded: inspect each row and the overall status.
No SSH tunnel, new Python packages or model download is needed. Rollback is simply
using a previous entrypoint; this package does not overwrite previous versions.

## Local Validation

```bash
python -B -m unittest discover -s metrics_v08 -p 'test_*.py' -v
```

Tests cover shared dependencies, missing fields, statement scope, report basis,
unit normalization, zero/negative denominators, growth versus percentage points,
registry cycles, multi-output planning and invalid model-proposed plans. Scripted
planner outputs are test fixtures, not real model inference. Multi-year tests
cover the user's exact question, year lists versus ranges, partial missing years,
per-year YoY, coverage by company/scope and ambiguous year-to-metric assignments.
Planner generation tests additionally cover the actual local template, no-thinking
flag, EOS versus token-cap termination, context budgeting, invalid-plan diagnostics
and non-execution of unfinished thinking output. No GPU inference is simulated as
a real model score.

Verified against the frozen fact digest, using only a training company:

| Question | Local deterministic result |
| --- | --- |
| Moutai 2023 gross margin | 91.9649%, PDF page 63 |
| Moutai 2024 gross margin | 91.9312% |
| Moutai 2025 gross margin | 91.1796%, PDF page 61 |
| Moutai 2024 sales net margin | Missing total net profit including minority interests |
| Moutai 2024 debt ratio | 19.0448% |
| Moutai 2024 parent-attributable profit | 862.2815 亿元 |
| Its YoY growth | 15.3800% |
| Gross margin YoY percentage-point change | -0.0337 percentage points |
| Total asset turnover | 0.5979 |
| Parent-only revenue | Missing parent-scope field |

These are snapshot-based calculations, not newly audited PDFs, independent
test-set results, or server model-mode results.
