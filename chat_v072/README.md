# Chat v0.7.2: derived metrics and execution boundaries

The server accepted a raw `gross_margin` read, received `Invalid metric or year`,
changed company from 600519 to 600418, and finally falsely claimed missing data.
The snapshot actually contains audited 2024 revenue and cost for 600519.

This additive entrypoint changes the chat protocol, not the frozen evaluation:

- Bind all company-specific reads, searches, calculations and final answers to
  the company resolved from the user's question. Do not silently rewrite an
  incorrect company code.
- Support `table_read(metric='gross_margin')` as an explicit composite tool:
  read audited revenue, read audited cost, and call the existing gross-margin
  calculator. Preserve all subcalls, operand facts, original citation IDs and
  source pages. Missing operands remain missing; parent scope is never replaced
  with consolidated data. Invalid basis, years or non-positive revenue fail.
- Give concrete tool feedback and add tool-contract instructions to the model.
- Validate structured final values against actual tool receipts. Block a missing
  evidence claim without a successful relevant missing lookup, or when a matching
  result already exists. Preserve rejected model text in `rejected_final`.
- Retain v0.7.1 period checks. Margin growth/percentage-point change is not
  implemented; it must not be answered with a single-year margin.

This is engineering support, not evidence the LoRA learned these new behaviors.
The numeric checks do not prove free-text prose correctness or cover every intent.
The original model may still exhaust its step budget or fail answer validation.
Do not combine v0.7.2 chat results with the frozen v0.7 evaluation score.

## Server

Requires existing v0.7 and v0.7.1, verified by dependency hashes. Upload ZIP and
checksum to `/calc/users/geyuboao/workspace/FinSight`, leave the old chat with
`exit`, then run:

```bash
cd /calc/users/geyuboao/workspace/FinSight
sha256sum -c finsight-chat-v0.7.2.zip.sha256 && unzip -n finsight-chat-v0.7.2.zip
bash chat_v072/run.sh chat --model tuned
```

Ask `贵州茅台2024年毛利率`. Expected tool result from the present annual fact
snapshot: `(170899152276.34 - 13789482367.98) / 170899152276.34 * 100`, in percent,
with citations `600519-2024-revenue` and `600519-2024-cost`, PDF physical page 63.
This expectation is a deterministic tool check, not a claimed GPU inference run.
Also check a quarterly question is rejected and parent-scope data is not replaced.

Traces are in `outputs/chat_v072/`. Returning to the v0.7.1 entrypoint is a rollback
but also restores its observed tool-handling defects. No old server files are
overwritten. LoRA weights and training data are untouched.
