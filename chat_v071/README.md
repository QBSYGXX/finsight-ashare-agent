# Chat contract patch v0.7.1

This isolated entrypoint wraps the existing v0.7 resolver and frozen agent.
It does not change training, evaluation, the step budget, or audited facts.
It depends on the installed v0.7 add-on and checks its file hashes.

## Incident and scope

The question `贵州茅台2024H1归母净利润、毛利率以及他们的同比增幅`
was silently interpreted as annual data, attempted an unsupported gross-margin
table metric, and exhausted the eight-step budget. No complete answer was returned.

The current fact schema supports annual facts only, for 2023-2025. The patch
rejects common half-year, quarterly, monthly and rolling-period requests before
model loading. It does not assert that the company never published those reports.
Explicit multi-result lists receive a decomposition message because the existing
final-result schema has one value. It does not execute a multi-part research plan.

The checks are conservative pattern checks, not a complete temporal-language
parser. A passed check does not prove a question is supported or an answer correct.
Single-result calculations still use the existing agent and may fail. Gross
margin requires audited revenue and cost plus the `gross_margin` calculation;
there is no `table_read(metric='gross_margin')` field. This patch does not add one.

## Server

Upload `finsight-chat-v0.7.1.zip` and its `.sha256` file to
`/calc/users/geyuboao/workspace/FinSight` and run:

```bash
cd /calc/users/geyuboao/workspace/FinSight
sha256sum -c finsight-chat-v0.7.1.zip.sha256 && unzip -n finsight-chat-v0.7.1.zip
bash chat_v071/run.sh chat --model tuned
```

Enter the original H1 question first. Expected: a period-not-supported message,
without `AGENT_STEP`, model loading or annual financial numbers. Enter `exit` to
leave. Trace and HTML reports are saved under `outputs/chat_v071/`.

For annual smoke checks, ask separately:

- 贵州茅台2024年全年合并口径归母净利润是多少亿元？
- 贵州茅台2024年全年合并口径归母净利润比2023年增长多少百分比？
- 贵州茅台2024年全年合并口径毛利率是多少？

Do not replace an H1 request with annual data unless the user changes the question.
Gross-margin year-over-year growth and percentage-point change are different
quantities. A composite result and H1 support require separate implementation.

Rollback: use `bash evaluation_v07/run.sh chat --model tuned`. It retains the
reported period-handling defect; no files from that entrypoint are overwritten.

## Local checks

```bash
python -B -m unittest discover -s chat_v071 -p 'test_*.py' -v
python -B chat_v071/chat.py --check '贵州茅台2024H1归母净利润、毛利率以及他们的同比增幅'
```

No server GPU inference is claimed by these tests.
