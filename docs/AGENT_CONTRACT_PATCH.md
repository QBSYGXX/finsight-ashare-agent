# Agent protocol correction (v0.3.2)

The server smoke trace for 20260924-000639 showed seven rejected tool calls:
the model copied a placeholder keyword from the system prompt and nested all
arguments inside it. Its last answer also omitted the closing final_result tag.
The GPU environment, model snapshots and deterministic tool replay had passed.

This patch replaces the placeholder with executable syntax examples using an
unrelated company/year. Dispatch errors report the actual Python tool signature.
An incomplete JSON answer still fails validation and prompts a fresh response.
Each step, model response and validation failure is now printed into run.log.
The parser, evidence checks and numerical scoring remain strict. No expected
answer is inserted, and no malformed call is silently converted to a valid one.

The subsequent 20260924-135648 server run successfully read the cost fact and
converted its unit, but repeatedly omitted only the closing final_result tag.
Version 0.3.2 adds that tag only after an explicit opening tag followed by one
complete JSON object that passes the existing final-field validation. Truncated
JSON, missing fields, trailing prose, extra actions and incomplete tool calls
are not repaired. Only leading complete thinking envelopes are ignored; text
inside JSON strings is preserved. The strict parse_action API still requires
complete tags. Retrieved-citation checks and numeric scoring are unchanged.

Raw model output is preserved in response and messages. A repaired event adds
format_repair=missing_final_result_close and normalized_response; run.log prints
AGENT_FORMAT_REPAIR. Completion of parsing does not imply a correct answer:
the evidence and numeric scorer must also pass. The same normalization applies
to direct, base and tuned evaluation modes through the shared ReportAgent.

Local regression checks use scripted responses; they are not evidence that
the 9B model succeeds. Repeat the real smoke test on the server after installing.

Local v0.3.2 validation: 71 tests passed, 4 Linux-only tests skipped on Windows.
The first three actual AGENT_RESPONSE records from run 20260924-135648 were
also replayed against the local audited facts: two tool calls succeeded and
structured task correctness, citation precision and citation recall all passed.
The fact's physical PDF page is 63. This is historical-output replay, not a new
GPU inference result. Its record is outputs/verification/final-envelope-v0.3.2.json
on the development machine; it is not a replacement for the server smoke trace.

Upload the ZIP and its SHA256 file into the existing FinSight directory:

```bash
cd /calc/users/geyuboao/workspace/FinSight &&
sha256sum -c finsight-agent-fix-v0.3.2.zip.sha256 &&
unzip -o finsight-agent-fix-v0.3.2.zip &&
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 &&
bash server/run.sh smoke
```

The installed environment, model weights, financial data and old run records
are reused. Only the source/test/doc files in the patch manifest are replaced.
Success requires LLM_EXECUTED=True and final EXIT=0. On failure, return the
new AGENT_RESPONSE/AGENT_ERROR lines and the saved cases.json.
