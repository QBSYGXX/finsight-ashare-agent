"""Read-only audit of the six-case v0.10 return archive; no inference required."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import zipfile


# 模块：实验产物核验；作用：重算评分、来源与数值；核心业务：否。
def audit(archive, facts_path):
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    assert digest == archive.with_suffix(".zip.sha256").read_text().split()[0]
    facts = {f["id"]: f for f in json.loads(facts_path.read_text(encoding="utf-8"))}
    scales = {"元": Decimal(1), "千元": Decimal(1000), "万元": Decimal(10000), "亿元": Decimal(100000000)}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        comparison = json.loads(z.read("comparison.json"))
        assert len(comparison["rows"]) == 12
        assert json.loads(z.read("status.json"))["exit_code"] == 0
        summaries, checked_results, checked_sources = [], 0, 0
        for i in range(1, 7):
            pair = []
            for j, policy in enumerate(("baseline", "budgeted")):
                trace = json.loads(z.read(f"case-{i}-{policy}/trace.json"))
                row = comparison["rows"][2 * (i - 1) + j]
                expected = row["case"]
                assert row["policy"] == policy and trace["question"] == expected["question"]
                if "requests" in expected:
                    actual = sorted([r["metric"], r["transform"]] for r in trace.get("plan", {}).get("requests", []))
                    match = actual == sorted(expected["requests"]) and trace["status"] in {"completed", "partial", "no_results"}
                else:
                    match = trace["status"] == expected["status"]
                assert match == row["plan_or_abstention_matches"]
                assert trace["status"] == row["status"]
                gen = trace.get("plan", {}).get("generation", trace.get("details", {}).get("generation"))
                assert gen == row["generation"]
                summaries.append({"case": i, "policy": policy, "status": trace["status"], "matched": match,
                                  "tokens": gen["generated_tokens"] if gen else None,
                                  "generation_seconds": gen["seconds"] if gen else None})
                pair.append(trace)
            if i <= 4:
                assert pair[0]["results"] == pair[1]["results"]
                assert pair[0]["plan"]["generation"]["prompt_sha256"] == pair[1]["plan"]["generation"]["prompt_sha256"]
                for result in pair[0]["results"]:
                    assert result["status"] == "ok"
                    assert result["citations"] == [s["id"] for s in result["sources"]]
                    for source in result["sources"]:
                        fact = facts[source["id"]]
                        assert fact["verified"] is True
                        assert all(value == fact[key] for key, value in source.items())
                        checked_sources += 1
                    company, year = result["company"], result["year"]
                    def money(field, y=year):
                        f = facts[f"{company}-{y}-{field}"]
                        return Decimal(f["value"]) * scales[f["unit"]]
                    with localcontext() as ctx:
                        ctx.prec = 40
                        if result["metric"] == "gross_margin":
                            value = (money("revenue") - money("cost")) / money("revenue") * 100
                        elif result["metric"] == "debt_ratio":
                            value = money("liabilities") / money("assets") * 100
                        elif result["metric"] == "revenue":
                            value = money("revenue") / scales[result["unit"]]
                        elif result["metric"] == "parent_net_profit":
                            if result["transform"] == "value":
                                value = money("net_profit") / scales[result["unit"]]
                            else:
                                previous = money("net_profit", year - 1)
                                value = (money("net_profit") - previous) / abs(previous) * 100
                        else:
                            raise AssertionError("Unexpected audited metric")
                        assert value.quantize(Decimal(".0001")) == Decimal(result["value"])
                    checked_results += 1
        totals = {}
        for policy in ("baseline", "budgeted"):
            for count in (4, 5):
                rows = [r for r in summaries if r["policy"] == policy and r["case"] <= count]
                seconds = sum(r["generation_seconds"] for r in rows)
                tokens = sum(r["tokens"] for r in rows)
                totals[f"{policy}_{count}_model_cases"] = {
                    "tokens": tokens, "seconds": round(seconds, 3), "tokens_per_generation_second": round(tokens / seconds, 3)}
        return {"archive": archive.name, "sha256": digest, "rows": summaries,
                "identical_paired_financial_results": checked_results, "verified_source_occurrences": checked_sources,
                "totals": totals,
                "limitations": ["Single-seed validation-company diagnostics, not a blind accuracy benchmark.",
                    "Case 6 rejects before inference. Case 5 fails intended clarification in both policies.",
                    "Generation throughput includes prefill; budgeted path also includes a second prefill.",
                    "Source verification uses the local audited snapshot; PDFs were not reopened."]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--facts", type=Path, default=Path(__file__).resolve().parents[1] / "data/verified/facts.json")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.archive, args.facts)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
