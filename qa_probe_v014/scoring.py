"""Explicit-intent regression scoring. No LLM judge, no self-generated answer key."""
import hashlib
import json


def canonical_rows(rows):
    normalized = []
    for row in rows:
        sources = sorted(row.get("sources", []), key=lambda s: s["id"])
        normalized.append({k: row.get(k) for k in (
            "company", "year", "scope", "metric", "transform", "status", "value", "unit")})
        normalized[-1].update(citations=sorted(row.get("citations", [])),
            source_sha256=hashlib.sha256(json.dumps(sources, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
    return sorted(normalized, key=lambda r: json.dumps(r, sort_keys=True))


def assess(case, reference, result, model_calls):
    expected = case["expect"]
    checks = {}
    if expected["kind"] == "boundary":
        checks["status"] = result.get("status") in expected["statuses"]
        checks["no_numeric_execution"] = not result.get("results") and not result.get("plan")
        if expected["must_not_call_model"]:
            checks["preflight_only"] = model_calls == 0
        if expected.get("reason"):
            checks["reason"] = result.get("details", {}).get("reason") == expected["reason"]
    else:
        plan = result.get("plan", {})
        checks["status"] = result.get("status") == reference["status"]
        checks["context"] = all(plan.get(k) == expected[k] for k in ("company", "years", "scope", "money_unit"))
        actual = plan.get("requests", [])
        checks["plan"] = sorted(json.dumps(r, sort_keys=True) for r in actual) == sorted(json.dumps(r, sort_keys=True) for r in expected["requests"])
        checks["values_units_and_sources"] = canonical_rows(result.get("results", [])) == reference["canonical_rows"]
        checks["model_used"] = model_calls > 0
    return {"passed": all(checks.values()), "checks": checks,
            "failed_checks": [k for k, value in checks.items() if not value]}


def summarize(rows, selected):
    groups = {}
    for row in rows:
        group = "model_called" if row["model_calls"] else "preflight_only"
        for key in (group, "category:" + row["category"]):
            count = groups.setdefault(key, {"executed": 0, "passed": 0})
            count["executed"] += 1
            count["passed"] += int(row["assessment"]["passed"])
    return {"selected": selected, "executed": len(rows), "passed": sum(r["assessment"]["passed"] for r in rows),
            "failed": sum(not r["assessment"]["passed"] for r in rows), "not_run": selected-len(rows),
            "groups": groups, "note": "Validation-company diagnostic regression, not independent accuracy. Preflight-only cases are reported separately. Tool references share the pinned financial engine."}
