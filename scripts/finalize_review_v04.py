"""Materialize the explicitly reviewed v0.4 snapshot, never approve new extraction."""
from collections import Counter
from contextlib import redirect_stdout
import csv
from decimal import Decimal
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight_ashare.common import file_hash, read_json, read_jsonl, source_digest, write_json
from finsight_ashare.data import FIELDS, audit
from finsight_ashare.dataset import make_tasks, replay, stratified
from finsight_ashare.tools import FinancialTools

REVIEWED_CANDIDATE_SHA256 = "cb95ef23047f0a311818e5eac3ac21f1d04a0a1508799047b491c233d19a8231"
VISUAL_SAMPLES = ["600519-2025", "000858-2023", "600690-2024", "000651-2024",
                  "300750-2025", "601012-2024", "600276-2025", "600887-2023",
                  "000333-2024", "000063-2025", "002594-2024", "300760-2024"]


def main():
    directory = ROOT / "data/review_v0.4"
    candidate_path = directory / "all-candidates.json"
    if file_hash(candidate_path) != REVIEWED_CANDIDATE_SHA256:
        raise ValueError("Candidate snapshot changed; new explicit review is required")
    pages = {(p["report_id"], p["page"]): p for p in read_jsonl(ROOT / "data/processed/pages.jsonl")}
    rows, selections = [], []
    for rid, metrics in read_json(candidate_path).items():
        for metric, candidates in metrics.items():
            # All 216 first/main-statement selections were inspected in this pinned snapshot.
            selected = candidates[0]
            header = {k: selected["header"][k] for k in ("page", "text")}
            columns = header
            unit_evidence = header
            if rid == "601012-2023" and metric in {"revenue", "cost", "net_profit"}:
                columns = {"page": 127, "text": "\n".join(pages[(rid, 127)]["text"].splitlines()[:9])}
                unit_evidence = columns
            if rid.startswith("002594"):
                unit_page = {2023: 153, 2024: 158, 2025: 140}[selected["year"]]
                line = next(s for s in pages[(rid, unit_page)]["text"].splitlines() if "均以人民币千元为单位表示" in s)
                unit_evidence = {"page": unit_page, "text": line}
            text = re.sub(r"\s+", "", unit_evidence["text"])
            match = re.search(r"(?:单位[:：]?(?:为)?(?:人民币)?|人民币)(千元|万元|亿元|元)", text)
            if not match:
                raise ValueError("Missing unit evidence: " + rid + " " + metric)
            current, prior = Decimal(selected["value"]), Decimal(selected["comparison_value"])
            normalization = "none"
            if rid.startswith("000333") and metric == "cost":
                assert current < 0 and prior < 0
                current, prior = -current, -prior
                normalization = "expense_magnitude"
            review = {"header": header, "unit_evidence": unit_evidence, "column_evidence": columns,
                      "columns": selected["header"]["columns"], "comparison_value": str(prior),
                      "normalization": normalization}
            import json
            row = {k: selected[k] for k in ("report_id", "year", "metric", "scope", "page", "evidence")}
            row.update(value=str(current), unit=match[1], status="approved",
                       reviewer="AI source-text review; image-sampled by report; v0.4 (not human review)",
                       statement_review=json.dumps(review, ensure_ascii=False))
            rows.append(row)
            selections.append({"report_id": rid, "metric": metric, "selected_candidate": 0,
                               "page": selected["page"], "candidate_count": len(candidates),
                               "image_reviewed": rid in VISUAL_SAMPLES})
    if len(rows) != 216:
        raise ValueError("Expected 216 reviewed facts")
    snapshot = directory / "audit.csv"
    with snapshot.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    # Preserve seed audit/facts before switching the active fact snapshot; never modify old runs.
    backup = ROOT / "outputs/data_backups/pre-v0.4"
    for name in ("data/audit.csv", "data/verified/facts.json", "data/verified/audit_receipt.json"):
        original, saved = ROOT / name, backup / name
        if original.exists() and not saved.exists():
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, saved)
    shutil.copy2(snapshot, ROOT / "data/audit.csv")
    facts = audit(ROOT)
    tools = FinancialTools.load(ROOT)
    omissions = []
    tasks = make_tasks(tools, ROOT, omissions=omissions)
    counts = Counter(t["split"] for t in tasks)
    log_path = ROOT / "outputs/data_review_v0.4/replay.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for i, task in enumerate(tasks):
            with redirect_stdout(log):
                replay(task, tools)
            if i % 100 == 0:
                log.flush()
                print(f"Reviewed task replay {i+1}/{len(tasks)}", flush=True)
    write_json(directory / "selections.json", selections)
    write_json(directory / "omissions.json", omissions)
    write_json(directory / "review_receipt.json", {
        "version": "0.4", "method": "AI text review of 216 selected source rows and statement headers; 12 report image samples; no human review claim",
        "visual_samples": VISUAL_SAMPLES, "candidate_sha256": REVIEWED_CANDIDATE_SHA256,
        "facts": len(facts), "reports": 36, "basis": "Current-year column in each year's own annual report",
        "cost_normalization": "Only Midea negative expense presentation becomes positive cost magnitude",
        "comparison_policy": "Exact normalized equality with current report's prior column; mismatched restatements AND rounding are excluded",
        "task_pool": dict(counts), "task_kinds": dict(Counter(t["kind"] for t in tasks)),
        "selected_before_tokenization": {s: len(stratified([t for t in tasks if t['split'] == s], 800 if s == 'train' else 100, 20260923)) for s in counts},
        "omitted_operations": len(omissions), "all_task_replays_passed": len(tasks),
        "source_digest": source_digest(ROOT),
        "replay_log_sha256": file_hash(log_path),
        "tokenization": "Pending server pinned qwen3_5_nothink tokenizer/template; no formal freeze yet",
        "files": {name: file_hash(ROOT / name) for name in ["data/audit.csv", "data/verified/facts.json", "data/manifest.json", "data/processed/pages.jsonl"]}})
    print(read_json(directory / "review_receipt.json"), flush=True)


if __name__ == "__main__":
    main()
