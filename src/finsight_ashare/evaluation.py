"""Deterministic scoring with inspectable per-question artifacts."""
import gc
import html
import json
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .agent import ReportAgent
from .common import ROOT, companies, config, digest, file_hash, now, read_json, read_jsonl, source_digest, write_json
from .dataset import validate_freeze
from .tools import FinancialTools


# 模块：评测；作用：按数值、单位、口径、证据逐项判分；核心业务：是。
def score(task, trace):
    expected = task["expected"]
    final = trace.get("final") or {}
    citations = set(final.get("citations", []))
    gold = set(expected["citations"])
    seen = set(trace.get("seen_fact_ids", []))
    correct_evidence = citations & gold & seen
    citation_precision = len(correct_evidence) / len(citations) if citations else (1.0 if not gold else 0.0)
    citation_recall = len(correct_evidence) / len(gold) if gold else (1.0 if not citations else 0.0)
    scope_ok = all(final.get(k) == expected[k] for k in ("company", "year", "scope"))
    value_ok = False
    if expected["status"] == "insufficient_evidence":
        value_ok = final.get("status") == "insufficient_evidence" and final.get("value") is None
    elif final.get("status") == "ok" and final.get("unit") == expected["unit"]:
        try:
            predicted = Decimal(str(final.get("value")))
            value_ok = predicted.is_finite() and abs(predicted - Decimal(expected["value"])) <= Decimal("0.00015")
        except (InvalidOperation, TypeError):
            pass
    calls = [e for e in trace.get("events", []) if e.get("tool_success") is not None]
    complete = trace.get("status") == "completed"
    return {"task_correct": bool(complete and scope_ok and value_ok and citation_precision == citation_recall == 1.0),
            "numeric_unit_correct": bool(complete and scope_ok and value_ok),
            "citation_precision": citation_precision, "citation_recall": citation_recall,
            "tool_calls": len(calls), "successful_tool_calls": sum(e["tool_success"] is True for e in calls),
            "seconds": trace["seconds"]}


def aggregate(rows):
    count = len(rows)
    if not count:
        raise ValueError("No evaluation results")
    scores = [r["score"] for r in rows]
    numeric = [r["score"] for r in rows if r["expected"]["status"] == "ok"]
    missing = [r["score"] for r in rows if r["expected"]["status"] == "insufficient_evidence"]
    calls = sum(s["tool_calls"] for s in scores)
    return {"questions": count, "task_accuracy": sum(s["task_correct"] for s in scores) / count,
            "numeric_questions": len(numeric),
            "numeric_unit_accuracy": sum(s["numeric_unit_correct"] for s in numeric) / len(numeric) if numeric else None,
            "abstention_accuracy": sum(s["task_correct"] for s in missing) / len(missing) if missing else None,
            "citation_precision": sum(s["citation_precision"] for s in numeric) / len(numeric) if numeric else None,
            "citation_recall": sum(s["citation_recall"] for s in numeric) / len(numeric) if numeric else None,
            "tool_success_rate": sum(s["successful_tool_calls"] for s in scores) / calls if calls else None,
            "tool_calls": calls, "mean_seconds": sum(s["seconds"] for s in scores) / count}


def adapter_files(adapter):
    required = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    if not all(p.exists() for p in required):
        raise ValueError("A saved LoRA adapter is required")
    return {p.name: file_hash(p) for p in required}


def evaluate(root=ROOT, split="validation", modes=("direct", "base", "tuned"), limit=None, backend="bge"):
    from .models import LocalModel
    receipt = validate_freeze(root)
    tasks = read_json(root / "data/frozen" / (split + "_tasks.json"))
    if limit:
        tasks = tasks[:limit]
    training = None
    if "tuned" in modes:
        training = read_json(root / "outputs/training/production.json")
        if training.get("status") != "completed" or training["dataset_digest"] != digest(receipt):
            raise ValueError("Completed full training on this frozen dataset is required")
        if training["adapter_hashes"] != adapter_files(root / training["adapter"]):
            raise ValueError("Adapter changed after successful training")
    model_lock = read_json(root / "models/snapshots.json")
    identity = {"dataset": digest(receipt), "models": model_lock, "split": split, "modes": list(modes),
                "source_digest": source_digest(root),
                "limit": limit, "backend": backend, "adapter": training["adapter_hashes"] if training else None,
                "protocol_version": "0.1.0"}
    output = root / "outputs/evaluation" / digest(identity)[:16]
    write_json(output / "run.json", {**identity, "status": "running", "started": now(), "note": "Latency excludes model load; includes retrieval and tools. Auto-scoring covers structured fields; prose needs manual review."})
    all_rows, summaries = [], {}
    model = None
    for mode in modes:
        previous_mode = getattr(model, "evaluation_mode", None)
        if model is None or (mode == "tuned") != (previous_mode == "tuned"):
            if model is not None:
                del model
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            model = LocalModel(root, root / training["adapter"] if mode == "tuned" else None)
            model.evaluation_mode = mode
        allowed = [c for c, item in companies(root).items() if item["split"] == split]
        tools = FinancialTools.load(root, allowed_companies=allowed, search_backend=backend)
        path = output / (mode + ".jsonl")
        rows = read_jsonl(path) if path.exists() else []
        done = {r["id"] for r in rows}
        if len(done) != len(rows) or not done <= {t["id"] for t in tasks}:
            raise ValueError("Evaluation resume file does not match this run")
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            for index, task in enumerate(tasks):
                if task["id"] in done:
                    continue
                print(f"[{mode}/{split}] {index+1}/{len(tasks)} {task['id']}", flush=True)
                started = time.perf_counter()
                context = tools.report_search(task["question"], task["company"], top_k=5) if mode == "direct" else None
                try:
                    trace = ReportAgent(model, tools, config(root)["max_steps"]).run(task["question"], direct_context=context)
                except ValueError as error:
                    trace = {"status": "model_input_error", "error": str(error), "final": None, "events": [], "seen_fact_ids": [], "seconds": 0}
                trace["seconds"] = round(time.perf_counter() - started, 3)
                row = {"id": task["id"], "question": task["question"], "kind": task["kind"], "mode": mode,
                       "expected": task["expected"], "trace": trace, "score": score(task, trace)}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                rows.append(row)
        summaries[mode] = aggregate(rows)
        all_rows.extend(rows)
        write_json(output / "summary.json", summaries)
    write_json(output / "failures.json", [r for r in all_rows if not r["score"]["task_correct"]])
    render_report(output / "report.html", all_rows, summaries)
    write_json(output / "run.json", {**identity, "status": "completed", "finished": now(), "limited_run": limit is not None,
                                    "questions_per_mode": len(tasks), "metrics_note": "Structured answer scoring; manual prose review still required"})
    print(f"Completed evaluation: {output}", flush=True)
    return output


# 模块：报告导出；作用：生成离线可读HTML与完整工具记录；核心业务：否。
def render_report(path, rows, summaries=None):
    chunks = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
              "<title>FinSight A-share experiment</title><style>body{font:15px/1.65 system-ui,sans-serif;max-width:1080px;margin:32px auto;padding:0 20px;color:#222}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f6;padding:14px}details{border-top:1px solid #ccc;padding:12px 0}summary{cursor:pointer}h1{font-size:25px}h2{font-size:19px}</style>",
              "<h1>FinSight A股财报实验记录</h1>", "<p>自动评分核对结构化数字、单位、口径和证据ID；自由文本分析仍需人工复核。</p>"]
    if summaries:
        chunks.append("<h2>评测汇总</h2><pre>" + html.escape(json.dumps(summaries, ensure_ascii=False, indent=2)) + "</pre>")
    for row in rows:
        chunks.append("<details><summary>" + html.escape(str(row.get("mode", "trace")) + " | " + row.get("question", "")) + "</summary><pre>" + html.escape(json.dumps(row, ensure_ascii=False, indent=2)) + "</pre></details>")
    chunks.append("</html>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(chunks), encoding="utf-8")
