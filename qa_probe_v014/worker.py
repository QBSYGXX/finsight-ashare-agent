"""Run a fixed diagnostic suite using one resident thinking-enabled engine."""
import argparse
import html
import json
import time
import traceback
from support import ADDON, ROOT, save, verify
from cases import CASES
from scoring import assess, summarize


def report(output, rows, summary):
    text = "<!doctype html><meta charset='utf-8'><title>FinSight 多题诊断</title><style>body{font:15px/1.6 sans-serif;max-width:1200px;margin:30px auto}td,th{padding:8px;border:1px solid #ddd;text-align:left}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>"
    text += "<h1>FinSight 多题诊断</h1><p>验证公司回归测试；程序预检与模型调用分开统计。不是独立测试集准确率。</p>"
    text += "<pre>" + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + "</pre><table><tr><th>题号</th><th>问题</th><th>模型调用</th><th>状态</th><th>通过</th><th>耗时</th><th>失败项</th></tr>"
    for r in rows:
        values = (r["id"], r["question"], r["model_calls"], r["status"], r["assessment"]["passed"], r["seconds"], ", ".join(r["assessment"]["failed_checks"]))
        text += "<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in values) + "</tr>"
    text += "</table><p>每题的trace.json、model_calls.json和assessment.json保存在对应题号目录。</p>"
    (output / "report.html").write_text(text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=["vllm", "vllm-mtp"], default="vllm-mtp")
    p.add_argument("--ids", default="")
    args = p.parse_args()
    from pathlib import Path
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    verify()
    from chat import verify_inputs, execute_question
    from engine import MetricEngine
    from resolver import CompanyResolver
    from adapter import Engine
    facts, _ = verify_inputs()
    references = json.loads((ADDON / "references.json").read_text(encoding="utf-8"))
    engine = MetricEngine(facts)
    resolver = CompanyResolver.load(ROOT, ROOT / "evaluation_v07")
    chosen = set(args.ids.split(",")) if args.ids else {c["id"] for c in CASES}
    if chosen - {c["id"] for c in CASES}:
        raise ValueError("Unknown case IDs: " + str(chosen))
    selected = [c for c in CASES if c["id"] in chosen]
    rows = []

    class Recorder:
        engine = None
        calls = None
        last_generation = None
        def generate(self, messages):
            if self.engine is None:
                self.engine = Engine(args.backend)
                save(output / "model.json", {"backend": args.backend, "load_seconds": self.engine.load_seconds,
                                             "thinking": True, "max_new_tokens": 8192})
            started = time.perf_counter()
            response = self.engine.generate(messages)
            self.last_generation = self.engine.last_generation
            self.calls.append({"messages": messages, "response": response, "generation": self.last_generation,
                               "call_seconds": time.perf_counter()-started})
            return response

    model = Recorder()
    try:
        for i, case in enumerate(selected):
            print(f"CASE_START={case['id']} INDEX={i+1}/{len(selected)} QUESTION={case['question']}", flush=True)
            model.calls = []
            model.last_generation = None
            started = time.perf_counter()
            try:
                result = execute_question(case["question"], engine, resolver, "model", model)
            except Exception as error:
                save(output / case["id"] / "error.json", {"error": repr(error), "traceback": traceback.format_exc()})
                raise
            finally:
                save(output / case["id"] / "model_calls.json", model.calls)
            assessment = assess(case, references[case["id"]], result, len(model.calls))
            save(output / case["id"] / "trace.json", result)
            save(output / case["id"] / "assessment.json", {"case": case, "reference": references[case["id"]], **assessment})
            row = {"id": case["id"], "category": case["category"], "question": case["question"], "status": result["status"],
                   "expected_kind": case["expect"]["kind"], "expected_status": references[case["id"]].get("status", case["expect"].get("statuses")),
                   "model_calls": len(model.calls), "seconds": round(time.perf_counter()-started, 3),
                   "generation": model.last_generation, "assessment": assessment}
            rows.append(row)
            save(output / "cases.json", rows)
            summary = summarize(rows, len(selected))
            save(output / "summary.json", summary)
            report(output, rows, summary)
            print("CASE_END=" + json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        summary = summarize(rows, len(selected))
        save(output / "summary.json", summary)
        report(output, rows, summary)
        if model.engine is not None and hasattr(model.engine, "llm"):
            try:
                save(output / "engine_metrics.json", [str(m) for m in model.engine.llm.get_metrics()])
            except Exception as error:
                save(output / "engine_metrics.json", {"unavailable": repr(error)})
        print("SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
