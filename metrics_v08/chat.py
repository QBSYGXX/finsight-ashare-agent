"""Multi-metric annual QA: constrained plan, deterministic results, per-item evidence."""
import argparse
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get("FINSIGHT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evaluation_v07"), str(ROOT / "chat_v071")]
from engine import MetricEngine, MetricError, load_registry
from planner import model_plan, request_context, rules_plan
from resolver import CompanyResolver
from finsight_ashare.common import digest, read_json, source_digest, write_json


def verify_inputs():
    pin = read_json(Path(__file__).with_name("dependency_pin.json"))
    for name, checksum in pin["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Dependency changed: " + name)
    facts = read_json(ROOT / "data/verified/facts.json")
    if digest(facts) != pin["facts_digest"] or digest(read_json(ROOT / "configs/project.json")) != pin["config_digest"]:
        raise ValueError("Fact or company snapshot changed")
    if source_digest(ROOT) != pin["source_digest"]:
        raise ValueError("Frozen source changed")
    return facts, pin


def execute_question(question, engine, resolver, planner_mode="rules", model=None):
    started = time.perf_counter()
    try:
        context = request_context(question, resolver, engine.available_years)
        plan = rules_plan(context, engine.registry) if planner_mode == "rules" else model_plan(context, engine.registry, model)
        result = engine.execute_years(plan["company"], plan["years"], plan["scope"], plan["requests"], plan["money_unit"])
        result.update(plan=plan, planner=planner_mode)
    except MetricError as error:
        result = {"status": error.code, "message": str(error), "details": error.details, "results": [], "planner": planner_mode}
    return {"question": question, "protocol": "metrics-v0.8.5", **result, "seconds": round(time.perf_counter() - started, 3)}


def write_report(path, result):
    body = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>财务指标查询记录</title>"
    body += "<style>body{max-width:1100px;margin:32px auto;padding:0 16px;font:15px/1.7 sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style>"
    body += "<h1>财务指标查询记录</h1><p>" + html.escape(result["question"]) + "</p><table><tr><th>年度</th><th>指标</th><th>变换</th><th>结果</th></tr>"
    for row in result["results"]:
        value = row["value"] + row["unit"] if row["status"] == "ok" else row.get("message", row["status"])
        body += "<tr>" + "".join("<td>" + html.escape(str(x)) + "</td>" for x in (row["year"], row["name"], row["transform"], value)) + "</tr>"
    body += "</table><h2>定义、计划与来源</h2><pre>" + html.escape(json.dumps(result, ensure_ascii=False, indent=2)) + "</pre></html>"
    path.write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question")
    parser.add_argument("--planner", choices=["rules", "model"], default="rules")
    parser.add_argument("--model", choices=["base", "tuned"], default="base")
    parser.add_argument("--thinking", choices=["on", "off"], default="off")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--decoding", choices=["auto", "greedy", "sample"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--catalog", action="store_true")
    args = parser.parse_args()
    if args.planner != "model" and (args.thinking == "on" or args.max_new_tokens is not None or args.context_length is not None or args.decoding != "auto" or args.seed != 42):
        parser.error("Thinking and generation limits require --planner model")
    if args.max_new_tokens is not None and not 64 <= args.max_new_tokens <= 8192:
        parser.error("--max-new-tokens must be between 64 and 8192")
    if args.context_length is not None and not 4096 <= args.context_length <= 32768:
        parser.error("--context-length must be between 4096 and 32768")
    effective_output = args.max_new_tokens if args.max_new_tokens is not None else (2048 if args.thinking == "on" else 512)
    effective_context = args.context_length if args.context_length is not None else (8192 if args.thinking == "on" else 4096)
    if effective_output >= effective_context:
        parser.error("Output budget must leave context space for the prompt")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    registry = load_registry()
    if args.catalog:
        print(json.dumps(registry, ensure_ascii=False, indent=2))
        return
    facts, pin = verify_inputs()
    engine = MetricEngine(facts, registry)
    resolver = CompanyResolver.load(ROOT, ROOT / "evaluation_v07")

    class LazyPlanner:
        model = None

        @property
        def last_generation(self):
            return self.model.last_generation if self.model else None

        def generate(self, messages):
            if self.model is None:
                adapter = None
                if args.model == "tuned":
                    from finsight_ashare.evaluation import adapter_files
                    production = read_json(ROOT / "outputs/training/production.json")
                    if production["status"] != "completed" or production["dataset_digest"] != pin["frozen_digest"]:
                        raise ValueError("Matching completed training required")
                    adapter = ROOT / production["adapter"]
                    if adapter_files(adapter) != production["adapter_hashes"]:
                        raise ValueError("Adapter hashes changed")
                from planner_model import PlannerModel
                self.model = PlannerModel(ROOT, adapter, thinking=args.thinking == "on",
                                          max_new_tokens=effective_output, context_length=effective_context,
                                          decoding=args.decoding, seed=args.seed)
            return self.model.generate(messages)

    model = LazyPlanner()
    print("PLANNER=" + args.planner + "; 规则模式不调用大模型；模型模式只生成查询计划。", flush=True)
    if args.planner == "model":
        print(f"THINKING={args.thinking} MAX_NEW_TOKENS={effective_output} CONTEXT_LENGTH={effective_context}", flush=True)
    while True:
        try:
            question = args.question if args.question is not None else input("财务指标> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            if args.question is not None:
                break
            continue
        result = execute_question(question, engine, resolver, args.planner, model)
        if "plan" in result:
            p = result["plan"]
            print(f"口径：{p['company']} | 年度{','.join(map(str, p['years']))}（全年） | {p['scope']} | 金额单位{p['money_unit']}", flush=True)
            if p["year_selection"] == "all_available":
                print("年份范围来自该公司、该报表口径在当前事实库中的覆盖；不代表公司全部历史年份。", flush=True)
            print("PLAN=" + json.dumps(p["requests"], ensure_ascii=False), flush=True)
        for row in result["results"]:
            if row["status"] == "ok":
                sources = "; ".join(dict.fromkeys(f"{s['report_id']} PDF第{s['page']}页" for s in row["sources"]))
                print(f"{row['year']}年 | {row['name']} [{row['transform']}] = {row['value']}{row['unit']} | {sources}", flush=True)
            else:
                print(f"{row['year']}年 | {row['name']} [{row['transform']}]：{row['message']}；{json.dumps(row['details'], ensure_ascii=False)}", flush=True)
        if result.get("message"):
            print(result["message"], flush=True)
        if "expected_requests" in result.get("details", {}):
            print("PLAN_EXPECTED=" + json.dumps(result["details"]["expected_requests"], ensure_ascii=False), flush=True)
            print("PLAN_PROPOSED=" + json.dumps(result["details"]["proposed_requests"], ensure_ascii=False), flush=True)
        print("STATUS=" + result["status"], flush=True)
        destination = ROOT / "outputs/metrics_v085" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        write_json(destination / "trace.json", result)
        write_report(destination / "report.html", result)
        print("TRACE=" + str(destination / "trace.json"), flush=True)
        if args.question is not None:
            break


if __name__ == "__main__":
    main()
