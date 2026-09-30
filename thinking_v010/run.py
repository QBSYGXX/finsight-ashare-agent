"""Offline bounded-thinking diagnostics. Original financial planner is pinned verbatim."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import threading
import traceback
import zipfile

ADDON = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FINSIGHT_ROOT", ADDON.parent)).resolve()
sys.path[:0] = [str(ADDON / "metrics"), str(ROOT / "src"), str(ROOT / "evaluation_v07"), str(ROOT / "chat_v071")]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def verify_package():
    manifest = json.loads((ADDON / "PACKAGE_MANIFEST.json").read_text())
    for name, checksum in manifest["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ADDON) or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError("Package file changed: " + name)
    return manifest


# Validation-company diagnostics, not an independent accuracy benchmark.
CASES = [
    {"question": "中兴通讯2024年毛利率、资产负债率", "requests": [["gross_margin", "value"], ["debt_ratio", "value"]]},
    {"question": "美的集团2024年营业收入", "requests": [["revenue", "value"]]},
    {"question": "中兴通讯2024年归母净利润及其同比增长率", "requests": [["parent_net_profit", "value"], ["parent_net_profit", "yoy_growth"]]},
    {"question": "美的集团可查询到的历年营业收入", "requests": [["revenue", "value"]]},
    {"question": "中兴通讯2024年毛利率同比", "status": "needs_clarification"},
    {"question": "中兴通讯2024年一季度归母净利润", "status": "unsupported_period"},
]


def assess(result, case):
    if "requests" in case:
        actual = sorted([r["metric"], r["transform"]] for r in result.get("plan", {}).get("requests", []))
        return actual == sorted(case["requests"]) and result["status"] in {"completed", "partial", "no_results"}
    # Preflight's exact code for unsupported periods is set in the pinned source.
    return result["status"] == case["status"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "chat", "evaluate"], nargs="?", default="smoke")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--think-budget", type=int, default=1024)
    parser.add_argument("--answer-budget", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question", default=CASES[0]["question"])
    args = parser.parse_args()
    if not 64 <= args.think_budget <= 8192 or not 64 <= args.answer_budget <= 2048:
        parser.error("Thinking budget: 64..8192; answer budget: 64..2048")
    if not 0 <= args.seed < 2**32 or not 9216 <= args.context_length <= 32768:
        parser.error("Invalid seed or context length (9216..32768)")
    if args.think_budget + args.answer_budget + 64 >= args.context_length:
        parser.error("Budgets must leave room for prompt and closing marker")
    output = ROOT / "outputs/thinking_v010" / Path(args.run_dir).name
    code = 1
    finished = threading.Event()
    started = time.perf_counter()
    def progress():
        while not finished.wait(30):
            print(f"PROCESS_RUNNING elapsed_seconds={time.perf_counter()-started:.0f}", flush=True)
    monitor = threading.Thread(target=progress, daemon=True)
    monitor.start()
    print("STAGE=verify_package_and_import_runtime", flush=True)
    try:
        manifest = verify_package()
        from chat import execute_question, verify_inputs, write_report
        from engine import MetricEngine
        from resolver import CompanyResolver
        from budget_model import BudgetPlanner
        facts, _ = verify_inputs()
        engine = MetricEngine(facts)
        resolver = CompanyResolver.load(ROOT, ROOT / "evaluation_v07")
        save(output / "request.json", {**vars(args), "package_commit": manifest["commit"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "model": "base"})
        holder = [None]

        class Lazy:
            policy = "budgeted"

            @property
            def last_generation(self):
                return holder[0].last_generation if holder[0] else None

            def generate(self, messages):
                if holder[0] is None:
                    print("STAGE=load_model; base model, single visible GPU, original PyTorch kernels", flush=True)
                    holder[0] = BudgetPlanner(ROOT, think_budget=args.think_budget,
                        answer_budget=args.answer_budget, context_length=args.context_length, seed=args.seed)
                holder[0].policy = self.policy
                return holder[0].generate(messages)

        model = Lazy()
        rows = []
        def ask(question, label):
            result = execute_question(question, engine, resolver, "model", model)
            result.update(budget_protocol="thinking-v0.10", policy=model.policy)
            save(output / label / "trace.json", result)
            write_report(output / label / "report.html", result)
            print("RESULT=" + json.dumps({"question": question, "policy": model.policy,
                "status": result["status"], "message": result.get("message"),
                "results": [{k: r.get(k) for k in ("year", "name", "status", "value", "unit", "citations")}
                            for r in result["results"]]}, ensure_ascii=False), flush=True)
            print("TRACE=" + str(output / label / "trace.json"), flush=True)
            return result

        if args.mode == "evaluate":
            for index, case in enumerate(CASES):
                for policy in ("baseline", "budgeted"):
                    model.policy = policy
                    print(f"CASE={index+1}/{len(CASES)} POLICY={policy}", flush=True)
                    result = ask(case["question"], f"case-{index+1}-{policy}")
                    rows.append({"case": case, "policy": policy, "status": result["status"],
                        "plan_or_abstention_matches": assess(result, case), "seconds": result["seconds"],
                        "generation": result.get("plan", {}).get("generation", result.get("details", {}).get("generation"))})
            save(output / "comparison.json", {"rows": rows, "note": "Six validation-company diagnostics; plan matching is not financial accuracy; preflight-only cases are marked by absent generation."})
        elif args.mode == "smoke":
            result = ask(args.question, "smoke")
            if result["status"] != "completed":
                raise RuntimeError("Smoke question did not complete: " + result["status"])
        else:
            index = 0
            while True:
                try:
                    question = input("财务指标> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if question.lower() in {"exit", "quit"}:
                    break
                if question:
                    index += 1
                    ask(question, f"chat-{index}")
        code = 0
    except Exception as error:
        save(output / "error.json", {"error": repr(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
    finally:
        finished.set()
        monitor.join()
        save(output / "status.json", {"exit_code": code, "status": "completed" if code == 0 else "failed",
            "note": "Evaluation process completion is not acceptance; inspect comparison and individual traces."})
        print(f"STATUS={'completed' if code == 0 else 'failed'} EXIT={code}", flush=True)
        dest = ROOT / "exports" / ("finsight-thinking-" + output.name + ".zip")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output).as_posix())
            log = Path(args.run_dir) / "run.log"
            if log.exists():
                archive.write(log, "run.log")
        checksum = hashlib.sha256(dest.read_bytes()).hexdigest()
        dest.with_suffix(".zip.sha256").write_bytes(f"{checksum}  {dest.name}\n".encode())
        print(f"RESULTS={dest}\nSHA256={checksum}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
