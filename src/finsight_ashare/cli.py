"""Thin command entrypoint. doctor works without pip installs."""
import argparse
import json
import os
import sys
import time

from .common import ROOT, companies, config, read_json, write_json


# 模块：命令入口；作用：按阶段调度，不把启动当作完成；核心业务：否。
def main(argv=None):
    parser = argparse.ArgumentParser(prog="finsight")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "setup", "_download"):
        sub.add_parser(name).add_argument("--offline", action="store_true")
    sub.add_parser("export")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--stage", choices=["fetch", "ingest", "audit", "dataset", "all"], default="all")
    prepare.add_argument("--only", help="One report ID, e.g. 600519-2024")
    prepare.add_argument("--pilot", action="store_true", help="Build 10 train-company cases, not a full training dataset")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--probe", action="store_true", help="20 optimizer steps and adapter save/load check")
    train_parser.add_argument("--pilot", action="store_true", help="Use 10 audited pilot cases; only allowed with --probe")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluation.add_argument("--modes", nargs="+", choices=["direct", "base", "tuned"], default=["direct", "base", "tuned"])
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--search", choices=["bge", "lexical"], default="bge")
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--replay", action="store_true", help="Deterministic tool pipeline only, without an LLM")
    smoke_parser.add_argument("--count", type=int, default=1)
    smoke_parser.add_argument("--search", choices=["bge", "lexical"], default="bge")
    chat = sub.add_parser("chat")
    chat.add_argument("--adapter", help="Local adapter directory relative to project root")
    chat.add_argument("--search", choices=["bge", "lexical"], default="bge")
    pack = sub.add_parser("package")
    pack.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if getattr(args, "offline", False):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    try:
        if args.command == "doctor":
            from .server import doctor
            return 0 if doctor(offline=args.offline)["ready_for_candidate_setup"] else 2
        if args.command == "setup":
            from .server import setup
            setup(offline=args.offline)
        elif args.command == "_download":
            from .server import download_models
            download_models(offline=args.offline)
        elif args.command == "prepare":
            from . import data, dataset
            stages = ["fetch", "ingest", "audit", "dataset"] if args.stage == "all" else [args.stage]
            for stage in stages:
                print("STAGE=" + stage, flush=True)
                if stage == "fetch":
                    data.fetch(only=args.only)
                elif stage == "ingest":
                    data.ingest()
                elif stage == "audit":
                    data.audit()
                else:
                    dataset.build(pilot=args.pilot)
        elif args.command == "train":
            from .server import train
            train(probe=args.probe, pilot=args.pilot)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            if args.limit is not None and args.limit <= 0:
                raise ValueError("--limit must be positive")
            evaluate(split=args.split, modes=args.modes, limit=args.limit, backend=args.search)
        elif args.command == "smoke":
            smoke(replay_only=args.replay, count=args.count, backend=args.search)
        elif args.command == "chat":
            interactive(adapter=args.adapter, backend=args.search)
        elif args.command == "export":
            from .packaging import export_results
            export_results()
        elif args.command == "package":
            from .packaging import build_release
            build_release(args.output)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


def bootstrap_example(root=ROOT):
    """Audited seed evidence is rechecked against the original PDF on each machine."""
    from . import data
    import csv
    seed = read_json(root / "examples/maotai_2024/seed.json")
    rows = data.manifest(root)
    target = next(r for r in rows if r["id"] == seed["report"]["id"])
    if target["url"] and target["url"] != seed["report"]["url"]:
        raise ValueError("Existing report selection differs from seed")
    target.update(seed["report"])
    write_json(root / "data/manifest.json", rows)
    data.fetch(root, only=target["id"])
    if next(r for r in data.manifest(root) if r["id"] == target["id"])["sha256"] != seed["report"]["sha256"]:
        raise ValueError("Official seed PDF changed; re-review it before using seed facts")
    data.ingest(root)
    audit_path = root / "data/audit.csv"
    with audit_path.open(encoding="utf-8-sig", newline="") as stream:
        audit_rows = list(csv.DictReader(stream))
    seed_keys = {(r["report_id"], str(r["year"]), r["metric"]) for r in seed["facts"]}
    preserved = [r for r in audit_rows if (r["report_id"], str(r["year"]), r["metric"]) not in seed_keys or r["status"] == "approved"]
    approved_keys = {(r["report_id"], str(r["year"]), r["metric"]) for r in preserved}
    additions = [r for r in seed["facts"] if (r["report_id"], str(r["year"]), r["metric"]) not in approved_keys]
    with audit_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=data.FIELDS)
        writer.writeheader()
        writer.writerows(preserved + additions)
    data.audit(root)


def smoke(root=ROOT, replay_only=False, count=1, backend="bge"):
    from .agent import ReportAgent
    from .dataset import make_tasks, stratified, replay
    from .evaluation import render_report, score
    from .tools import FinancialTools
    if not 1 <= count <= 10:
        raise ValueError("--count must be 1..10")
    if not (root / "data/verified/facts.json").exists():
        bootstrap_example(root)
    tools = FinancialTools.load(root, allowed_companies=[c for c, x in companies(root).items() if x["split"] == "train"],
                                search_backend="lexical" if replay_only else backend)
    tasks = stratified([t for t in make_tasks(tools, root) if t["split"] == "train"], count, config(root)["seed"])
    if len(tasks) != count:
        raise ValueError("Not enough audited smoke tasks")
    model = None
    if not replay_only:
        from .models import LocalModel
        model = LocalModel(root)
    results = []
    for i, task in enumerate(tasks):
        print(f"SMOKE {i+1}/{count}: {task['question']}", flush=True)
        trace = replay(task, tools) if replay_only else ReportAgent(model, tools).run(task["question"])
        row = {"id": task["id"], "mode": "scripted_tool_replay" if replay_only else "base_agent",
               "question": task["question"], "trace": trace, "score": score(task, trace)}
        print(json.dumps(trace["final"], ensure_ascii=False, indent=2), flush=True)
        results.append(row)
    output = root / "outputs/smoke" / (time.strftime("%Y%m%d-%H%M%S") + ("-replay" if replay_only else "-llm"))
    write_json(output / "cases.json", results)
    render_report(output / "report.html", results)
    print(f"SMOKE_OUTPUT={output}\nLLM_EXECUTED={not replay_only}", flush=True)
    if not all(row["score"]["task_correct"] for row in results):
        raise ValueError("Some smoke answers failed evidence/numeric checks; inspect the saved traces")


def interactive(root=ROOT, adapter=None, backend="bge"):
    from .agent import ReportAgent
    from .models import LocalModel
    from .tools import FinancialTools
    from .evaluation import render_report
    model = LocalModel(root, root / adapter if adapter else None)
    tools = FinancialTools.load(root, search_backend=backend)
    while True:
        print("输入财报问题（输入 exit 结束）：", flush=True)
        try:
            question = input().strip()
        except EOFError:
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        trace = ReportAgent(model, tools).run(question)
        print(json.dumps(trace["final"] or {"status": trace["status"]}, ensure_ascii=False, indent=2), flush=True)
        output = root / "outputs/chat" / time.strftime("%Y%m%d-%H%M%S")
        write_json(output / "trace.json", trace)
        render_report(output / "report.html", [{"question": question, "trace": trace}])
        print(f"REPORT={output / 'report.html'}", flush=True)
