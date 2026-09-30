"""Executable teacher trajectories, company splits and frozen evaluation sets."""
import json
import random
from collections import Counter
from decimal import Decimal

from .agent import ReportAgent
from .common import ROOT, companies, config, digest, file_hash, read_json, source_digest, write_json
from .protocol import SYSTEM, action
from .tools import FinancialTools, METRICS


class ReplayModel:
    """Offline plumbing verification only. This is not a trained model."""
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, messages):
        return next(self.responses)


# 模块：任务生成；作用：从真实已审计表格构建可执行问题与数值答案；核心业务：是。
def make_tasks(tools, root=ROOT, omissions=None):
    tasks = []
    for code, company in companies(root).items():
        facts = {(f["year"], f["metric"]): f for f in tools.facts.values() if f["company"] == code}
        for (year, metric), fact in sorted(facts.items()):
            operations = [("convert", [fact], unit, f"{year}年{METRICS[metric]}是多少？以{unit}表示。") for unit in ("元", "万元", "亿元")]
            previous = facts.get((year-1, metric))
            if previous:
                operations.append(("difference", [fact, previous], "亿元", f"{year}年{METRICS[metric]}较{year-1}年增减多少亿元？"))
                if Decimal(previous["value"]) > 0:
                    operations.append(("growth", [fact, previous], "%", f"{year}年{METRICS[metric]}同比增长率是多少？"))
            ratio_specs = {"revenue": [("cost", "gross_margin", "毛利率")],
                           "liabilities": [("assets", "ratio", "资产负债率")],
                           "operating_cashflow": [("net_profit", "ratio", "经营现金流净额占归母净利润的百分比")],
                           "net_profit": [("revenue", "ratio", "归母净利润占营业收入的百分比（非总净利率）")]}
            for other, op, title in ratio_specs.get(metric, []):
                other_fact = facts.get((year, other))
                if other_fact and Decimal(other_fact["value"]) > 0:
                    operations.append((op, [fact, other_fact], "%", f"{year}年{title}是多少？"))
            for operation, operands, unit, query in operations:
                args = {"operation": operation, "fact_ids": [f["id"] for f in operands], "unit": unit}
                try:
                    result = tools.calculate(**args)
                except ValueError as error:
                    if omissions is not None:
                        omissions.append({"company": code, "year": year, "metric": metric,
                                          "arguments": args, "reason": str(error)})
                    continue
                question = f"{company['name']}（{code}）合并报表口径：{query}请给出依据和PDF物理页码。"
                calls = [action("report_search", query=METRICS[metric], company=code, year=year, top_k=1)]
                calls += [action("table_read", company=code, year=f["year"], metric=f["metric"]) for f in operands]
                calls += [action("calculate", **args)]
                source_text = "；".join(f"{f['report_year']}年年报PDF第{f['page']}页，{f['year']}年度{METRICS[f['metric']]} {f['value']}{f['unit']}" for f in operands)
                label = {"convert": METRICS[metric], "growth": METRICS[metric]+"同比增长率", "difference": METRICS[metric]+"同比增减额",
                         "gross_margin": "毛利率", "ratio": {"liabilities": "资产负债率", "operating_cashflow": "经营现金流净额/归母净利润", "net_profit": "归母净利润/营业收入"}.get(metric, "比率")}[operation]
                answer = {"status": "ok", "company": code, "year": year, "scope": "consolidated",
                          "value": result["value"], "unit": result["unit"], "citations": result["citations"],
                          "answer": f"{company['name']}{year}年合并口径{label}为{result['value']}{result['unit']}。依据：{source_text}。"}
                tasks.append({"id": digest([code, year, metric, args])[:20], "company": code, "year": year,
                              "split": company["split"], "kind": operation, "question": question,
                              "expected": answer, "responses": calls + ["<final_result>" + json.dumps(answer, ensure_ascii=False) + "</final_result>"]})
        # Missing-scope questions are grounded in the snapshot, not claims about the real world.
        for year in sorted({k[0] for k in facts}):
            for metric in ("revenue", "net_profit", "operating_cashflow", "assets", "liabilities", "cost"):
                if (year, metric) not in facts:
                    continue
                question = f"{company['name']}（{code}）{year}年母公司单体口径{METRICS[metric]}是多少？当前工具仅收录核验后的合并表，证据不足请明确说明。"
                answer = {"status": "insufficient_evidence", "company": code, "year": year, "scope": "parent",
                          "value": None, "unit": "亿元", "citations": [], "answer": f"当前快照没有{company['name']}{year}年母公司单体口径{METRICS[metric]}的已核验数据，不能用合并报表数字替代。"}
                tasks.append({"id": digest([code, year, metric, "missing_parent"])[:20], "company": code, "year": year,
                              "split": company["split"], "kind": "missing_scope", "question": question, "expected": answer,
                              "responses": [action("table_read", company=code, year=year, metric=metric, scope="parent"),
                                            "<final_result>" + json.dumps(answer, ensure_ascii=False) + "</final_result>"]})
    return tasks


def stratified(tasks, size, seed):
    groups = {}
    for task in tasks:
        groups.setdefault((task["company"], task["kind"]), []).append(task)
    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    result = []
    while len(result) < min(size, len(tasks)):
        for key in sorted(groups):
            if groups[key] and len(result) < size:
                result.append(groups[key].pop())
    return result


def replay(task, tools):
    trace = ReportAgent(ReplayModel(task["responses"]), tools).run(task["question"])
    if trace["status"] != "completed" or trace["final"] != task["expected"] or any(e.get("error") for e in trace["events"]):
        raise ValueError("Teacher trajectory replay failed: " + task["id"])
    return trace


# 模块：数据冻结；作用：公司隔离、真实执行、长度筛选、锁定评测；核心业务：是。
def build(root=ROOT, pilot=False):
    options = config(root)
    tools = FinancialTools.load(root)
    if not pilot:
        expected = {(c, y, m) for c in companies(root) for y in options["years"] for m in METRICS}
        observed = {(f["company"], f["year"], f["metric"]) for f in tools.facts.values()}
        missing = sorted(expected - observed)
        if missing:
            write_json(root / "data/processed/missing_facts.json", missing)
            raise ValueError(f"Missing {len(missing)} audited company/year/metric cells. See data/processed/missing_facts.json; use --pilot for the first 10 cases.")
    omissions = []
    tasks = make_tasks(tools, root, omissions=omissions)
    destination = root / ("data/pilot" if pilot else "data/frozen")
    if (destination / "manifest.json").exists():
        frozen = read_json(destination / "manifest.json")
        if frozen["facts_digest"] != digest(list(tools.facts.values())):
            raise ValueError("Facts changed since freeze; preserve the old experiment and explicitly choose a new data version")
        print("Frozen dataset already exists; verifying checksums.", flush=True)
        validate_freeze(root, pilot=pilot)
        return frozen
    selected = {split: stratified([t for t in tasks if t["split"] == split],
                 (10 if pilot else options["train_target"]) if split == "train" else options["evaluation_per_split"], options["seed"])
                for split in (["train"] if pilot else ["train", "validation", "test"])}
    if pilot and len(selected["train"]) < 10:
        raise ValueError("At least 10 executable train-company pilot questions are required")
    if not pilot and any(len(selected[s]) != options["evaluation_per_split"] for s in ("validation", "test")):
        raise ValueError("Not enough distinct evaluation questions; refusing to pad to 100")
    tokenizer = template = None
    if not pilot:
        from transformers import AutoTokenizer
        from llamafactory.data.template import TEMPLATES
        tokenizer = AutoTokenizer.from_pretrained(str(root / "models/Qwen3.5-9B"), local_files_only=True)
        template = TEMPLATES["qwen3_5_nothink"]
        template.fix_special_tokens(tokenizer)
    training, rejected, token_lengths = [], [], []
    for i, task in enumerate(selected["train"]):
        trace = replay(task, tools)
        messages = trace["messages"]
        # Tool observations deliberately use user turns, matching FinSight; only assistant turns have loss.
        if tokenizer is not None:
            pairs = template.encode_multiturn(tokenizer, messages[1:], system=SYSTEM)
            length = sum(len(a) + len(b) for a, b in pairs)
            token_lengths.append({"id": task["id"], "tokens": length})
            if length > options["context_length"]:
                rejected.append({"id": task["id"], "reason": "over_context", "tokens": length})
                continue
        training.append({"id": task["id"], "system": SYSTEM,
                         "conversations": [{"from": "human" if m["role"] == "user" else "gpt", "value": m["content"]} for m in messages[1:]]})
        if i % 50 == 0:
            print(f"Trajectory replay: {i+1}/{len(selected['train'])}", flush=True)
    if not training:
        raise ValueError("No training trajectories survived validation")
    for split, items in selected.items():
        write_json(destination / (split + "_tasks.json"), items)
    write_json(destination / "train.json", training)
    write_json(destination / "dataset_info.json", {"ashare_train": {"file_name": "train.json", "formatting": "sharegpt",
                "columns": {"messages": "conversations", "system": "system"}, "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}}})
    write_json(destination / "rejected.json", rejected)
    write_json(destination / "omissions.json", omissions)
    write_json(destination / "token_lengths.json", token_lengths)
    artifacts = ["train.json", "dataset_info.json", "rejected.json", "omissions.json", "token_lengths.json"] + [s + "_tasks.json" for s in selected]
    receipt = {"facts_digest": digest(list(tools.facts.values())), "config_digest": digest(options), "pilot": pilot,
               "source_digest": source_digest(root),
               "pages_sha256": file_hash(root / "data/processed/pages.jsonl"),
               "usable_training": len(training), "rejected": len(rejected), "target": options["train_target"],
               "max_selected_tokens": max((x["tokens"] for x in token_lengths), default=None),
               "generator": "deterministic executable templates; no human-authored or teacher-LLM claim",
               "counts": {s: dict(Counter(t["kind"] for t in items)) for s, items in selected.items()},
               "artifacts": {name: digest(read_json(destination / name)) for name in artifacts}}
    write_json(destination / "manifest.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return receipt


def validate_freeze(root=ROOT, pilot=False):
    path = root / ("data/pilot" if pilot else "data/frozen")
    receipt = read_json(path / "manifest.json")
    if receipt["facts_digest"] != digest(read_json(root / "data/verified/facts.json")):
        raise ValueError("Audited fact snapshot changed")
    if receipt["config_digest"] != digest(config(root)):
        raise ValueError("Project configuration changed after dataset freeze")
    if receipt["source_digest"] != source_digest(root):
        raise ValueError("Source code changed after dataset freeze; preserve this experiment and create a new dataset version")
    if receipt["pages_sha256"] != file_hash(root / "data/processed/pages.jsonl"):
        raise ValueError("Retrieval page snapshot changed after dataset freeze")
    for name, expected in receipt["artifacts"].items():
        if digest(read_json(path / name)) != expected:
            raise ValueError("Frozen artifact changed: " + name)
    split_map = companies(root)
    for split in (["train"] if pilot else ["train", "validation", "test"]):
        for task in read_json(path / (split + "_tasks.json")):
            if split_map[task["company"]]["split"] != split:
                raise ValueError("Company leakage detected")
    return receipt
