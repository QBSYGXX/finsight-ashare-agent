"""Extract statement-scoped candidates for explicit review; never auto-approve."""
import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight_ashare.common import read_json, read_jsonl, write_json

TITLE = re.compile(r"^(?:\d{4}年度)?(?:[一二三四五六七八九十\d]+[、.])?(合并及公司|合并|母公司|公司)?(资产负债表|利润表|现金流量表|所有者权益变动表|股东权益变动表|综合收益表)(?:[（(]续[）)])?$")
NUM = re.compile(r"(?<![\d.])(?:\(-?\d[\d,]*(?:\.\d+)?\s*\)|-?\d[\d,]*(?:\.\d+)?)(?![\d.])")
LABELS = {
    "assets": ("资产负债表", ["资产总计", "资产总额"]),
    "liabilities": ("资产负债表", ["负债合计", "负债总计", "负债总额"]),
    "revenue": ("利润表", ["营业收入"]),
    "cost": ("利润表", ["营业成本"]),
    "net_profit": ("利润表", ["归属于母公司股东的净利润", "归属于母公司所有者的净利润", "归属于母公司普通股股东", "归属于母公司股东的净亏损", "归属于母公司所有者的净亏损", "归属于上市公司股东的净利润"]),
    "operating_cashflow": ("现金流量表", ["经营活动产生的现金流量净额", "经营活动产生/(使用)的现金流量净额"]),
}


def compact(text):
    return re.sub(r"\s+", "", text)


def numeric_tokens(text):
    found = []
    for match in NUM.finditer(text):
        token = match.group()
        value = Decimal(token.replace(",", "").replace(" ", "").replace("(", "-").replace(")", ""))
        if abs(value) >= 1000:
            found.append((token, str(value)))
    return found


def extract(pages):
    candidates = defaultdict(list)
    section = None
    header = None
    for page in pages:
        lines = page["text"].splitlines()
        for i, line in enumerate(lines):
            match = TITLE.fullmatch(compact(line))
            if match:
                scope, kind = match.groups()
                section = kind if scope in {"合并", "合并及公司"} else None
                header = {"page": page["page"], "title": line.strip(), "text": "\n".join(lines[i:i+15]),
                          "columns": 4 if scope == "合并及公司" else 2}
            if section not in {"资产负债表", "利润表", "现金流量表"}:
                continue
            for metric, (kind, aliases) in LABELS.items():
                if kind != section:
                    continue
                # Joined lines are necessary for wrapped Chinese labels, but stop after the first row with numbers.
                for count in range(1, 4):
                    snippet = "\n".join(lines[i:i+count])
                    text = compact(NUM.sub("", snippet))
                    if not any(a in text for a in aliases):
                        continue
                    if metric == "liabilities" and any(s in text for s in ["流动负债合计", "非流动负债合计"]):
                        continue
                    tokens = numeric_tokens(snippet)
                    if len(tokens) < header["columns"]:
                        continue
                    values = tokens[-header["columns"]:]
                    candidate = {"report_id": page["report_id"], "year": page["report_year"], "metric": metric,
                                 "page": page["page"], "evidence": snippet, "value": values[0][1],
                                 "comparison_value": values[1][1], "scope": "consolidated", "header": header,
                                 "tokens": values, "status": "pending"}
                    duplicate = next((x for x in candidates[metric] if x["page"] == candidate["page"] and x["value"] == candidate["value"]), None)
                    if duplicate is None:
                        candidates[metric].append(candidate)
                    elif len(candidate["evidence"]) < len(duplicate["evidence"]):
                        duplicate.update(candidate)
                    break
    return dict(candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    args = parser.parse_args()
    reports = defaultdict(list)
    for p in read_jsonl(ROOT / "data/processed/pages.jsonl"):
        reports[p["report_id"]].append(p)
    result = {}
    for rid, pages in reports.items():
        if args.only and rid != args.only:
            continue
        result[rid] = extract(pages)
        print(rid, " ".join(f"{m}={len(result[rid].get(m, []))}" for m in LABELS), flush=True)
    path = ROOT / "data/review_v0.4" / ((args.only or "all") + "-candidates.json")
    write_json(path, result)
    print("CANDIDATES=" + str(path), flush=True)


if __name__ == "__main__":
    main()
