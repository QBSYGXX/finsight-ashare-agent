"""Extract three small public examples from the audited result ZIP, without raw logs."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "6217e83ff40acd10a1917bd3c19244b72fc18064e8fedf833237ccc10603bc18"
CASES = [
    ("数值查询：基础模型", "evaluation/base/diagnostic/M01.json"),
    ("原文证据：基础模型", "evaluation/base/diagnostic/R02.json"),
    ("保留的退化：新LoRA", "evaluation/tuned/diagnostic/R02.json"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Not the audited result archive")
    cases = []
    with zipfile.ZipFile(args.archive) as z:
        for title, member in CASES:
            payload = z.read(member)
            d = json.loads(payload)
            cases.append({
                "title": title, "member": member, "member_sha256": hashlib.sha256(payload).hexdigest(),
                **{k: d[k] for k in ("question", "protocol", "status", "context", "results", "quotes", "task_plan", "seconds", "completeness_verified")},
                "actions": [{"step": e["step"], "action": e.get("action"),
                             "error": e.get("error")} for e in d["events"]],
            })
    result = {"archive": args.archive.name, "archive_sha256": EXPECTED,
              "note": "Selected recorded cases, not a live demo or independent benchmark. Times exclude initial model loading.",
              "cases": cases}
    dest = ROOT / "examples/research_v1"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "cases.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    escape = html.escape
    sections = []
    for case in cases:
        rows = "".join("<tr>" + "".join("<td>" + escape(str(v)) + "</td>" for v in
                         [r["name"], r.get("value", "—"), r.get("unit", ""), r["status"],
                          "; ".join(str(s["report_id"]) + " PDF第" + str(s["page"]) + "页" for s in r.get("sources", []))]) + "</tr>"
                       for r in case["results"])
        quotes = "".join('<blockquote>' + escape(q["quote"]) + '</blockquote><p><a href="' + escape(q["source"]["source_url"], quote=True)
                         + '">原始年报</a> · PDF第' + str(q["source"]["page"]) + '页</p>' for q in case["quotes"])
        failure = '<p class="failure">模型把narrative登记为false，未检索原文就拒答。说明微调并非全面优于基础模型。</p>' if 'tuned' in case['member'] else ''
        sections.append('<section><h2>' + escape(case["title"]) + '</h2><p>' + escape(case["question"]) + '</p><p>状态：<code>'
                        + escape(case["status"]) + '</code> · 本次耗时 ' + str(case["seconds"]) + ' 秒</p>'
                        + ('<table><tr><th>指标</th><th>数值</th><th>单位</th><th>状态</th><th>来源</th></tr>' + rows + '</table>' if rows else '')
                        + quotes + failure + '<details><summary>查看模型动作</summary><pre>'
                        + escape(json.dumps(case["actions"], ensure_ascii=False, indent=2)) + '</pre></details></section>')
    page = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FinSight · 真实运行案例</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;color:#20334a;background:#f5f7fa}section{background:white;border:1px solid #dce2ea;border-radius:10px;margin:24px 0;padding:24px}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #dce2ea;text-align:left;padding:8px}pre,blockquote{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:16px;margin:16px 0}a{color:#1557a0}.failure{border-left:4px solid #bb6324;padding-left:12px}details{margin-top:20px}</style>
<h1>FinSight · 真实运行案例</h1>
<p>Research v1.0 / 2026-09-30 · 以下摘自2026-09-29同轮对照结果，不调用模型。</p>
<p>只展示三条选定记录，不能代替完整评测。数字来自财务工具，文字是原文摘录；来源存在不自动证明回答完整。耗时不含模型初始加载。</p>
'''
    page += "\n".join(sections)
    page += '<p>详细来源、页码、公式、动作及原记录哈希见 <a href="cases.json">cases.json</a>；完整结果见 <a href="../../docs/experiments/lora-v0161-result-audit-20260930.md">核验报告</a>。</p></html>\n'
    (dest / "report.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"EXAMPLES={dest} CASES={len(cases)}")


if __name__ == "__main__":
    main()
