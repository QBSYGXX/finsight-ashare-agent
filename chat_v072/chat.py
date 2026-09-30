"""Interactive chat with company constraints and derived gross-margin support."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("FINSIGHT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT / "chat_v071"))
from preflight import dispatch, inspect_request


def inspect_chat_request(question):
    check = inspect_request(question)
    margin = "毛利率" in question or "gross margin" in question.lower()
    change = any(w in question.lower() for w in ("同比", "增幅", "增长", "百分点", "变化", "yoy", "growth", "change"))
    if check["status"] == "accepted" and margin and change:
        return {"status": "needs_decomposition", "reason": "margin_change_not_supported",
                "message": "当前只支持单年度毛利率；毛利率同比增长率与百分点变化需另行计算，不能用当年毛利率代替。"}
    return check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "tuned"], default="tuned")
    parser.add_argument("--check")
    args = parser.parse_args()
    if args.check is not None:
        print(json.dumps(inspect_chat_request(args.check), ensure_ascii=False, indent=2))
        return
    pin = json.loads((Path(__file__).parent / "dependency_pin.json").read_text(encoding="utf-8"))
    for name, checksum in pin.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Chat dependency changed: " + name)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "evaluation_v07")]
    from finsight_ashare.common import config, digest, read_json, write_json
    from finsight_ashare.dataset import validate_freeze
    from finsight_ashare.evaluation import adapter_files, render_report
    from finsight_ashare.tools import FinancialTools
    from resolver import CompanyResolver
    from guarded_agent import GuardedAgent

    agent = None

    def get_agent():
        nonlocal agent
        if agent is None:
            frozen = validate_freeze(ROOT)
            production = read_json(ROOT / "outputs/training/production.json")
            if production["status"] != "completed" or production["dataset_digest"] != digest(frozen):
                raise ValueError("Matching completed training required")
            adapter = ROOT / production["adapter"] if args.model == "tuned" else None
            if adapter is not None and adapter_files(adapter) != production["adapter_hashes"]:
                raise ValueError("Adapter hashes changed")

            class LazyModel:
                model = None

                def generate(self, messages):
                    if self.model is None:
                        from finsight_ashare.models import LocalModel
                        self.model = LocalModel(ROOT, adapter)
                    return self.model.generate(messages)

            agent = GuardedAgent(LazyModel(), FinancialTools.load(ROOT, search_backend="bge"),
                                 CompanyResolver.load(ROOT, ROOT / "evaluation_v07"), config(ROOT)["max_steps"])
        return agent

    while True:
        try:
            question = input("财报问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        check = inspect_chat_request(question)
        if check["status"] != "accepted":
            trace = {"question": question, "status": check["status"], "request_check": check,
                     "final": None, "events": [], "messages": [], "seen_fact_ids": [], "seconds": 0}
        else:
            trace = dispatch(question, get_agent, config(ROOT)["years"])
        if trace["request_check"]["status"] != "accepted":
            print(trace["request_check"]["message"], flush=True)
        elif trace["resolution"]["status"] != "resolved":
            print(trace["resolution"]["message"], flush=True)
        elif trace.get("final") is not None:
            print(json.dumps(trace["final"], ensure_ascii=False, indent=2), flush=True)
        else:
            print("本次模型回答未通过执行或证据检查，状态：" + trace["status"] +
                  "。不能据此声称财报数据缺失，请保留TRACE排查。", flush=True)
        destination = ROOT / "outputs/chat_v072" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        write_json(destination / "trace.json", {"model": args.model, **trace})
        render_report(destination / "report.html", [{"question": question, "trace": trace}])
        print("TRACE=" + str(destination / "trace.json"), flush=True)


if __name__ == "__main__":
    main()
