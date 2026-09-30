"""Add preflight validation without mutating the v0.7 evaluation or frozen source."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

from preflight import dispatch, inspect_request

ROOT = Path(os.environ.get("FINSIGHT_ROOT", Path(__file__).resolve().parents[1])).resolve()


def verify_dependencies():
    pin = json.loads((Path(__file__).parent / "dependency_pin.json").read_text(encoding="utf-8"))
    for name, expected in pin.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned v0.7 dependency changed: " + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "tuned"], default="tuned")
    parser.add_argument("--check", help="Check a question without loading data or a model")
    args = parser.parse_args()
    if args.check is not None:
        print(json.dumps(inspect_request(args.check), ensure_ascii=False, indent=2))
        return
    verify_dependencies()
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "evaluation_v07"))
    from finsight_ashare.common import config, digest, read_json, write_json
    from finsight_ashare.dataset import validate_freeze
    from finsight_ashare.evaluation import adapter_files, render_report
    from finsight_ashare.tools import FinancialTools
    from resolved_agent import ResolvedAgent
    from resolver import CompanyResolver

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

            agent = ResolvedAgent(LazyModel(), FinancialTools.load(ROOT, search_backend="bge"),
                                  CompanyResolver.load(ROOT, ROOT / "evaluation_v07"),
                                  config(ROOT)["max_steps"])
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
        trace = dispatch(question, get_agent, config(ROOT)["years"])
        check = trace["request_check"]
        if check["status"] != "accepted":
            print(check["message"], flush=True)
        elif trace["resolution"]["status"] != "resolved":
            print(trace["resolution"]["message"], flush=True)
        elif trace.get("final") is not None:
            print(json.dumps(trace["final"], ensure_ascii=False, indent=2), flush=True)
        else:
            print("本次未完成回答，状态：" + trace["status"] + "。这不等于财报数据不存在。"
                  "请核对年度与口径，并将问题拆为单个指标或计算结果。", flush=True)
        destination = ROOT / "outputs/chat_v071" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        write_json(destination / "trace.json", {"model": args.model, **trace})
        render_report(destination / "report.html", [{"question": question, "trace": trace}])
        print("TRACE=" + str(destination / "trace.json"), flush=True)


if __name__ == "__main__":
    main()
