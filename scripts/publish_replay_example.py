"""Publish a small, explicitly non-LLM example from a completed local replay."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight_ashare.common import read_json, write_json
from finsight_ashare.evaluation import render_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()
    rows = read_json(args.cases)
    if len(rows) != 10 or any(r["mode"] != "scripted_tool_replay" or not r["score"]["task_correct"] for r in rows):
        raise ValueError("Expected exactly ten successful non-LLM replay cases")
    artifact = {"status": "verified_tool_replay_only", "llm_executed": False,
                "notice": "预设动作的真实财报工具回放；未调用9B，不是模型评测或训练结果。", "cases": rows}
    output = ROOT / "examples/maotai_2024"
    write_json(output / "tool_replay.json", artifact)
    render_report(output / "tool_replay.html", rows, {"notice": artifact["notice"], "cases": len(rows)})
    print("Published clearly labeled tool replay example: " + str(output))


if __name__ == "__main__":
    main()
