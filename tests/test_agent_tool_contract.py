import inspect
import re

import pytest

from finsight_ashare.agent import ReportAgent
from finsight_ashare.dataset import ReplayModel, make_tasks
from finsight_ashare.evaluation import score
from finsight_ashare.protocol import SYSTEM, parse_action, parse_call


BAD_CALL = ("<think>\n\n</think>\n\n<execute>call_tool(tool_name='table_read', "
            "\u53c2\u6570\u540d={'company': '600519', 'year': 2024, "
            "'metric': 'cost', 'scope': 'consolidated'})</execute>")


def cost_task(tools):
    return next(t for t in make_tasks(tools) if t["kind"] == "convert"
                and t["expected"]["citations"] == ["600519-2024-cost"]
                and t["expected"]["unit"] == "\u4e07\u5143")


def test_prompt_tool_examples_match_real_signatures(tools):
    examples = re.findall(r"<execute>(call_tool\(.*?)</execute>", SYSTEM)
    assert len(examples) == 3
    for example in examples:
        arguments = parse_call(example)
        name = arguments.pop("tool_name")
        inspect.signature(getattr(tools, name)).bind(**arguments)


def test_literal_placeholder_rejected_with_concrete_signature(tools):
    _, content = parse_action(BAD_CALL)
    with pytest.raises(ValueError, match="unsupported keywords") as error:
        tools.call(**parse_call(content))
    assert "table_read(company, year, metric, scope='consolidated')" in str(error.value)


def test_invalid_server_responses_remain_failures(tools):
    task = cost_task(tools)
    incomplete = task["responses"][-1].removesuffix("</final_result>")
    trace = ReportAgent(ReplayModel([BAD_CALL] * 7 + [incomplete]), tools).run(task["question"])
    assert trace["status"] == "step_limit"
    assert trace["final"] is None
    assert trace["seen_fact_ids"] == []
    assert not any(event["tool_success"] for event in trace["events"])
    assert not score(task, trace)["task_correct"]


def test_corrected_call_and_closed_final_can_recover(tools, capsys):
    task = cost_task(tools)
    final = task["responses"][-1]
    responses = [BAD_CALL, *task["responses"][:-1], final.removesuffix("</final_result>")[:-1], final]
    trace = ReportAgent(ReplayModel(responses), tools).run(task["question"])
    assert trace["events"][0]["tool_success"] is False
    assert "unsupported keywords" in trace["events"][0]["error"]
    assert "error" in trace["events"][-2]
    assert "format_repair" not in trace["events"][-2]
    assert trace["status"] == "completed"
    assert score(task, trace)["task_correct"]
    feedback = [message["content"] for message in trace["messages"] if message["role"] == "user"]
    assert any("Expected: table_read(company, year, metric" in text for text in feedback)
    logs = capsys.readouterr().out
    assert "AGENT_STEP=1" in logs and "AGENT_RESPONSE=" in logs and "AGENT_ERROR=" in logs
