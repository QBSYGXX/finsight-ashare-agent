import json
import pytest

from finsight_ashare.agent import ReportAgent
from finsight_ashare.dataset import ReplayModel, make_tasks, replay
from finsight_ashare.protocol import action, parse_action, parse_call


@pytest.mark.parametrize("code", [
    "__import__('os').system('echo unsafe')", "call_tool(tool_name='table_read', company=open('secret').read())",
    "call_tool(**{'tool_name':'table_read'})", "(lambda: 1)()", "call_tool(tool_name='a', tool_name='b')",
    "call_tool(tool_name='calculate', fact_ids=[x for x in []])", "call_tool(tool_name='calculate'); print('bad')",
])
def test_arbitrary_code_never_executes(code):
    with pytest.raises((ValueError, SyntaxError)):
        parse_call(code)


def test_valid_action_and_truncated_tag():
    kind, content = parse_action(action("table_read", company="600519", year=2024, metric="revenue"))
    assert kind == "code"
    assert parse_call(content)["year"] == 2024
    with pytest.raises(ValueError):
        parse_action("<execute>call_tool(tool_name='table_read')")


def test_ungrounded_citation_rejected_and_step_limit_explicit(tools):
    task = make_tasks(tools)[0]
    agent = ReportAgent(ReplayModel([task["responses"][-1]]), tools, max_steps=1)
    trace = agent.run(task["question"])
    assert trace["status"] == "step_limit"
    assert trace["final"] is None
    assert "not been retrieved" in trace["events"][0]["error"]


def test_calculation_must_follow_retrieval(tools):
    response = action("calculate", operation="convert", fact_ids=["600519-2024-revenue"])
    trace = ReportAgent(ReplayModel([response]), tools, max_steps=1).run("Revenue?")
    assert trace["events"][0]["tool_success"] is False


def test_ten_real_fact_trajectories_replay(tools):
    tasks = make_tasks(tools)
    assert len(tasks) >= 10
    for task in tasks:
        result = replay(task, tools)
        assert result["final"] == task["expected"]
        assert result["messages"][-1]["role"] == "assistant"
