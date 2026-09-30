import json

import pytest

from finsight_ashare.agent import ReportAgent
from finsight_ashare.dataset import ReplayModel, make_tasks
from finsight_ashare.evaluation import score
from finsight_ashare.protocol import normalize_final_envelope, parse_action, parse_final


@pytest.fixture
def task(tools):
    return next(t for t in make_tasks(tools) if t["kind"] == "convert"
                and t["expected"]["citations"] == ["600519-2024-cost"]
                and t["expected"]["unit"] == "\u4e07\u5143")


def unclosed(task, **changes):
    _, content = parse_action(task["responses"][-1])
    final = json.loads(content)
    final.update(changes)
    return "<think>\n\n</think>\n\n<final_result>" + json.dumps(final, ensure_ascii=False)


def test_repair_preserves_raw_output_and_passes_scoring(task, tools, capsys):
    raw = unclosed(task)
    with pytest.raises(ValueError):
        parse_action(raw)
    trace = ReportAgent(ReplayModel([*task["responses"][:-1], raw]), tools).run(task["question"])
    event = trace["events"][-1]
    assert trace["status"] == "completed"
    assert score(task, trace)["task_correct"]
    assert event["format_repair"] == "missing_final_result_close"
    assert event["response"] == trace["messages"][-1]["content"] == raw
    assert event["normalized_response"] == raw + "</final_result>"
    assert "AGENT_FORMAT_REPAIR=missing_final_result_close" in capsys.readouterr().out


@pytest.mark.parametrize("suffix", [" trailing text", " {}", "<execute>call_tool()</execute>",
                                   "<final_result>{}</final_result>"])
def test_extra_content_is_not_repaired(task, suffix):
    with pytest.raises(ValueError):
        normalized, _ = normalize_final_envelope(unclosed(task) + suffix)
        _, content = parse_action(normalized)
        parse_final(content)


@pytest.mark.parametrize("payload", ['{"status":"ok"}', '{"status":', "[]", "null"])
def test_incomplete_or_wrong_json_is_not_repaired(payload):
    with pytest.raises(ValueError):
        normalize_final_envelope("<final_result>" + payload)


def test_truncated_tool_and_missing_open_are_not_repaired(task):
    for raw in (task["responses"][0].removesuffix("</execute>"),
                json.dumps(task["expected"])):
        assert normalize_final_envelope(raw) == (raw, None)
        with pytest.raises(ValueError):
            parse_action(raw)


def test_valid_closed_answer_is_unchanged(task):
    raw = task["responses"][-1]
    assert normalize_final_envelope(raw) == (raw, None)


def test_direct_baseline_uses_same_repair(task, tools, facts):
    trace = ReportAgent(ReplayModel([unclosed(task)]), tools).run(task["question"], direct_context=facts)
    assert score(task, trace)["task_correct"]
    assert trace["events"][0]["format_repair"] == "missing_final_result_close"


def test_thinking_text_inside_answer_is_preserved(task):
    answer = "Literal <think>evidence text</think> and <execute> in a JSON string."
    raw = unclosed(task, answer=answer)
    normalized, _ = normalize_final_envelope(raw)
    _, content = parse_action(normalized)
    assert parse_final(content)["answer"] == answer


@pytest.mark.parametrize("changes", [{"value": "1"}, {"unit": "\u5143"},
                                    {"year": 2023}, {"scope": "parent"}])
def test_repair_does_not_bypass_scoring(task, tools, changes):
    trace = ReportAgent(ReplayModel([*task["responses"][:-1], unclosed(task, **changes)]), tools).run(task["question"])
    assert trace["status"] == "completed"
    assert not score(task, trace)["task_correct"]


@pytest.mark.parametrize("retrieve,citations", [(False, ["600519-2024-cost"]),
                                              (True, ["invented-fact"])])
def test_unseen_citations_are_rejected(task, tools, retrieve, citations):
    responses = list(task["responses"][:-1]) if retrieve else []
    responses.append(unclosed(task, citations=citations))
    trace = ReportAgent(ReplayModel(responses), tools, max_steps=len(responses)).run(task["question"])
    assert trace["status"] == "step_limit"
    assert trace["final"] is None
    assert "has not been retrieved" in trace["events"][-1]["error"]
    assert not score(task, trace)["task_correct"]
