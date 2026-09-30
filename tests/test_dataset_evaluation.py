from copy import deepcopy
import pytest

from finsight_ashare.common import read_json, write_json
from finsight_ashare.dataset import build, make_tasks, replay, validate_freeze
from finsight_ashare.evaluation import score, aggregate


def test_pilot_freeze_detects_mutation(project):
    receipt = build(project, pilot=True)
    assert receipt["usable_training"] == 10
    validate_freeze(project, pilot=True)
    train = read_json(project / "data/pilot/train.json")
    assert all(turn["from"] in {"human", "gpt"} for item in train for turn in item["conversations"])
    train[0]["conversations"][0]["value"] = "modified"
    write_json(project / "data/pilot/train.json", train)
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        validate_freeze(project, pilot=True)


def test_full_training_blocks_incomplete_company_coverage(project):
    with pytest.raises(ValueError, match="Missing"):
        build(project)


def test_wrong_unit_and_wrong_citation_fail(tools):
    task = next(t for t in make_tasks(tools) if t["kind"] == "growth")
    trace = replay(task, tools)
    assert score(task, trace)["task_correct"] is True
    trace["final"]["unit"] = "亿元"
    assert score(task, trace)["task_correct"] is False
    trace["final"]["unit"] = "%"
    trace["final"]["citations"] = ["fabricated"]
    assert score(task, trace)["citation_precision"] == 0
    assert score(task, trace)["task_correct"] is False


def test_correct_number_with_wrong_year_fails(tools):
    task = make_tasks(tools)[0]
    trace = replay(task, tools)
    trace["final"]["year"] = 2020
    assert not score(task, trace)["numeric_unit_correct"]


def test_aggregate_separates_abstentions_from_numeric_tasks(tools):
    tasks = make_tasks(tools)
    pair = [next(t for t in tasks if t["kind"] == k) for k in ("growth", "missing_scope")]
    rows = [{"expected": t["expected"], "score": score(t, replay(t, tools))} for t in pair]
    summary = aggregate(rows)
    assert summary["questions"] == 2
    assert summary["numeric_questions"] == 1
    assert summary["task_accuracy"] == 1
    assert summary["abstention_accuracy"] == 1
