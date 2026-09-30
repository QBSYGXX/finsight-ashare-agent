from copy import deepcopy
from pathlib import Path
import pytest

from finsight_ashare.common import ROOT, read_json, write_json, write_jsonl
from finsight_ashare.tools import FinancialTools


@pytest.fixture
def facts():
    seed = read_json(ROOT / "examples/maotai_2024/seed.json")
    report = seed["report"]
    return [{"id": f"600519-{f['year']}-{f['metric']}", "company": "600519", "year": f["year"], "metric": f["metric"],
             "value": f["value"], "unit": f["unit"], "scope": f["scope"], "report_id": f["report_id"], "page": f["page"],
             "report_year": 2024, "source_url": report["url"], "pdf_sha256": report["sha256"],
             "evidence": f["evidence"], "verified": True} for f in seed["facts"]]


@pytest.fixture
def tools(facts):
    return FinancialTools(facts)


@pytest.fixture
def project(tmp_path, facts):
    write_json(tmp_path / "configs/project.json", read_json(ROOT / "configs/project.json"))
    write_json(tmp_path / "data/verified/facts.json", facts)
    write_jsonl(tmp_path / "data/processed/pages.jsonl", [])
    return tmp_path
