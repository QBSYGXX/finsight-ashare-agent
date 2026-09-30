import csv
import pytest

from finsight_ashare.common import file_hash, read_json, write_json, write_jsonl
from finsight_ashare.data import FIELDS, audit


@pytest.fixture
def audit_project(project):
    pdf = project / "data/raw/600519-2024.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-test-only-not-a-financial-source")
    checksum = file_hash(pdf)
    write_json(project / "data/manifest.json", [{"id": "600519-2024", "company": "600519", "report_year": 2024,
                "url": "https://static.cninfo.com.cn/test-fixture.pdf", "sha256": checksum}])
    write_jsonl(project / "data/processed/pages.jsonl", [{"report_id": "600519-2024", "page": 1, "pdf_sha256": checksum,
                "text": "Test fixture 营业收入 2024年度 1,000.00 元"}])
    return project


def write_review(project, **changes):
    row = dict(report_id="600519-2024", year=2024, metric="revenue", value="1000.00", unit="元", scope="consolidated",
               page=1, evidence="营业收入 2024年度 1,000.00 元", reviewer="test fixture", status="approved")
    row.update(changes)
    with (project / "data/audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)


def test_pending_candidate_cannot_be_used(audit_project):
    write_review(audit_project, status="pending")
    with pytest.raises(ValueError, match="No approved"):
        audit(audit_project)


def test_approved_fact_requires_exact_numeric_token(audit_project):
    write_review(audit_project, value="100.00")
    with pytest.raises(ValueError, match="absent"):
        audit(audit_project)


def test_invented_excerpt_or_wrong_page_fails(audit_project):
    write_review(audit_project, evidence="invented 1,000.00 evidence")
    with pytest.raises(ValueError, match="verbatim"):
        audit(audit_project)


def test_replaced_pdf_requires_new_extraction(audit_project):
    write_review(audit_project)
    assert len(audit(audit_project)) == 1
    pdf = audit_project / "data/raw/600519-2024.pdf"
    pdf.write_bytes(b"%PDF-changed-test-fixture")
    manifest = read_json(audit_project / "data/manifest.json")
    manifest[0]["sha256"] = file_hash(pdf)
    write_json(audit_project / "data/manifest.json", manifest)
    with pytest.raises(ValueError, match="older PDF"):
        audit(audit_project)
