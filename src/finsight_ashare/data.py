"""Official report acquisition, page extraction and explicit fact review gate."""
import csv
import json
import re
import urllib.parse
import urllib.request
from decimal import Decimal
from pathlib import Path

from .common import ROOT, companies, config, file_hash, read_json, read_jsonl, write_json, write_jsonl
from .tools import METRICS, UNITS

HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}
FIELDS = ["report_id", "year", "metric", "value", "unit", "scope", "page", "evidence", "reviewer", "status", "statement_review"]
ALIASES = {"net_profit": ["归属于上市公司股东的净利润", "归属于母公司股东的净利润"],
           "assets": ["资产总计", "总资产"], "liabilities": ["负债合计", "总负债"]}


# 模块：财报获取；作用：从巨潮解析报告，缓存并校验PDF；核心业务：否。
def request_json(url, form):
    request = urllib.request.Request(url, data=urllib.parse.urlencode(form).encode(), headers=HEADERS)
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def discover(company, year):
    results = request_json("https://www.cninfo.com.cn/new/information/topSearch/detailOfQuery",
                           {"keyWord": company["code"], "maxSecNum": 10, "maxListNum": 10})
    securities = results if isinstance(results, list) else results.get("keyBoardList", [])
    security = next((s for s in securities if s.get("code") == company["code"]), None)
    if not security:
        raise ValueError("Cannot resolve CNINFO orgId; add an official PDF URL to data/manifest.json")
    data = request_json("https://www.cninfo.com.cn/new/hisAnnouncement/query", {
        "pageNum": 1, "pageSize": 30, "tabName": "fulltext", "stock": company["code"] + "," + security["orgId"],
        "searchkey": "", "column": "sse" if company["code"].startswith("6") else "szse",
        "plate": "", "category": "category_ndbg_szsh", "seDate": f"{year+1}-01-01~{year+1}-12-31",
        "sortName": "time", "sortType": "desc", "isHLtitle": "false"})
    matches = []
    for item in data.get("announcements") or []:
        title = re.sub("<[^>]+>", "", item["announcementTitle"])
        if str(year) in title and "年度报告" in title and not any(t in title for t in ("摘要", "英文", "取消", "提示", "更正公告")):
            matches.append({"title": title, "url": "https://static.cninfo.com.cn/" + item["adjunctUrl"]})
    unique = {m["url"]: m for m in matches}
    if len(unique) != 1:
        raise ValueError("Report selection is ambiguous or unavailable: " + json.dumps(list(unique.values()), ensure_ascii=False))
    return next(iter(unique.values()))


def manifest(root=ROOT):
    path = root / "data/manifest.json"
    if not path.exists():
        rows = [{"id": f"{c['code']}-{year}", "company": c["code"], "report_year": year, "url": "", "title": "", "sha256": ""}
                for c in companies(root).values() for year in config(root)["years"]]
        write_json(path, rows)
    return read_json(path)


def download_pdf(url, destination):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"static.cninfo.com.cn", "www.sse.com.cn", "disc.static.szse.cn"}:
        raise ValueError("Use an official HTTPS disclosure PDF URL (CNINFO/SSE/SZSE)")
    temporary = destination.with_suffix(".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
        size = 0
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > 150 * 1024 * 1024:
                raise ValueError("Report exceeds 150 MB")
            stream.write(block)
    with temporary.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("Download is not a PDF")
    temporary.replace(destination)


def fetch(root=ROOT, only=None):
    rows = manifest(root)
    for index, item in enumerate(rows):
        if only and item["id"] != only:
            continue
        print(f"[{index+1}/{len(rows)}] report {item['id']}", flush=True)
        path = root / "data/raw" / (item["id"] + ".pdf")
        if path.exists() and item["sha256"] and file_hash(path) == item["sha256"] and item.get("downloaded_url") == item["url"]:
            print("  verified cache hit", flush=True)
            continue
        if not item["url"]:
            item.update(discover(companies(root)[item["company"]], item["report_year"]))
            write_json(root / "data/manifest.json", rows)
        download_pdf(item["url"], path)
        item["sha256"] = file_hash(path)
        item["downloaded_url"] = item["url"]
        write_json(root / "data/manifest.json", rows)
        print(f"  downloaded and hashed: {path}", flush=True)


def compact(text):
    return re.sub(r"\s+", "", text)


# 模块：数据审计；作用：候选抽取不等于核验，人工确认后才进入事实库；核心业务：是。
def ingest(root=ROOT):
    import pymupdf
    pages, candidates = [], []
    rows = manifest(root)
    for item in rows:
        path = root / "data/raw" / (item["id"] + ".pdf")
        if not path.exists():
            continue
        if not item["sha256"] or file_hash(path) != item["sha256"]:
            raise ValueError("Report hash mismatch: " + item["id"])
        print("Extracting pages: " + item["id"], flush=True)
        with pymupdf.open(path) as doc:
            report_pages = [{"report_id": item["id"], "company": item["company"], "report_year": item["report_year"], "pdf_sha256": item["sha256"],
                             "page": i+1, "text": page.get_text(sort=True)} for i, page in enumerate(doc)]
        if sum(len(p["text"].strip()) for p in report_pages) < 1000:
            raise ValueError("Scanned/empty PDF requires OCR and manual verification: " + item["id"])
        pages.extend(report_pages)
        for metric, label in METRICS.items():
            aliases = ALIASES.get(metric, [label])
            selected = next((p for p in report_pages if any(compact(a) in compact(p["text"]) for a in aliases)), None)
            if selected:
                lines = selected["text"].splitlines()
                row = next((i for i, line in enumerate(lines) if any(compact(a) in compact(line) for a in aliases)), 0)
                snippet = " ".join(lines[max(0, row-2):row+5])[:1000]
                candidates.append({"report_id": item["id"], "year": item["report_year"], "metric": metric, "value": "", "unit": "",
                                   "scope": "consolidated", "page": selected["page"], "evidence": snippet, "reviewer": "", "status": "pending"})
    if not pages:
        raise ValueError("No downloaded reports. Run prepare --stage fetch first.")
    write_jsonl(root / "data/processed/pages.jsonl", pages)
    audit = root / "data/audit.csv"
    existing = []
    if audit.exists():
        with audit.open(encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
    keys = {(r["report_id"], r["metric"], str(r["year"])) for r in existing}
    additions = [r for r in candidates if (r["report_id"], r["metric"], str(r["year"])) not in keys]
    with audit.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing + additions)
    print(f"Extracted {len(pages)} pages. Review {audit}; existing review rows preserved.", flush=True)


def validate_statement_review(row, report, pages):
    """Validate the reviewed column selection and retain compact comparison provenance."""
    review = json.loads(row["statement_review"])
    for key in ("header", "unit_evidence", "column_evidence"):
        excerpt = review[key]
        page = pages[(report["id"], excerpt["page"])]
        if page["pdf_sha256"] != report["sha256"] or not excerpt["text"].strip() or compact(excerpt["text"]) not in compact(page["text"]):
            raise ValueError("Statement review excerpt/hash mismatch: " + key)
    header = compact(review["header"]["text"])
    if "合并" not in header or int(row["year"]) != report["report_year"]:
        raise ValueError("Statement review requires consolidated current-year columns")
    columns = compact(review["column_evidence"]["text"])
    year = int(row["year"])
    if str(year) not in columns or not (str(year - 1) in columns or "期初" in columns or f"{year}年1月1日" in columns):
        raise ValueError("Current/prior column dates are missing")
    unit_text = compact(review["unit_evidence"]["text"])
    unit_match = re.search(r"(?:单位[:：]?(?:为)?(?:人民币)?|人民币)(千元|万元|亿元|元)", unit_text)
    if not unit_match or unit_match[1] != row["unit"]:
        raise ValueError("Statement unit does not match the source declaration")
    token_pattern = r"(?<![\d.])(?:\(-?\d[\d,]*(?:\.\d+)?\s*\)|-?\d[\d,]*(?:\.\d+)?)(?![\d.])"
    tokens = re.findall(token_pattern, row["evidence"])
    def number(token):
        clean = compact(token).replace(",", "")
        return -Decimal(clean[1:-1]) if clean.startswith("(") else Decimal(clean)
    # This reviewed corpus contains monetary rows above 1,000; omit note indices.
    monetary = [t for t in tokens if abs(number(t)) >= 1000]
    if len(monetary) != review["columns"] or review["columns"] not in (2, 4):
        raise ValueError("Unexpected statement column count")
    current, prior = map(number, monetary[:2])
    normalization = review.get("normalization", "none")
    if normalization == "expense_magnitude":
        if report["company"] != "000333" or row["metric"] != "cost" or current >= 0 or prior >= 0:
            raise ValueError("Expense normalization is restricted to reviewed Midea cost rows")
        current, prior = -current, -prior
    elif normalization != "none":
        raise ValueError("Unknown normalization")
    if current != Decimal(row["value"]) or prior != Decimal(review["comparison_value"]):
        raise ValueError("Reviewed values do not match selected current/prior columns")
    return {"basis": "current column of own annual report; not a latest-restated time series",
            "normalization": normalization,
            "comparison": {"year": year - 1, "value": str(prior), "unit": row["unit"],
                           "page": int(row["page"]), "verified": True}}


def audit(root=ROOT):
    rows = {r["id"]: r for r in manifest(root)}
    pages = {(p["report_id"], p["page"]): p for p in read_jsonl(root / "data/processed/pages.jsonl")}
    verified, used_keys = [], set()
    with (root / "data/audit.csv").open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            if row["status"] != "approved":
                continue
            try:
                report = rows[row["report_id"]]
                year, page = int(row["year"]), int(row["page"])
                key = (report["company"], year, row["metric"], row["scope"])
                if key in used_keys:
                    raise ValueError("Duplicate/restated fact; select a single documented basis")
                if row["metric"] not in METRICS or row["unit"] not in UNITS or row["scope"] != "consolidated":
                    raise ValueError("Unsupported metric, unit or scope")
                if year not in config(root)["years"] or year > report["report_year"]:
                    raise ValueError("Invalid fiscal year")
                if not row["reviewer"].strip() or len(row["evidence"].strip()) < 8:
                    raise ValueError("Reviewer and verbatim evidence are required")
                evidence = compact(row["evidence"])
                source_page = pages[(report["id"], page)]
                if source_page["pdf_sha256"] != report["sha256"]:
                    raise ValueError("Extracted page belongs to an older PDF; rerun ingest")
                page_text = compact(source_page["text"])
                if evidence not in page_text:
                    raise ValueError("Evidence is not a verbatim excerpt from the cited PDF page")
                value = Decimal(row["value"].replace(",", ""))
                if not value.is_finite():
                    raise ValueError("Non-finite value")
                tokens = re.findall(r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?(?![\d.])", row["evidence"])
                if not any(Decimal(token.replace(",", "")) == value for token in tokens):
                    raise ValueError("Value is absent from the evidence excerpt")
                extra = validate_statement_review(row, report, pages) if row.get("statement_review") else {}
                used_keys.add(key)
                verified.append({"id": f"{report['company']}-{year}-{row['metric']}", "company": report["company"],
                    "year": year, "metric": row["metric"], "value": str(value), "unit": row["unit"], "scope": row["scope"],
                    "report_id": report["id"], "report_year": report["report_year"], "page": page,
                    "source_url": report["url"], "pdf_sha256": report["sha256"], "evidence": row["evidence"],
                    "reviewer": row["reviewer"], "verified": True, **extra})
            except (ValueError, KeyError, ArithmeticError) as error:
                raise ValueError(f"audit.csv line {line}: {error}") from error
    if not verified:
        raise ValueError("No approved facts; do not train on unreviewed extraction")
    for report_id in {f["report_id"] for f in verified}:
        if file_hash(root / "data/raw" / (report_id + ".pdf")) != rows[report_id]["sha256"]:
            raise ValueError("PDF changed after extraction: " + report_id)
    write_json(root / "data/verified/facts.json", verified)
    write_json(root / "data/verified/audit_receipt.json", {"facts": len(verified), "audit_csv_sha256": file_hash(root / "data/audit.csv")})
    print(f"Verified {len(verified)} facts, including exact numeric token and page checks.", flush=True)
    return verified
