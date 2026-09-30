"""Conservative request checks for the annual, single-result chat contract."""
import re
import unicodedata


PERIOD = re.compile(
    r"(?<![a-z])(?:h\s*[12]|q\s*[1-4])(?![a-z0-9])"
    r"|上半年|下半年|半年度|半年报|中报|中期|季度|季报|单季|第?[一二三四1234]季"
    r"|(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s*月"
    r"|20\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?"
    r"|(?<![a-z])(?:ttm|ytd)(?![a-z])|滚动|年初至今"
    r"|first\s+half|second\s+half|half[- ]?year|semi[- ]?annual|quarter",
    re.IGNORECASE,
)
METRIC = r"(?:归母净利润|净利润|毛利率|营业收入|营收|营业成本|总资产|资产总计|总负债|负债合计|经营现金流净额|经营现金流|资产负债率)"
MULTI = re.compile(METRIC + r"\s*(?:、|,|以及|及|和|与)\s*" + METRIC)
VALUE_AND_CHANGE = re.compile(METRIC + r"\s*(?:以及|及|和|与)\s*(?:其|它的|对应的)?\s*同比")


def inspect_request(question, years=(2023, 2024, 2025)):
    text = unicodedata.normalize("NFKC", question).strip().casefold()
    if not text:
        return {"status": "needs_clarification", "reason": "empty_question", "message": "请输入财报问题。"}
    match = PERIOD.search(text)
    if match:
        return {
            "status": "unsupported_period", "reason": "annual_data_only", "matched": match.group(),
            "message": "当前已核验数据仅覆盖2023—2025年年度报表，不支持半年、季度、月度或滚动期间。"
                       "本题不会使用全年数据代替所问期间，也不会启动模型。"
                       "这不表示公司没有披露该期间数据；需要先补充对应报告并核验。",
        }
    if re.search(r"今年|去年|本年|上年|最新|最近", text) and not re.search(r"20\d{2}", text):
        return {"status": "needs_clarification", "reason": "relative_year",
                "message": "请明确要查询的年度（2023、2024或2025）；当前快照不是实时财报数据库。"}
    mentioned = {int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", text)}
    if mentioned - set(years):
        return {"status": "unsupported_year", "reason": "outside_snapshot",
                "message": "所问年份超出当前已核验年度快照，请限定为2023、2024或2025年。"}
    pairs = list(MULTI.finditer(text))
    # A formula's operands are not separate requested outputs.
    formula_only = "毛利率" in text and all(
        re.fullmatch(r"(?:营业收入|营收)\s*(?:和|与|、|及)\s*营业成本", m.group()) for m in pairs
    )
    if (pairs and not formula_only) or VALUE_AND_CHANGE.search(text):
        return {"status": "needs_decomposition", "reason": "multiple_requested_results",
                "message": "本入口一次返回一个指标或一个计算结果，请将多个指标、指标值与同比拆开提问。"
                           "例如分别问某公司某年度归母净利润、归母净利润同比、毛利率。"
                           "毛利率的同比增长率与同比变化多少个百分点不同，请明确所需定义。"}
    return {"status": "accepted", "reason": "no_known_contract_conflict", "message": None}


def dispatch(question, agent_factory, years=(2023, 2024, 2025)):
    check = inspect_request(question, years)
    if check["status"] != "accepted":
        return {"question": question, "status": check["status"], "request_check": check,
                "final": None, "events": [], "messages": [], "seen_fact_ids": [], "seconds": 0,
                "model_invoked": False}
    trace = agent_factory().run(question)
    trace["request_check"] = check
    return trace
