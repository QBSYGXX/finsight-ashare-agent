from copy import deepcopy
from decimal import Decimal
import pytest

from finsight_ashare.tools import FinancialTools


def test_unit_conversion_uses_revenue_not_total_revenue(tools):
    result = tools.calculate("convert", ["600519-2024-revenue"], "亿元")
    assert result["value"] == "1708.9915"
    assert result["sources"][0]["page"] == 63
    assert result["unit"] == "亿元"


def test_growth_and_gross_margin(tools):
    assert tools.calculate("growth", ["600519-2024-revenue", "600519-2023-revenue"])["value"] == "15.7120"
    assert tools.calculate("gross_margin", ["600519-2024-revenue", "600519-2024-cost"])["value"] == "91.9312"


@pytest.mark.parametrize("operation,ids", [
    ("growth", ["600519-2023-revenue", "600519-2024-revenue"]),
    ("growth", ["600519-2024-revenue", "600519-2023-cost"]),
    ("ratio", ["600519-2024-revenue", "600519-2023-cost"]),
    ("gross_margin", ["600519-2024-cost", "600519-2024-revenue"]),
    ("convert", ["600519-2024-revenue", "600519-2024-cost"]),
])
def test_invalid_financial_formulas_fail(tools, operation, ids):
    with pytest.raises(ValueError):
        tools.calculate(operation, ids)


@pytest.mark.parametrize("value", ["0", "-100"])
def test_nonpositive_growth_base_is_not_reported_as_normal_growth(facts, value):
    changed = deepcopy(facts)
    next(f for f in changed if f["id"] == "600519-2023-revenue")["value"] = value
    with pytest.raises(ValueError, match="Non-positive"):
        FinancialTools(changed).calculate("growth", ["600519-2024-revenue", "600519-2023-revenue"])


def test_no_fallback_between_scopes_or_years(tools):
    assert tools.table_read("600519", 2024, "revenue", scope="parent")["status"] == "missing"
    assert tools.table_read("600519", 2025, "revenue")["status"] == "missing"


def test_all_tools_enforce_company_split(facts):
    restricted = FinancialTools(facts, allowed_companies=["002594"])
    for call in [lambda: restricted.table_read("600519", 2024, "revenue"),
                 lambda: restricted.report_search("营业收入", "600519"),
                 lambda: restricted.calculate("convert", ["600519-2024-revenue"])]:
        with pytest.raises(ValueError, match="outside this split"):
            call()


def test_unverified_numeric_value_cannot_enter_calculation(facts):
    facts[0]["verified"] = False
    with pytest.raises(ValueError, match="Unverified"):
        FinancialTools(facts).calculate("convert", [facts[0]["id"]])
