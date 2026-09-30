from copy import deepcopy
import json

import pytest

from finsight_ashare.data import validate_statement_review
from finsight_ashare.dataset import make_tasks
from finsight_ashare.tools import FinancialTools


def cross_report(facts):
    result = deepcopy(facts)
    current = next(f for f in result if f['id'] == '600519-2024-revenue')
    previous = next(f for f in result if f['id'] == '600519-2023-revenue')
    previous['report_id'] = '600519-2023'
    current['comparison'] = {'verified': True, 'year': 2023, 'value': previous['value'], 'unit': previous['unit']}
    return result, current, previous


def test_thousand_yuan_conversion(facts):
    facts[0].update(value='100000', unit='千元')
    assert FinancialTools(facts).calculate('convert', [facts[0]['id']])['value'] == '1.0000'


def test_cross_report_matching_comparison_allows_growth(facts):
    items, current, previous = cross_report(facts)
    assert FinancialTools(items).calculate('growth', [current['id'], previous['id']])['value'] == '15.7120'


@pytest.mark.parametrize('operation', ['growth', 'difference'])
def test_restated_comparative_cannot_mix_with_old_report(facts, operation):
    items, current, previous = cross_report(facts)
    current['comparison']['value'] = '100000000000'
    with pytest.raises(ValueError, match='Reporting basis mismatch'):
        FinancialTools(items).calculate(operation, [current['id'], previous['id']])


def test_missing_comparison_fails_closed_and_omission_is_recorded(facts, project):
    items, current, previous = cross_report(facts)
    del current['comparison']
    omissions = []
    make_tasks(FinancialTools(items), project, omissions)
    assert any(o['arguments']['fact_ids'] == [current['id'], previous['id']] and 'audited prior-year' in o['reason'] for o in omissions)


def test_ratio_cannot_mix_reporting_snapshots(facts):
    items = deepcopy(facts)
    next(f for f in items if f['id'] == '600519-2024-cost')['report_id'] = '600519-2025'
    with pytest.raises(ValueError, match='same report snapshot'):
        FinancialTools(items).calculate('gross_margin', ['600519-2024-revenue', '600519-2024-cost'])


def review_fixture(metric='cost', company='000333', normalization='expense_magnitude'):
    header = '合并利润表 2024年度 2023年度 单位：千元'
    evidence = '营业成本 (2,000) (1,500)'
    report = {'id': company + '-2024', 'company': company, 'report_year': 2024, 'sha256': 'test'}
    excerpt = {'page': 1, 'text': header}
    review = {'header': excerpt, 'unit_evidence': excerpt, 'column_evidence': excerpt,
              'columns': 2, 'comparison_value': '1500', 'normalization': normalization}
    row = {'year': '2024', 'value': '2000', 'unit': '千元', 'metric': metric, 'page': '1',
           'evidence': evidence, 'statement_review': json.dumps(review, ensure_ascii=False)}
    pages = {(report['id'], 1): {'text': header + '\n' + evidence, 'pdf_sha256': 'test'}}
    return row, report, pages


def test_negative_expense_is_normalized_with_explicit_provenance():
    result = validate_statement_review(*review_fixture())
    assert result['normalization'] == 'expense_magnitude'
    assert result['comparison']['value'] == '1500'


def test_negative_profit_must_not_be_abs_normalized():
    with pytest.raises(ValueError, match='restricted'):
        validate_statement_review(*review_fixture(metric='net_profit'))


def test_negative_profit_preserves_sign():
    row, report, pages = review_fixture(metric='net_profit', company='601012', normalization='none')
    row['value'] = '-2000'
    review = json.loads(row['statement_review'])
    review['comparison_value'] = '-1500'
    row['statement_review'] = json.dumps(review)
    assert validate_statement_review(row, report, pages)['comparison']['value'] == '-1500'


@pytest.mark.parametrize('change', ['unit', 'column', 'source_hash'])
def test_review_rejects_wrong_units_columns_and_source(change):
    row, report, pages = review_fixture()
    if change == 'unit':
        row['unit'] = '元'
    elif change == 'column':
        row['value'] = '1500'
    else:
        report['sha256'] = 'changed'
    with pytest.raises(ValueError):
        validate_statement_review(row, report, pages)
