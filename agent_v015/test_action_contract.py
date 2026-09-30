"""模块：动作兼容回归；作用：重现真实格式/重试错误，确保不掩盖语义错误；核心业务：否。"""
import json

import pytest

from agent_v015.action_contract import parse_envelope
from agent_v015.agent import answer,Session,context,parse_action,prompt
from agent_v015.test_agent import fixture,Scripted,plan_action


def test_single_tool_envelope_preserves_all_financial_arguments():
    requests=[{'metric':'gross_margin','year':2025,'transform':'value'},
              {'metric':'gross_margin','year':2024,'transform':'value'}]
    action,notes=parse_envelope(json.dumps({'plan':{'requests':requests,'narrative':False}}))
    assert action=={'tool':'plan','arguments':{'requests':requests,'narrative':False}}
    assert notes==['single_tool_envelope']
    # Normalizing the envelope must not silently turn prior-year value into yoy_pp.
    assert action['arguments']['requests'][1]['transform']=='value'


@pytest.mark.parametrize('raw',[
    '{"plan":{},"finish":{}}','{"exec":{"cmd":"anything"}}',
    '{"tool":"plan","arguments":{},"extra":1}',
    '{"tool":"plan","tool":"finish","arguments":{}}',
    '{"tool":"plan","arguments":{"requests":[],"requests":[1]}}',
    '{"plan":[]}', '[{"tool":"finish","arguments":{}}]',
    '<think>{"plan":{"requests":[]}}',
])
def test_ambiguous_or_unsafe_format_is_not_repaired(raw):
    with pytest.raises(ValueError):parse_action(raw)


def test_same_source_selection_is_normalized_but_not_fabricated():
    selection={'evidence_id':'observed','first_sentence':2,'last_sentence':4}
    raw={'tool':'finish','arguments':{'evidence_ids':['observed'],'selections':[selection]}}
    action,notes=parse_envelope(json.dumps(raw))
    assert action['arguments']=={'evidence_ids':[],'selections':[selection]}
    assert notes==['redundant_full_excerpt_ids_removed']
    raw['arguments']['evidence_ids']=['different']
    with pytest.raises(ValueError,match='different sources'):parse_envelope(json.dumps(raw))


def test_failed_finish_can_retry_after_required_result_arrives(fixture):
    root,index,_=fixture
    finish={'tool':'finish','arguments':{'selections':[]}}
    model=Scripted([plan_action(['gross_margin']),finish,
                    {'tool':'financial_metric','arguments':{'metric':'gross_margin','year':2024}},finish])
    result=answer('中兴通讯2024年毛利率',root,index,model)
    assert result['status']=='completed' and result['results'][0]['value']=='40.0000'
    assert 'not been checked' in result['events'][1]['error']
    assert result['events'][1]['state_fingerprint']!=result['events'][3]['state_fingerprint']


def test_failed_finish_can_retry_after_evidence_arrives(fixture):
    root,index,_=fixture
    sid=index.search('信用','000063',[2024])[0]['id']
    finish={'tool':'finish','arguments':{'selections':[{'evidence_id':sid,'first_sentence':0,'last_sentence':0}]}}
    model=Scripted([plan_action(narrative=True),finish,
                    {'tool':'report_search','arguments':{'query':'信用','year':2024}},finish])
    result=answer('中兴通讯2024年信用风险',root,index,model)
    assert result['status']=='completed'
    assert 'Unknown evidence_id' in result['events'][1]['error']


def test_failed_action_still_blocked_without_progress(fixture):
    root,index,_=fixture
    finish={'tool':'finish','arguments':{}}
    model=Scripted([plan_action(['gross_margin']),finish,finish])
    result=answer('中兴通讯2024年毛利率',root,index,model,max_steps=3)
    assert result['status']=='step_limit' and 'Repeated action' in result['events'][-1]['error']


def test_successful_search_still_blocked_after_unrelated_progress(fixture):
    root,index,_=fixture
    search={'tool':'report_search','arguments':{'query':'信用','year':2024}}
    model=Scripted([plan_action(['gross_margin'],narrative=True),search,
                    {'tool':'financial_metric','arguments':{'metric':'gross_margin','year':2024}},search])
    result=answer('中兴通讯2024年毛利率和信用风险',root,index,model,max_steps=4)
    assert 'Repeated action' in result['events'][-1]['error']


def test_live_shorthand_and_redundant_citation_are_auditable(fixture):
    root,index,_=fixture
    sid=index.search('信用','000063',[2024])[0]['id']
    model=Scripted([{'plan':{'requests':[],'narrative':True}},
                    {'report_search':{'query':'信用','year':2024}},
                    {'finish':{'evidence_ids':[sid],'selections':[{'evidence_id':sid,'first_sentence':0,'last_sentence':0}]}}])
    result=answer('中兴通讯2024年信用风险',root,index,model)
    assert result['status']=='completed'
    assert result['events'][0]['action_normalization']==['single_tool_envelope']
    assert result['events'][-1]['action_normalization']==['single_tool_envelope','redundant_full_excerpt_ids_removed']
    assert result['quotes'][0]['match_mode']=='selected_sentences'


def test_normalization_does_not_admit_wrong_year_or_modify_plan(fixture):
    root,index,config=fixture
    session=Session(root,index,context('中兴通讯2024年毛利率',config,index))
    action=parse_action('{"plan":{"requests":[{"metric":"gross_margin","year":2023}],"narrative":false}}')
    with pytest.raises(ValueError,match='Year'):session.call(action['tool'],action['arguments'])
    assert session.task_plan is None


def test_prompt_uses_catalog_definitions_not_evaluation_reference(fixture):
    root,index,config=fixture
    session=Session(root,index,context('中兴通讯2024年毛利率',config,index))
    text=prompt(session)
    assert session.finance.registry['debt_ratio']['formula'] in text
    assert session.finance.registry['cash_parent_profit_ratio']['formula'] in text
    assert 'selections' in text and 'yoy_pp' in text
    assert '37.9127' not in text and 'natural_v1' not in text
