"""模块：选句回归测试；作用：验证原文坐标、边界与越权拒绝；核心业务：否。"""
import pytest

from agent_v015.excerpts import annotate, select
from agent_v015.agent import Session, context, answer
from agent_v015.test_agent import fixture, Scripted, plan_action


def test_partial_edges_are_not_presented_as_complete_sentences():
    page={'text':'上一节融资渠道发生变化。\n（5）客户信用风险\n本集团通过客户资信调查及信用额度管理来控制信用风险。\n接下来一句在片段末尾被截断。'}
    start=5
    source={'text':page['text'][start:-3],'start':start}
    hit=annotate(source,page)
    assert len(hit['sentences'])==1
    assert hit['sentences'][0]['text']=='本集团通过客户资信调查及信用额度管理来控制信用风险。'
    s=hit['sentences'][0]
    assert hit['text'][s['start']:s['end']]==s['text']


def test_layout_wrap_does_not_split_sentence_or_change_number():
    text='本集团对应收账\n\n款进行持续监控，比例为15%（上年14%）。'
    hit=annotate({'text':text,'start':0},{'text':text})
    assert hit['sentences'][0]['text']==text
    assert len(hit['sentences'])==1


def test_no_sentence_or_unverified_offset_is_explicit():
    table='营业收入       1000       900\n营业成本      600       500'
    assert annotate({'text':table,'start':0},{'text':table})['excerpt_boundary']=='no_complete_sentence_in_excerpt'
    assert annotate({'text':table,'start':1},{'text':table})['excerpt_boundary']=='unverified_offset'


def test_select_copies_observed_source_and_rejects_invalid_ids(fixture):
    root,index,config=fixture
    s=Session(root,index,context('中兴通讯2024年信用风险',config,index))
    s.plan([],narrative=True)
    hit=s.report_search('信用',2024)['hits'][0]
    selection={'evidence_id':hit['id'],'first_sentence':0,'last_sentence':0}
    result=s.finish(selections=[selection])
    q=result['quotes'][0]
    assert q['quote']=='客户信用风险管理措施包括客户资信调查和信用额度管理。'
    assert '服务器命令' not in q['quote']
    start,end=q['source_span']
    assert q['quote']==hit['text'][start:end]
    assert q['page_span']==[start,end]
    for first,last in [(-1,0),(0,999),(True,0),(1,0),('0',0)]:
        with pytest.raises(ValueError):select(hit,first,last)
    with pytest.raises(ValueError):s.finish(selections=[{**selection,'evidence_id':'invented'}])
    redundant=s.finish(selections=[selection],evidence_ids=[hit['id']])
    assert redundant['quotes']==result['quotes']
    assert redundant['selection_normalization']==['redundant_full_excerpt_ids_removed']
    with pytest.raises(ValueError):s.finish(selections=[selection],evidence_ids=['a_different_source'])


def test_model_selects_numbered_sentences_through_live_call_loop(fixture):
    root,index,_=fixture
    hit=index.search('信用','000063',[2024])[0]
    model=Scripted([plan_action(narrative=True),
                    {'tool':'report_search','arguments':{'query':'信用','year':2024}},
                    {'tool':'finish','arguments':{'selections':[{'evidence_id':hit['id'],'first_sentence':0,'last_sentence':0}]}}])
    result=answer('中兴通讯2024年信用风险',root,index,model)
    assert result['status']=='completed' and result['quotes'][0]['match_mode']=='selected_sentences'
    assert result['events'][1]['observation']['hits'][0]['sentences']


def test_read_page_records_segment_offsets(fixture):
    root,index,config=fixture
    page=next(iter(index.pages.values()))
    page['text']='本集团持续监控客户信用风险并采取适当措施。'*120
    s=Session(root,index,context('中兴通讯2024年信用风险',config,index))
    hit=s.report_search('信用',2024)['hits'][0]
    segments=s.read_page(hit['id'])['segments']
    segment=segments[1]
    q=select(segment,0,0)
    assert segment['start']==1500
    assert page['text'][q['page_span'][0]:q['page_span'][1]]==q['quote']
