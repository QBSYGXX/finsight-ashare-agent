"""模块：诊断评分测试；作用：防止缺口径、伪引用和额外目标被算作通过；核心业务：否。"""
from copy import deepcopy
import json

from agent_v015.natural_evaluation import SPEC, score, run
from agent_v015.storage import read
from agent_v015.test_agent import fixture


def numeric_result():
    spec=read(SPEC)
    case=spec['cases'][0]
    rid,target=next(iter(case['numeric'].items()))
    result={'status':'completed','context':case['context'],
            'task_plan':{'requests':case['requests'],'narrative':False},'events':[],
            'results':[{'result_id':rid,'company':'000063','scope':'consolidated','status':'ok',
                        'year':2024,'metric':'gross_margin','transform':'value','period':'annual',
                        **target,'sources':[spec['reference_sources'][sid] for sid in target['citations']]}]}
    return case,result,spec


def test_frozen_cases_are_validation_only_and_separated_from_original():
    from agent_v015.evaluation import CASES
    spec=read(SPEC)
    assert len(spec['cases'])==28 and sum(c['preflight'] for c in spec['cases'])==4
    assert len({c['id'] for c in spec['cases']})==28
    assert {c['context']['company'] for c in spec['cases']}=={'000063','000333'}
    assert not ({c['question'] for c in CASES}&{c['question'] for c in spec['cases']})


def test_natural_scorer_rejects_wrong_value_scope_sources_and_extra_targets():
    case,result,spec=numeric_result()
    assert score(case,result,spec)['passed']
    for field,value in [('value','999'),('unit','亿元'),('scope','parent'),('citations',['invented'])]:
        changed=deepcopy(result)
        changed['results'][0][field]=value
        assert not score(case,changed,spec)['passed']
    changed=deepcopy(result);changed['results'].append(deepcopy(changed['results'][0]))
    assert not score(case,changed,spec)['passed']
    changed=deepcopy(result);changed['results'][0]['sources'][0]['page']+=1
    assert not score(case,changed,spec)['passed']
    changed=deepcopy(result);changed['task_plan']['requests']=[]
    assert not score(case,changed,spec)['passed']


def test_fabricated_quote_is_not_passed_by_correct_numeric_answer():
    case,result,spec=numeric_result()
    result['quotes']=[{'quote':'这是一段不在工具观察中的虚构证据。','source':{'id':'invented'}}]
    assert not score(case,result,spec)['passed']


def test_natural_runner_stops_on_runtime_error(fixture,tmp_path,monkeypatch):
    import agent_v015.natural_evaluation as module
    root,index,_=fixture
    monkeypatch.setattr(module,'sha_lf',lambda _:read(SPEC)['facts_lf_sha256'])
    class Broken:
        calls=0
        def generate(self,messages):
            self.calls+=1
            raise RuntimeError('synthetic engine startup failure')
    model=Broken()
    result=run(root,index,model,tmp_path/'natural')
    assert model.calls==1 and result['execution_status']=='failed' and len(result['not_run'])==27
    assert result['diagnostic_pass']=={'passed':0,'total':1}
