"""模块：真实语料回放；作用：显式参考计划检查工具与评分，不代表模型准确率；核心业务：否。"""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_v015.agent import answer
from agent_v015.natural_evaluation import SPEC,score
from agent_v015.retrieval import Index
from agent_v015.storage import read,save


def choose(evidence, narrative):
    candidates=[]
    for source in evidence:
        if source['page'] not in narrative['gold_pages']:continue
        spans=source.get('sentences',[])
        for first in range(len(spans)):
            for last in range(first,len(spans)):
                quote=source['text'][spans[first]['start']:spans[last]['end']]
                if len(quote)>1200:break
                compact=''.join(quote.split())
                if all(any(a in compact for a in group) for group in narrative['anchor_groups']):
                    candidates.append((len(quote),{'evidence_id':source['id'],'first_sentence':first,'last_sentence':last}))
    if not candidates:raise AssertionError('No complete observed sentence span covers the reviewed reference anchors')
    return min(candidates,key=lambda item:item[0])[1]


class ReferenceReplay:
    def __init__(self,case):
        self.case=case
        self.actions=['plan']+case['requests']+(['search','page'] if case['narrative'] else [])+['finish']
        self.position=0
        self.last_generation={'test_double':True,'name':'reference_intent_replay_not_model_accuracy'}

    def generate(self,messages):
        action=self.actions[self.position];self.position+=1
        n=self.case['narrative']
        if action=='plan':
            tool,args='plan',{'requests':self.case['requests'],'narrative':bool(n)}
        elif action=='search':
            tool,args='report_search',{'query':n['query'],'year':n['year'],'top_k':6}
        elif action=='page':
            observation=json.loads(messages[-1]['content'].split('\n',1)[1])
            hit=next(h for h in observation['hits'] if h['page'] in n['gold_pages'])
            tool,args='read_page',{'evidence_id':hit['id']}
        elif action=='finish':
            tool,args='finish',{'abstain':self.case['status']=='insufficient_evidence'}
            if n:
                observed=[]
                for m in messages:
                    if m['content'].startswith('TOOL_DATA'):
                        obs=json.loads(m['content'].split('\n',1)[1])
                        observed+=obs.get('hits',[])+obs.get('segments',[])
                args['selections']=[choose(observed,n)]
        else:tool,args='financial_metric',action
        return json.dumps({'tool':tool,'arguments':args},ensure_ascii=False)


def main():
    index=Index(ROOT,ROOT/'outputs/agent_v015/index/validation-bm25')
    spec=read(SPEC)
    records=[]
    for case in spec['cases']:
        result=answer(case['question'],ROOT,index,ReferenceReplay(case))
        verdict=score(case,result,spec)
        print(case['id'],json.dumps(verdict,ensure_ascii=False),flush=True)
        assert verdict['passed'],result
        for quote in result.get('quotes',[]):
            page=index.pages[(quote['source']['report_id'],quote['source']['page'])]
            start,end=quote['page_span']
            assert page['text'][start:end]==quote['quote']
        records.append({'id':case['id'],'verdict':verdict,'result':result})
    save(ROOT/'outputs/agent_v015/v0154-reference-replay.json',{
        'test_type':'explicit_reference_intents_real_corpus_tools_not_model_accuracy',
        'actual_model_run':False,'records':records})
    # 已观察的服务器选段：证明信用风险例子能剔除前一节利率风险及页尾。
    from agent_v015.agent import Session
    session=Session(ROOT,index,{'company':'000063','name':'中兴通讯','years':[2024],'scope':'consolidated'})
    session.plan([],narrative=True)
    hits=session.report_search('客户信用风险 客户资信评级与授信',2024,6)['hits']
    original=next(h for h in hits if h['id']=='000063-2024:p45:c864')
    selection=choose([original],spec['cases'][12]['narrative'])
    quote=session.finish(selections=[selection])['quotes'][0]
    assert '利率掉期' not in quote['quote'] and '客户资信调查' in quote['quote']
    save(ROOT/'outputs/agent_v015/v0154-excerpt-example.json',{
        'test_type':'deterministic_sentence_selection_not_model_choice','before':original['text'],'after':quote})
    print(f'REAL_CORPUS_REFERENCE_REPLAY=28; ACTUAL_MODEL_RUN=false; EXCERPT_CHARS={len(original["text"])}->{len(quote["quote"])}',flush=True)


if __name__=='__main__':main()
