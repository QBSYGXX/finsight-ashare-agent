"""模块：自然问法诊断；作用：冻结参考、独立评分、记录重试和覆盖；核心业务：否。"""
import json
from pathlib import Path

from .agent import answer
from .storage import read, save, sha, sha_lf

SPEC = Path(__file__).with_name('natural_v1.json')


# 模块：分层评分；作用：核对意图、财务值、来源、选句，不把协议通过称为语义正确；核心业务：否。
def score(case, result, spec):
    rows = result.get('results', [])
    mapped = {r.get('result_id'):r for r in rows}
    expected = set(case['numeric']) | set(case['row_status'])
    checks = {'status':result['status']==case['status'],
              'exact_targets':set(mapped)==expected and len(mapped)==len(rows)}
    if case['preflight']:
        checks['no_model_calls'] = not result.get('events')
    else:
        checks['context'] = all(result.get('context',{}).get(k)==v for k,v in case['context'].items())
        plan = result.get('task_plan') or {}
        key = lambda r:(r.get('year'),r.get('metric'),r.get('transform'))
        actual = [key(r) for r in plan.get('requests',[])]
        checks['plan'] = (set(actual)=={key(r) for r in case['requests']} and
                          len(actual)==len(case['requests']) and plan.get('narrative')==bool(case['narrative']))
    for rid, target in case['numeric'].items():
        row = mapped.get(rid,{})
        sources = row.get('sources',[])
        year,metric,transform=rid.split(':')
        checks[rid] = (row.get('status')=='ok' and row.get('value')==target['value'] and row.get('unit')==target['unit']
                       and (row.get('year'),row.get('metric'),row.get('transform'),row.get('period'))==(int(year),metric,transform,'annual')
                       and set(row.get('citations',[]))==set(target['citations'])
                       and len(sources)==len(target['citations']) and {s.get('id') for s in sources}==set(target['citations'])
                       and all(all(s.get(k)==v for k,v in spec['reference_sources'][s['id']].items()) for s in sources)
                       and all(row.get(k)==case['context'][k] for k in ['company','scope']))
    for rid, status in case['row_status'].items():
        row = mapped.get(rid,{})
        year,metric,transform=rid.split(':')
        checks[rid] = (row.get('status')==status and row.get('value') is None and not row.get('citations')
                       and (row.get('year'),row.get('metric'),row.get('transform'),row.get('period'))==(int(year),metric,transform,'annual')
                       and all(row.get(k)==case['context'][k] for k in ['company','scope']))
    quotes = result.get('quotes',[])
    if case['narrative']:
        n = case['narrative']
        relevant = [q for q in quotes if q['source']['report_id']==f'{n["company"]}-{n["year"]}' and q['source']['page'] in n['gold_pages']]
        text = ''.join(''.join(q['quote'].split()) for q in relevant)
        checks['narrative_anchor_proxy'] = all(any(a in text for a in group) for group in n['anchor_groups'])
    elif not case['numeric']:
        checks['no_substitute_quotes'] = not quotes
    # 引文必须在本次观察中出现，正文、元数据和字符位置一起验证。
    observed = {}
    for event in result.get('events',[]):
        obs = event.get('observation',{})
        for hit in obs.get('hits',[]) + obs.get('segments',[]): observed[hit['id']]=hit
        for row in obs.get('results',[]):
            for source in row.get('sources',[]): observed[source['id']]={**source,'text':source['evidence']}
    valid = True
    for quote in quotes:
        source=observed.get(quote['source']['id'])
        span=quote.get('source_span',[])
        if source is None or len(span)!=2:
            valid=False;continue
        start,end=span
        valid &= (type(start) is int and type(end) is int and 0<=start<end<=len(source['text'])
                  and source['text'][start:end]==quote['quote']
                  and all(quote['source'].get(k)==source.get(k) for k in ['company','report_id','page','pdf_sha256']))
        if quote.get('match_mode')=='selected_sentences':
            ids=quote.get('sentence_ids',[])
            spans=source.get('sentences',[])
            valid &= bool(ids) and all(type(i) is int and 0<=i<len(spans) for i in ids)
            if ids and all(type(i) is int and 0<=i<len(spans) for i in ids):
                valid &= ids==list(range(ids[0],ids[-1]+1)) and [start,end]==[spans[ids[0]]['start'],spans[ids[-1]]['end']]
                valid &= quote.get('page_span')==[source['start']+start,source['start']+end]
    checks['observed_quote_integrity']=bool(valid)
    return {'passed':all(checks.values()),'checks':checks}


# 模块：评测编排；作用：模型只看问题、参考独立保存，逐题落盘并在运行错误时停止；核心业务：否。
def run(root,index,model,dest,retrieval_only=False):
    spec=read(SPEC)
    if sha_lf(root/'data/verified/facts.json')!=spec['facts_lf_sha256']:
        raise ValueError('Natural diagnostic reference facts changed; do not silently regenerate the reference')
    if index.meta['split']!='validation':
        raise ValueError('Natural diagnostics require the validation index')
    save(dest/'evaluation_spec.json',spec)
    records=[]
    for case in spec['cases']:
        print('NATURAL_CASE='+case['id']+' QUESTION='+case['question'],flush=True)
        row={'id':case['id'],'question':case['question'],'category':case['category'],'preflight':case['preflight']}
        if case['narrative']:
            n=case['narrative']
            hits=index.search(n['query'],n['company'],[n['year']],6)
            row['retrieval']={'hit_at_6':any(h['page'] in n['gold_pages'] for h in hits),
                              'hits':[{'id':h['id'],'page':h['page']} for h in hits]}
        if not retrieval_only:
            result=answer(case['question'],root,index,model)
            save(dest/(case['id']+'.json'),result)
            row.update(verdict=score(case,result,spec),status=result['status'],seconds=result['seconds'],
                       model_calls=len(result.get('events',[])),
                       error_actions=sum('error' in e for e in result.get('events',[])),
                       sentence_quotes=sum(q.get('match_mode')=='selected_sentences' for q in result.get('quotes',[])),
                       quote_characters=sum(len(q['quote']) for q in result.get('quotes',[])))
        records.append(row)
        save(dest/'progress.json',records)
        print('NATURAL_RESULT='+json.dumps(row,ensure_ascii=False),flush=True)
        if row.get('status')=='runtime_error':break
    scored=[r for r in records if 'verdict' in r]
    model_rows=[r for r in scored if not r['preflight']]
    preflight=[r for r in scored if r['preflight']]
    retrieval=[r['retrieval'] for r in records if 'retrieval' in r]
    count=lambda rows:{'passed':sum(r['verdict']['passed'] for r in rows),'total':len(rows)}
    summary={'protocol':spec['protocol'],'evaluation_spec_sha256':sha(SPEC),'backend':index.meta['backend'],
             'execution_status':'failed' if any(r.get('status')=='runtime_error' for r in records) else 'completed',
             'not_run':[c['id'] for c in spec['cases'] if c['id'] not in {r['id'] for r in records}],
             'diagnostic_pass':count(scored),'model_invoking_pass':count(model_rows),'preflight_pass':count(preflight),
             'model_cases_with_errors':sum(r['error_actions']>0 for r in model_rows),
             'retrieval_hit_at_6':{'hit':sum(r['hit_at_6'] for r in retrieval),'total':len(retrieval)},
             'limitations':spec['limitations'],'records':records}
    save(dest/'summary.json',summary)
    print('NATURAL_SUMMARY='+json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=False),flush=True)
    return summary
