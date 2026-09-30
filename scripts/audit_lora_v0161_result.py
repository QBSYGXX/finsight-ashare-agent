"""Audit the completed protocol-aligned LoRA experiment without running a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_v015.storage import read,save,sha,verify
from agent_v015.evaluation import CASES,score as diagnostic_score
from agent_v015.natural_evaluation import score as natural_score
from agent_v015.finance import MetricEngine


def audit(path):
    archive_hash=sha(path)
    assert archive_hash==path.with_suffix('.zip.sha256').read_text().split()[0]
    verify(ROOT)
    facts=read(ROOT/'data/verified/facts.json')
    fact_map={f['id']:f for f in facts}
    pages={}
    with (ROOT/'data/processed/pages.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);pages[(row['report_id'],row['page'])]=row
    engine=MetricEngine(facts)
    records=[];sources_count=quotes_count=numeric_count=0
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
        load=lambda n:json.loads(z.read(n))
        manifest=load('EXPORT_MANIFEST.json')
        assert set(z.namelist())==set(manifest)|{'EXPORT_MANIFEST.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
        pin=load('release/PACKAGE_MANIFEST.json')
        assert pin==read(ROOT/'lora_v0161/PACKAGE_MANIFEST.json')
        assert all(sha(ROOT/n)==h for n,h in {**pin['files'],**pin['dependencies']}.items())
        assert load('all-status.json')['status']=='completed'
        prepared=load('prepare.json');full=load('full/completion.json');probe=load('probe/completion.json')
        merged=load('merge.json');state=load('full/trainer_state.json')
        assert full['status']==probe['status']==merged['status']=='completed'
        assert full['adapter_reload'] and probe['adapter_reload']
        assert full['global_step']==state['global_step']==state['max_steps']==144
        assert probe['global_step']==20 and full['epoch']==state['epoch']==2
        assert full['training_identity']==probe['training_identity']
        assert full['experiment_digest']==probe['experiment_digest']==merged['experiment_digest']==prepared['experiment_digest']
        assert merged['adapter_hashes']==full['adapter_hashes']
        assert merged['updated_tensors']==merged['numerically_changed_tensors']==248
        assert merged['preserved_mtp_tensors']==15
        for stage in ('probe','full'):
            done=load(stage+'/completion.json')
            assert hashlib.sha256(z.read(stage+'/adapter/adapter_config.json')).hexdigest()==done['adapter_hashes']['adapter_config.json']
            assert math.isfinite(load(stage+'/reload.json')['loss'])
        configs={arm:load('evaluation/'+arm+'/run_config.json') for arm in ('base','tuned')}
        for key in ('experiment_digest','backend','thinking','seed','max_tokens','index'):
            assert configs['base'][key]==configs['tuned'][key],key
        assert configs['tuned']['adapter']==full['adapter_hashes']
        assert not configs['base']['independent_test_used'] and not configs['tuned']['independent_test_used']
        local_spec=read(ROOT/'agent_v015/natural_v1.json')
        for arm in ('base','tuned'):
            spec=load(f'evaluation/{arm}/natural/evaluation_spec.json')
            assert spec==local_spec
            for group,cases in (('diagnostic',CASES),('natural',spec['cases'])):
                summary=load(f'evaluation/{arm}/{group}/summary.json')
                assert summary['execution_status']=='completed' and not summary['not_run']
                original={r['id']:r for r in summary['records']}
                assert set(original)=={c['id'] for c in cases}
                passed=0
                for case in cases:
                    trace=load(f'evaluation/{arm}/{group}/{case["id"]}.json')
                    assert trace['question']==case['question']
                    verdict=diagnostic_score(case,trace) if group=='diagnostic' else natural_score(case,trace,spec)
                    assert verdict==original[case['id']]['verdict']
                    passed+=verdict['passed']
                    for row in trace.get('results',[]):
                        for source in row.get('sources',[]):
                            assert all(fact_map[source['id']][k]==v for k,v in source.items())
                            sources_count+=1
                        computed=engine.execute(row['company'],row['year'],row['scope'],
                            [{'metric':row['metric'],'transform':row['transform']}])['results'][0]
                        for key in ('status','value','unit','citations','sources'):
                            assert computed.get(key)==row.get(key),(case['id'],key)
                        numeric_count+=1
                    for quote in trace.get('quotes',[]):
                        source=quote['source'];page=pages[(source['report_id'],source['page'])]
                        assert source['pdf_sha256']==page['pdf_sha256']
                        start,end=quote['page_span']
                        assert page['text'][start:end]==quote['quote']
                        quotes_count+=1
                    events=trace.get('events',[])
                    records.append({'arm':arm,'group':group,'id':case['id'],'passed':verdict['passed'],
                        'failed_checks':[k for k,v in verdict['checks'].items() if not v],
                        'status':trace['status'],'preflight':bool(case.get('preflight')),'seconds':trace['seconds'],
                        'model_calls':len(events),'error_actions':sum('error' in e for e in events),
                        'generated_tokens':sum(e.get('generation',{}).get('generated_tokens',0) for e in events),
                        'generation_seconds':sum(e.get('generation',{}).get('seconds',0) for e in events)})
                assert summary['diagnostic_pass']=={'passed':passed,'total':len(cases)}
        comparison=load('comparison.json')
        for group in ('diagnostic','natural'):
            arms={arm:{r['id']:r for r in records if r['arm']==arm and r['group']==group} for arm in ('base','tuned')}
            for key,condition in [('gained',lambda a,b:not a and b),('regressed',lambda a,b:a and not b),('both_failed',lambda a,b:not a and not b)]:
                expected=[k for k in arms['base'] if condition(arms['base'][k]['passed'],arms['tuned'][k]['passed'])]
                assert expected==comparison['groups'][group][key]
        aggregate={}
        for arm in ('base','tuned'):
            rows=[r for r in records if r['arm']==arm]
            models=[r for r in rows if not r['preflight']]
            warm=[r for r in models if r['id']!='R01']
            aggregate[arm]={'passed':sum(r['passed'] for r in rows),'total':len(rows),
                'model_passed':sum(r['passed'] for r in models),'model_total':len(models),
                'calls':sum(r['model_calls'] for r in rows),'error_actions':sum(r['error_actions'] for r in rows),
                'warm_model_questions':len(warm),'warm_median_seconds':statistics.median(r['seconds'] for r in warm),
                'warm_total_seconds':sum(r['seconds'] for r in warm),
                'note':'All 33 warm model questions, including failed tasks; not a successful-answer-only speed claim.'}
        train_summary=next(r for r in reversed(state['log_history']) if 'train_runtime' in r)
        paired={arm:{(r['group'],r['id']):r for r in records if r['arm']==arm} for arm in ('base','tuned')}
        common=[key for key,row in paired['base'].items() if row['passed'] and paired['tuned'][key]['passed']
                and not row['preflight'] and row['id']!='R01']
        common_timing={arm:{'median_seconds':statistics.median(paired[arm][k]['seconds'] for k in common),
                           'total_seconds':sum(paired[arm][k]['seconds'] for k in common)} for arm in paired}
        result={'archive':path.name,'sha256':archive_hash,'verified_export_hashes':len(manifest),
            'verified_source_files':len(pin['files']),'verified_dependencies':len(pin['dependencies']),
            'scores_recomputed':len(records),'numeric_rows_recomputed':numeric_count,'source_records_checked':sources_count,
            'quotes_checked_against_local_pages':quotes_count,'training':{'steps':144,'epochs':2,'world_size':load('full/provenance.json')['world_size'],**train_summary},
            'merge':{'updated_tensors':248,'preserved_mtp_tensors':15},'aggregate':aggregate,
            'common_passed_warm_timing':{'questions':len(common),'arms':common_timing,
                'note':'Post-hoc common-passed subset, single paired run, excludes first cold query. Not fixed-token throughput.'},
            'comparison':comparison,'records':records,
            'limits':['Development diagnostics, not an independent test.','Adapter and merged tensors are NOT in this ZIP; completion and tensor checks are server receipts, not local weight revalidation.',
                      'Frozen-score pass/fail can hide content regressions: inspect R02 and N13.','Recomputation uses pinned deterministic engine and archived facts/pages, not a new PDF audit.']}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.archive);save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','comparison')},ensure_ascii=False,indent=2))
