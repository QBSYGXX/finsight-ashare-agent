"""模块：服务器动作回放；作用：复现协议修复并保留语义失败，不声称新模型准确率；核心业务：否。"""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_v015.agent import Session,answer,context
from agent_v015.action_contract import parse_envelope
from agent_v015.excerpts import annotate
from agent_v015.evaluation import CASES,score as score_old
from agent_v015.natural_evaluation import score as score_natural
from agent_v015.retrieval import Index
from agent_v015.storage import read,save

ARCHIVE=ROOT.parent/'_release_work/finsight-agent-v015-20260928-165930-9842418f.zip'
EXPECTED='bd056900a1e1936a68458585f20614432f303a936a0c58c4cde27bd435730d34'


def main():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==EXPECTED
    index=Index(ROOT,ROOT/'outputs/agent_v015/index/validation-bm25')
    config=read(ROOT/'configs/project.json')
    spec=read(ROOT/'agent_v015/natural_v1.json')
    records=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('EXPORT_MANIFEST.json'))
        assert set(z.namelist())==set(manifest)|{'EXPORT_MANIFEST.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==v for n,v in manifest.items())
        for name in ['diagnostic/R04','natural/N15','natural/N16','natural/N18']:
            trace=json.loads(z.read(name+'.json'))
            session=Session(ROOT,index,context(trace['question'],config,index))
            for position,event in enumerate(trace['events']):
                raw_action=event.get('action',{})
                if raw_action.get('tool')=='finish' and raw_action['arguments'].get('selections'):
                    action,notes=parse_envelope(event['model_output'])
                    result=session.call(action['tool'],action['arguments'])
                    result={**result,'events':trace['events'][:position]}
                    case=next(c for c in (CASES if name.startswith('diagnostic') else spec['cases']) if c['id']==name.split('/')[1])
                    verdict=score_old(case,result) if name.startswith('diagnostic') else score_natural(case,result,spec)
                    assert verdict['passed'],(name,result)
                    for q in result['quotes']:
                        page=index.pages[(q['source']['report_id'],q['source']['page'])]
                        start,end=q['page_span'];assert page['text'][start:end]==q['quote']
                    records.append({'case':name,'test':'original_server_selection_with_verified_prior_observations',
                                    'normalization':notes,'verdict':verdict,'quotes':result['quotes']})
                    print('ORIGINAL_SELECTION_ACCEPTED='+name,flush=True)
                    break
                if 'observation' not in event:continue
                if raw_action['tool'] in {'plan','financial_metric'}:
                    session.call(raw_action['tool'],raw_action['arguments'])
                elif raw_action['tool'] in {'report_search','read_page'}:
                    obs=event['observation']
                    for source in obs.get('hits',[])+obs.get('segments',[]):
                        page=index.pages[(source['report_id'],source['page'])]
                        assert source['pdf_sha256']==page['pdf_sha256']
                        assert page['text'][source['start']:source['end']]==source['text']
                        checked=annotate(source,page)
                        assert checked['sentences']==source['sentences']
                        session.evidence[source['id']]=checked
            else:raise AssertionError('Missing original selection action')

        for name in ['diagnostic/M01','natural/N11']:
            trace=json.loads(z.read(name+'.json'))
            action,notes=parse_envelope(trace['events'][0]['model_output'])
            session=Session(ROOT,index,context(trace['question'],config,index))
            result=session.call(action['tool'],action['arguments'])
            assert result['status']=='planned'
            records.append({'case':name,'test':'original_plan_envelope_only_not_followup_model_run','normalization':notes})
            print('ORIGINAL_PLAN_ACCEPTED='+name,flush=True)

        trace=json.loads(z.read('natural/N01.json'))
        class RecordedOutputs:
            def __init__(self):self.outputs=iter(e['model_output'] for e in trace['events'])
            def generate(self,messages):return next(self.outputs)
        result=answer(trace['question'],ROOT,index,RecordedOutputs())
        case=next(c for c in spec['cases'] if c['id']=='N01')
        verdict=score_natural(case,result,spec)
        assert result['status']=='completed' and len(result['events'])==7
        assert not verdict['passed'] and not verdict['checks']['exact_targets']
        records.append({'case':'natural/N01','test':'original_output_sequence_retry_fixed_but_extra_targets_still_fail',
                        'verdict':verdict,'result':result})
        print('N01_FINISH_RETRY_COMPLETED; SEMANTIC_VERDICT_STILL_FAILED=true',flush=True)

        trace=json.loads(z.read('natural/N07.json'))
        action,_=parse_envelope(trace['events'][0]['model_output'])
        assert action['arguments']['requests'][1]=={'metric':'gross_margin','year':2024,'transform':'value'}
        session=Session(ROOT,index,context(trace['question'],config,index))
        try:session.call(action['tool'],action['arguments'])
        except ValueError as error:assert 'Year' in str(error)
        else:raise AssertionError('Invalid target year must still be rejected')
        print('N07_WRONG_YEAR_NOT_SILENTLY_REPAIRED=true',flush=True)
    save(ROOT/'outputs/agent_v015/v0155-server-action-replay.json',{
        'test_type':'recorded_server_actions_and_verified_observations_not_new_model_accuracy',
        'archive_sha256':EXPECTED,'actual_model_run':False,'records':records})
    print('SERVER_ACTION_REPLAY_PASS; ACTUAL_MODEL_RUN=false',flush=True)


if __name__=='__main__':main()
