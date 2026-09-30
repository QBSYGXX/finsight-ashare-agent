"""模块：实验验收测试；作用：验证隔离、标签、恢复示范及对照归因边界；核心业务：否。"""
import json
from pathlib import Path
import pytest

from .common import ADDON,ROOT,read,save,run_path
from .corpus import blocked
from .tokenization import encode_sample
from .train import language_targets,probe_rows
from .merge import adapter_pairs
from .evaluate import compare
from agent_v015.action_contract import parse_envelope


class CharacterTokenizer:
    def apply_chat_template(self,messages,**kw):
        assert kw=={'tokenize':False,'add_generation_prompt':True,'enable_thinking':False}
        return ''.join(m['content'] for m in messages)+'<|im_start|>assistant\n<think>\n\n</think>\n\n'
    def encode(self,text,add_special_tokens=False):return list(text.encode('utf-8'))
    def decode(self,ids,skip_special_tokens=False):return bytes(ids).decode('utf-8')
    def convert_tokens_to_ids(self,text):return text.encode()[-1]


@pytest.fixture
def row():
    return {'id':'sample','category':'numeric','messages':[{'role':'system','content':'规则'},
        {'role':'user','content':'问题'}],'target':'{"tool":"finish","arguments":{"selections":[]}}'}


def test_only_suffix_supervised(row):
    encoded=encode_sample(CharacterTokenizer(),row,{'max_length':10000,'max_action_tokens':1000})
    labels=encoded['labels'];prefix=labels.index(next(x for x in labels if x!=-100))
    assert all(x==-100 for x in labels[:prefix])
    assert labels[prefix:]==encoded['input_ids'][prefix:]
    assert bytes(labels[prefix:]).decode()==row['target']+'<|im_end|>'


def test_no_silent_truncation(row):
    with pytest.raises(ValueError,match='No truncation'):
        encode_sample(CharacterTokenizer(),row,{'max_length':5,'max_action_tokens':1000})


def test_no_shorthand_target(row):
    row['target']='{"finish":{"selections":[]}}'
    with pytest.raises(ValueError,match='canonical'):
        encode_sample(CharacterTokenizer(),row,{'max_length':10000,'max_action_tokens':1000})


def test_prefix_boundary_rejected(row):
    class Bad(CharacterTokenizer):
        def encode(self,text,add_special_tokens=False):
            ids=super().encode(text)
            if text.endswith('<|im_end|>'):ids[0]=0
            return ids
    with pytest.raises(ValueError,match='boundary'):
        encode_sample(Bad(),row,{'max_length':10000,'max_action_tokens':1000})


def test_native_template_rejected(row):
    class Wrong(CharacterTokenizer):
        def apply_chat_template(self,*args,**kwargs):return 'old nonthinking format'
    with pytest.raises(ValueError,match='template'):
        encode_sample(Wrong(),row,{'max_length':10000,'max_action_tokens':1000})


def test_language_layers_only():
    names=['model.language_model.layers.0.linear_attn.in_proj_qkv','model.visual.blocks.0.qkv',
           'mtp.layers.0.self_attn.q_proj','lm_head','model.language_model.layers.3.self_attn.q_proj']
    assert language_targets(names)==sorted([names[0],names[-1]])


def test_merge_inventory():
    module='model.language_model.layers.3.self_attn.q_proj'
    a='base_model.model.'+module+'.lora_A.weight';b=a.replace('lora_A','lora_B')
    assert adapter_pairs({a,b},{module+'.weight'})=={module+'.weight':(a,b)}
    with pytest.raises(ValueError,match='Missing'):adapter_pairs({a},{module+'.weight'})
    with pytest.raises(ValueError,match='Unexpected'):adapter_pairs({a,b,'lm_head.weight'},{module+'.weight'})
    with pytest.raises(ValueError,match='outside'):
        adapter_pairs({a.replace(module,'mtp.layers.0.q_proj'),b.replace(module,'mtp.layers.0.q_proj')},{'mtp.layers.0.q_proj.weight'})


@pytest.mark.parametrize('value',['../old','a/b','a\\b','', '.',None])
def test_run_path_bounds(value):
    with pytest.raises(ValueError):run_path(value)


def test_corpus_frozen_and_train_only():
    rows=[json.loads(line) for line in (ADDON/'data/train.jsonl').open(encoding='utf-8')]
    audit=read(ADDON/'data/audit.json')
    companies=read(ROOT/'configs/project.json')['companies']
    train={c['code'] for c in companies if c['split']=='train'}
    assert len(rows)==audit['actions'] and len({r['id'] for r in rows})==len(rows)
    assert {r['company'] for r in rows}==train
    for row in rows:
        assert not blocked(json.dumps(row,ensure_ascii=False),companies)
        assert parse_envelope(row['target'])[1]==[]
        assert row['messages'][0]['role']=='system' and row['messages'][-1]['role']=='user'
    recovery=[r for r in rows if r['category']=='recovery']
    assert len({r['trajectory_id'] for r in recovery})==8
    assert all(r['step']!=2 for r in recovery) # injected premature finish is never supervised
    assert any('工具或格式错误' in r['messages'][-1]['content'] for r in recovery)
    assert {'plan','financial_metric','report_search','read_page','finish'}=={json.loads(r['target'])['tool'] for r in rows}


def test_probe_covers_longest_each_category():
    rows=[{'id':str(i),'category':str(i%4),'input_ids':list(range(i+1))} for i in range(80)]
    selected=probe_rows(rows)
    assert len(selected)==32 and {r['id'] for r in selected}>={'76','77','78','79'}


def test_comparison_preserves_regressions(tmp_path):
    for arm,passed in [('base',[True,False]),('tuned',[False,True])]:
        group={'diagnostic_pass':{'passed':1,'total':2},'model_invoking_pass':{'passed':1,'total':2},
               'preflight_pass':{'passed':0,'total':0},
               'records':[{'id':str(i),'verdict':{'passed':p}} for i,p in enumerate(passed)]}
        save(tmp_path/'evaluation'/arm/'summary.json',{'diagnostic':group,'natural':group})
        save(tmp_path/'evaluation'/arm/'run_config.json',dict.fromkeys(['experiment_digest','backend','thinking','seed','max_tokens','index'],'same'))
    compare(tmp_path)
    assert read(tmp_path/'comparison.json')['groups']['natural']['regressed']==['0']
    assert read(tmp_path/'comparison.json')['groups']['natural']['gained']==['1']
    path=tmp_path/'evaluation/tuned/run_config.json';config=read(path);config['seed']='different';save(path,config)
    with pytest.raises(ValueError,match='Unfair comparison'):compare(tmp_path)
