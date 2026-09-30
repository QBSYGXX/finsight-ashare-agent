"""Regression gates for removed arguments, ratio semantics and immutable outputs."""
import ast
from dataclasses import make_dataclass
import math
from pathlib import Path
import zipfile
import pytest
from .common import ROOT, ADDON, read
from .train import training_arguments, require_fresh_output


def real_api_shape():
    wheels=list((ROOT/'.cache/vllm-offline/wheelhouse-v013').glob('transformers-5.8.0-*.whl'))
    if not wheels:
        pytest.skip('Pinned Transformers wheel unavailable; server validates actual constructor')
    with zipfile.ZipFile(wheels[0]) as z:
        tree=ast.parse(z.read('transformers/training_args.py'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TrainingArguments')
    fields=[(n.target.id,object,None) for n in cls.body if isinstance(n,ast.AnnAssign)
            and isinstance(n.target,ast.Name)]
    api=make_dataclass('Actual580Fields',fields)
    # Execute the actual upstream method, without importing Torch on Windows.
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_warmup_steps')
    ns={'math':math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'upstream_warmup','exec'),ns)
    api.get_warmup_steps=ns['get_warmup_steps']
    return api


@pytest.mark.parametrize('world',[1,2,4])
@pytest.mark.parametrize('probe',[True,False])
def test_actual_580_fields_and_ratio(world,probe,tmp_path):
    config=read(ADDON/'config.json')
    args=training_arguments(real_api_shape(),config,tmp_path/'new',world,probe)
    assert args.gradient_accumulation_steps*world==32
    assert args.max_steps==(20 if probe else -1)
    assert args.get_warmup_steps(143)==8
    assert not hasattr(args,'overwrite_output_dir')
    assert args.warmup_ratio is None


def test_rejects_incompatible_api_instead_of_dropping_parameters(tmp_path):
    class Missing:
        def __init__(self,output_dir):
            raise AssertionError('Must reject signature before instantiation')
    with pytest.raises(TypeError):
        training_arguments(Missing,read(ADDON/'config.json'),tmp_path,1,True)


def test_existing_training_output_stays_untouched(tmp_path):
    output=tmp_path/'probe';output.mkdir()
    saved=output/'evidence.json';saved.write_text('original')
    with pytest.raises(FileExistsError):require_fresh_output(output)
    assert saved.read_text()=='original'
    require_fresh_output(tmp_path/'fresh')


def test_training_data_and_recipe_unchanged():
    for name in ('config.json','data/train.jsonl','data/trajectories.json','data/audit.json'):
        assert (ADDON/name).read_bytes()==(ROOT/'lora_v016'/name).read_bytes()
