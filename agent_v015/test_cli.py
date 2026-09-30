"""模块：评测交付测试；作用：确保两组共用模型、失败停止与导出一致；核心业务：否。"""
import hashlib
import json
import sys
from types import SimpleNamespace
import zipfile

import pytest

from agent_v015 import cli,evaluation,natural_evaluation,retrieval


@pytest.mark.parametrize('failed',[False,True])
def test_combined_evaluation_exports_exact_run_and_stops_on_failure(tmp_path,monkeypatch,failed):
    calls=[]
    model=object()
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'verify',lambda:{'test_double':True})
    monkeypatch.setattr(cli,'LazyModel',lambda args:model)
    monkeypatch.setattr(retrieval,'Index',lambda *args:SimpleNamespace(meta={'backend':'synthetic'}))
    def baseline(root,index,actual_model,dest,retrieval_only):
        assert actual_model is model
        calls.append('diagnostic')
        result={'execution_status':'failed' if failed else 'completed','diagnostic_pass':{'passed':0,'total':1}}
        cli.save(dest/'summary.json',result)
        return result
    def natural(root,index,actual_model,dest,retrieval_only):
        assert actual_model is model
        calls.append('natural')
        result={'execution_status':'completed','diagnostic_pass':{'passed':1,'total':1}}
        cli.save(dest/'summary.json',result)
        return result
    monkeypatch.setattr(evaluation,'run',baseline)
    monkeypatch.setattr(natural_evaluation,'run',natural)
    monkeypatch.setattr(sys,'argv',['cli','evaluate','--suite','all','--export-after'])
    if failed:
        with pytest.raises(RuntimeError,match='runtime error'):cli.main()
    else:cli.main()
    assert calls==(['diagnostic'] if failed else ['diagnostic','natural'])
    archive=next((tmp_path/'exports').glob('*.zip'))
    assert archive.with_suffix('.zip.sha256').read_text().split()[0]==hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('EXPORT_MANIFEST.json'))
        assert set(z.namelist())==set(manifest)|{'EXPORT_MANIFEST.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==v for n,v in manifest.items())
        assert ('natural/summary.json' in z.namelist()) is not failed
        assert json.loads(z.read('completed.json'))['status']==('failed' if failed else 'completed')
    with pytest.raises(ValueError,match='already exists'):
        cli.export_run(next((tmp_path/'outputs/agent_v015/runs').iterdir()))
