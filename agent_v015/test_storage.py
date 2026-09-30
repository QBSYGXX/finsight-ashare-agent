"""Cross-platform data hashes must tolerate only physical line-ending differences."""
import hashlib
import pytest

from agent_v015.storage import save,sha,sha_lf,verify


@pytest.fixture
def pinned(tmp_path):
    data=tmp_path/"data/verified/facts.json"
    data.parent.mkdir(parents=True)
    data.write_bytes(b'[\r\n  {"value": "123", "evidence": "escaped \\r\\n preserved"}\r\n]\r\n')
    code=tmp_path/"agent_v015/example.py"
    code.parent.mkdir()
    code.write_bytes(b"a = 1\n")
    save(tmp_path/"agent_v015/PACKAGE_MANIFEST.json",{
        "files":{"agent_v015/example.py":sha(code)},
        "dependencies":{"data/verified/facts.json":sha(data)},
        "dependency_lf_sha256":{"data/verified/facts.json":sha_lf(data)}})
    return tmp_path,data,code


def test_crlf_and_lf_both_valid_without_rewriting(pinned):
    root,data,_=pinned
    original=data.read_bytes()
    verify(root)
    assert data.read_bytes()==original
    data.write_bytes(original.replace(b"\r\n",b"\n"))
    lf=data.read_bytes()
    verify(root)
    assert data.read_bytes()==lf


@pytest.mark.parametrize("old,new",[(b"123",b"124"),(b"value",b"other"),(b"escaped",b"changed"),(b"\\r\\n",b"\\n")])
def test_changed_financial_content_rejected(pinned,old,new):
    root,data,_=pinned
    data.write_bytes(data.read_bytes().replace(b"\r\n",b"\n").replace(old,new))
    with pytest.raises(ValueError,match="数据内容校验失败"):
        verify(root)


def test_code_still_requires_exact_hash(pinned):
    root,_,code=pinned
    code.write_bytes(b"a = 1\r\n")
    with pytest.raises(ValueError,match="文件校验失败"):
        verify(root)
