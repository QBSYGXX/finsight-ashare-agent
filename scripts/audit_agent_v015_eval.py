"""Audit the returned v0.15.2 diagnostic ZIP; replay protocol fixes without any LLM."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_v015.agent import Session,context
from agent_v015.evaluation import CASES,score
from agent_v015.retrieval import Index
from agent_v015.storage import read,save,sha

EXPECTED="bf2a721e92d9af5fa4812140692a32b4bd036959fba585d76883a2d5a2488cf8"
ARCHIVE=ROOT.parent/"_release_work/finsight-agent-v015-20260928-112612-ddc06fea.zip"


def main():
    assert sha(ARCHIVE)==EXPECTED
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read("EXPORT_MANIFEST.json"))
        assert set(z.namelist())==set(manifest)|{"EXPORT_MANIFEST.json"}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
        config=json.loads(z.read("run_config.json"))
        # Cases/scoring must remain byte-identical to those used by this server run.
        assert sha(ROOT/"agent_v015/evaluation.py")==config["package"]["files"]["agent_v015/evaluation.py"]
        original=json.loads(z.read("summary.json"))
        traces={c["id"]:json.loads(z.read(c["id"]+".json")) for c in CASES}
    records=[]
    for case in CASES:
        trace=traces[case["id"]]
        assert trace["question"]==case["question"]
        verdict=score(case,trace)
        old=next(r for r in original["records"] if r["id"]==case["id"])
        assert verdict==old["verdict"]
        records.append({"id":case["id"],"passed":verdict["passed"],"preflight":bool(case.get("preflight")),
                        "status":trace["status"],"model_calls":len(trace["events"]),"seconds":trace["seconds"]})
    model_rows=[r for r in records if not r["preflight"]]
    preflight=[r for r in records if r["preflight"]]
    assert sum(r["passed"] for r in records)==9 and len(model_rows)==10 and sum(r["passed"] for r in model_rows)==6
    rates=[e["generation"]["tokens_per_second"] for trace in traces.values() for e in trace["events"] if e.get("generation")]
    audit={"archive":ARCHIVE.name,"sha256":EXPECTED,"member_hashes_verified":len(manifest),
           "scores_recomputed":13,"diagnostic_pass":"9/13","model_invoking_pass":"6/10","preflight_pass":"3/3",
           "median_generation_tokens_per_second":statistics.median(rates),"records":records,
           "limitations":"Validation diagnostics, not blind accuracy. Generation timings include prefill and exclude model initialization."}
    save(ROOT/"docs/experiments/agent-v0152-server-audit-20260928.json",audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))

    # Explicit reviewed intents are inserted for the NEW plan contract. This tests
    # bookkeeping/sources, not how a real model would generate that new plan.
    index=Index(ROOT,ROOT/"outputs/agent_v015/index/validation-bm25")
    chunks={c["id"]:c for c in index.chunks}
    replay=[]
    for case_id in ["M02","M03","M04","B04"]:
        case=next(c for c in CASES if c["id"]==case_id)
        session=Session(ROOT,index,context(case["question"],read(ROOT/"configs/project.json"),index))
        requested=list(case.get("numeric",{}))+list(case.get("row_status",{}))
        requests=[]
        for target in requested:
            year,metric,transform=target.split(":")
            requests.append({"metric":metric,"year":int(year),"transform":transform})
        session.call("plan",{"requests":requests,"narrative":False})
        for request in requests:session.call("financial_metric",request)
        if case_id=="M02":
            try:session.call("financial_metric",{"metric":"parent_net_profit","year":2024})
            except ValueError:pass
            else:raise AssertionError("Unrequested substitute metric accepted")
            result=session.call("finish",{})
        else:
            # Reuse the exact first finish action which failed on the server.
            action=next(e["action"] for e in traces[case_id]["events"] if e.get("action",{}).get("tool")=="finish")
            for evidence_id in action["arguments"].get("evidence_ids",[]):
                if evidence_id in chunks:
                    session.evidence[evidence_id]=chunks[evidence_id]
            result=session.call("finish",action["arguments"])
        verdict=score(case,result)
        assert verdict["passed"],(case_id,result)
        replay.append({"id":case_id,"passed":True,"status":result["status"]})
    save(ROOT/"outputs/agent_v015/v0153-failure-replay.json",{
        "test_type":"new_contract_with_explicit_reference_intents_and_old_finish_actions_not_model_accuracy","records":replay})
    print("NEW_CONTRACT_FAILURE_REPLAY_PASS=4; NEW_MODEL_NOT_RUN")


if __name__=="__main__":main()
