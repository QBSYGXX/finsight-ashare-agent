"""Run explicit scripted tool trajectories against real validation documents; NOT model accuracy."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent_v015.agent import answer,Session,context
from agent_v015.evaluation import CASES,score
from agent_v015.retrieval import Index,build
from agent_v015.storage import save,read


class Replay:
    def __init__(self,case):
        self.case,self.actions,self.position=case,["plan"],0
        if case.get("gold_pages"):
            self.actions += ["search","page"]
        self.metrics=list(case.get("numeric",{}))+list(case.get("row_status",{}))
        self.actions += self.metrics+["finish"]
        self.hits=[]
        self.segments=[]
        self.last_generation={"test_double":True,"name":"explicit scripted tool replay"}
    def generate(self,messages):
        action=self.actions[self.position]
        self.position+=1
        if action=="plan":
            requests=[]
            for metric_id in self.metrics:
                year,metric,transform=metric_id.split(":")
                requests.append({"metric":metric,"year":int(year),"transform":transform})
            obj={"tool":"plan","arguments":{"requests":requests,"narrative":bool(self.case.get("gold_pages"))}}
        elif action=="search":
            obj={"tool":"report_search","arguments":{"query":self.case["query"],"year":self.case["year"],"top_k":6}}
        elif action=="page":
            observation=json.loads(messages[-1]["content"].split("\n",1)[1])
            self.hits=observation["hits"]
            hit=next(h for h in self.hits if h["page"] in self.case["gold_pages"])
            obj={"tool":"read_page","arguments":{"evidence_id":hit["id"]}}
        elif action=="finish":
            quotes=[]
            if self.case.get("gold_pages"):
                for message in messages:
                    if message["content"].startswith("TOOL_DATA"):
                        observation=json.loads(message["content"].split("\n",1)[1])
                        self.segments+=observation.get("segments",[])
                segment=next(s for s in self.segments if all(a in s["text"] for a in self.case["anchors"]))
                # Pick an exact bounded window containing the anchor terms.
                positions=[segment["text"].find(a) for a in self.case["anchors"]]
                start=max(0,min(positions)-100)
                quote=segment["text"][start:start+1100]
                quotes=[{"evidence_id":segment["id"],"quote":quote}]
            obj={"tool":"finish","arguments":{"quotes":quotes,"result_ids":self.metrics}}
        else:
            year,metric,transform=action.split(":")
            obj={"tool":"financial_metric","arguments":{"year":int(year),"metric":metric,"transform":transform}}
        return json.dumps(obj,ensure_ascii=False)


def main():
    index=Index(ROOT,build(backend="bm25"))
    records=[]
    for case in CASES:
        result=answer(case["question"],ROOT,index,Replay(case))
        verdict=score(case,result)
        records.append({"case":case["id"],"verdict":verdict,"status":result["status"]})
        print(case["id"],verdict,flush=True)
        assert verdict["passed"],result
    save(ROOT/"outputs/agent_v015/local-scripted-replay.json",{
        "test_type":"scripted_model_replay_not_model_accuracy","cases":records,"actual_model_run":False})
    print("SCRIPTED_REAL_CORPUS_REPLAY_PASS=13; REAL_LLM_NOT_RUN",flush=True)
    action=read(ROOT/"agent_v015/quote_regression.json")["action"]
    ctx=context("中兴通讯2024年毛利率和客户信用风险",read(ROOT/"configs/project.json"),index)
    session=Session(ROOT,index,ctx)
    session.plan([{"metric":"gross_margin","year":2024}],narrative=True)
    ids={q["evidence_id"] for q in action["arguments"]["quotes"]}
    session.evidence={c["id"]:c for c in index.chunks if c["id"] in ids}
    assert set(session.evidence)==ids
    session.financial_metric("gross_margin",2024)
    result=session.call(action["tool"],action["arguments"])
    assert result["status"]=="completed" and result["results"][0]["value"]=="37.9127"
    for quote in result["quotes"]:
        assert quote["match_mode"]=="whitespace_restored"
        original=session.evidence[quote["source"]["id"]]["text"]
        start,end=quote["source_span"]
        assert quote["quote"]==original[start:end]
    save(ROOT/"outputs/agent_v015/server-step3-quote-replay.json",{
        "test_type":"observed_server_action_replay_not_new_model_run","result":result})
    print("SERVER_STEP3_REPLAY_PASS=3_QUOTES; GROSS_MARGIN=37.9127%; REAL_LLM_NOT_RERUN",flush=True)


if __name__=="__main__":main()
