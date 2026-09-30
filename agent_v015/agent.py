"""模块：Agent 决策循环；作用：受控调用工具、保留证据、组装可核验答案；核心业务：是。"""
import json
import re
import time
import unicodedata

from .finance import MetricEngine
from .excerpts import annotate, select
from .action_contract import ActionGuard, finish_arguments, parse_envelope, prompt
from .storage import read


class RequestError(ValueError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def context(question, config, index):
    text = unicodedata.normalize("NFKC", question).strip()
    if not text or len(text)>2000:
        raise RequestError("needs_clarification", "请提供不超过2000字的财报问题")
    if re.search(r"半年|半年度|季度|季报|一季|三季|中报|[Hh][12]|[Qq][1-4]|TTM|滚动|月度",text):
        raise RequestError("unsupported_period", "当前知识库仅含年度报告；不使用全年数代替季度或半年数")
    matched = [c for c in config["companies"] if c["name"] in text or re.search(r"(?<!\d)"+c["code"]+r"(?!\d)",text)]
    if len(matched)!=1:
        raise RequestError("needs_clarification", "请指定一家公司的完整证券简称或六位代码")
    company = matched[0]
    if company["split"] == "test":
        raise RequestError("reserved_test", "该公司属于尚未开放的独立测试集")
    if company["code"] not in index.meta["companies"]:
        raise RequestError("outside_index", "当前索引未包含该公司；可构建development索引后查询")
    years = sorted(set(map(int,re.findall(r"(?<!\d)(20\d{2})(?!\d)",text))))
    span = re.search(r"(20\d{2})\s*年?\s*[-—~至到]\s*(20\d{2})",text)
    if span:
        start,end = map(int,span.groups())
        if start>end or end-start>11:
            raise RequestError("needs_clarification", "年份区间最多12年，且开始年份不得晚于结束年份")
        years = sorted(set(years)|set(range(start,end+1)))
    if not years and re.search(r"历年|所有.*年份|全部.*年份",text):
        years = sorted({p["report_year"] for p in index.pages.values() if p["company"]==company["code"]})
    if not years or len(years)>12:
        raise RequestError("needs_clarification", "请提供年度、年份区间或说明查询所有可用年份")
    return {"company":company["code"],"name":company["name"],"years":years,
            "scope":"parent" if "母公司口径" in text or "母公司报表" in text else "consolidated"}


def parse_action(text):
    return parse_envelope(text)[0]


# 模块：原文引用对齐；作用：忽略排版空白查找，再从证据还原原始文本；核心业务：是。
def source_quote(text, quote):
    if not isinstance(quote,str) or not 12<=len(quote)<=1200:
        raise ValueError("Quote must contain 12..1200 characters")
    if not quote.strip():
        raise ValueError("Empty quote")
    start=text.find(quote)
    if start>=0:
        return quote,start,start+len(quote),"exact"
    positions=[i for i,c in enumerate(text) if not c.isspace()]
    compact="".join(text[i] for i in positions)
    target="".join(c for c in quote if not c.isspace())
    start=compact.find(target)
    if start<0 or compact.find(target,start+1)>=0:
        raise ValueError("Quote text differs from evidence or has ambiguous matches; use evidence_ids to select the observed excerpt")
    first,last=positions[start],positions[start+len(target)-1]+1
    # Return the source substring, never the model's potentially reformatted number/string.
    return text[first:last],first,last,"whitespace_restored"


class Session:
    def __init__(self, root, index, ctx):
        self.index, self.ctx = index, ctx
        facts=read(root/"data/verified/facts.json")
        self.finance = MetricEngine(facts)
        self.fact_sources={f["id"]:f for f in facts}
        self.evidence, self.rows = {}, {}
        self.task_plan=None

    # 模块：任务清单；作用：锁定本题目标，程序负责保留成功/缺失结果；核心业务：是。
    def plan(self, requests, narrative=False):
        if self.task_plan is not None:
            raise ValueError("Plan is already registered; do not change targets after tool execution")
        if not isinstance(requests,list) or len(requests)>24 or type(narrative) is not bool:
            raise ValueError("plan requires up to 24 requests and a boolean narrative")
        normalized=[]
        ids=[]
        for request in requests:
            if not isinstance(request,dict) or set(request)-{"metric","year","transform"}:
                raise ValueError("Each request accepts only metric, year, transform")
            metric,year,transform=request.get("metric"),request.get("year"),request.get("transform","value")
            self.year(year)
            if not isinstance(metric,str) or metric not in self.finance.registry or transform not in {"value","yoy_growth","yoy_difference","yoy_pp"}:
                raise ValueError("Plan uses unknown metric or transform")
            result_id=f"{year}:{metric}:{transform}"
            if result_id in ids:raise ValueError("Duplicate planned request")
            ids.append(result_id)
            normalized.append({"metric":metric,"year":year,"transform":transform})
        self.task_plan={"requests":normalized,"narrative":narrative,"result_ids":ids}
        return {"status":"planned","plan":self.task_plan,"task_state":self.task_state()}

    def task_state(self):
        if self.task_plan is None:return {"next":"Call plan first"}
        pending=[r for r in self.task_plan["requests"] if f"{r['year']}:{r['metric']}:{r['transform']}" not in self.rows]
        return {"pending_requests":pending,"resolved_results":{k:r["status"] for k,r in self.rows.items()},
                "narrative_requested":self.task_plan["narrative"],
                "guidance":"missing_fields等状态是当前已核验快照的终态；勿用其他指标或原文数字替代。继续其他pending_requests；全部完成且已取得所需叙述证据后调用finish。finish自动保留全部目标结果，无需填写result_ids。"}

    def year(self,year):
        if type(year) is not int or year not in self.ctx["years"]:
            raise ValueError("Year must belong to the user request")

    def report_search(self, query, year, top_k=4):
        self.year(year)
        available=sorted({p["report_year"] for p in self.index.pages.values() if p["company"]==self.ctx["company"]})
        if year not in available:
            return {"status":"outside_report_coverage","hits":[],"available_report_years":available,
                    "retry_query_useful":False,"message":"该公司在此索引中没有该年度报告，换检索词不会补出报告；这不表示公司未披露。"}
        hits = [annotate(hit,self.index.pages[(hit["report_id"],hit["page"])])
                for hit in self.index.search(query,self.ctx["company"],[year],top_k)]
        for hit in hits:
            self.evidence[hit["id"]] = hit
        return {"status":"ok" if hits else "no_hits", "hits":hits}

    def read_page(self, evidence_id):
        if evidence_id not in self.evidence:
            raise ValueError("Read a page only through an already retrieved evidence_id")
        hit = self.evidence[evidence_id]
        page = self.index.pages[(hit["report_id"],hit["page"])]
        # Full pages can contain huge flattened tables. Expose bounded exact-text segments.
        segments = []
        for start in range(0,min(len(page["text"]),12000),1500):
            row = annotate({**page,"id":f"{hit['report_id']}:p{hit['page']}:s{start}",
                            "start":start,"end":min(start+1500,len(page["text"])),
                            "text":page["text"][start:start+1500]},page)
            self.evidence[row["id"]] = row
            segments.append(row)
        return {"status":"ok", "segments":segments,"page_truncated":len(page["text"])>12000}

    def financial_metric(self, metric, year, transform="value", scope=None):
        self.year(year)
        if scope is not None and scope!=self.ctx["scope"]:
            raise ValueError("Scope cannot differ from the user request")
        if not isinstance(metric,str) or metric not in self.finance.registry:
            raise ValueError("Metric is not registered")
        if transform not in {"value","yoy_growth","yoy_difference","yoy_pp"}:
            raise ValueError("Unknown transform")
        result_id=f"{year}:{metric}:{transform}"
        if self.task_plan is not None and result_id not in self.task_plan["result_ids"]:
            raise ValueError("Metric is outside the registered user targets; do not substitute another metric")
        result = self.finance.execute(self.ctx["company"],year,self.ctx["scope"],[{"metric":metric,"transform":transform}])
        row = result["results"][0]
        row["result_id"] = f"{year}:{metric}:{transform}"
        self.rows[row["result_id"]] = row
        for source in row.get("sources",[]):
            fact=self.fact_sources[source["id"]]
            self.evidence[fact["id"]]={**fact,"text":fact["evidence"],"evidence_kind":"audited_fact"}
        result["task_state"]=self.task_state()
        return result

    def finish(self, quotes=None, result_ids=None, abstain=False, evidence_ids=None, selections=None):
        if quotes is None: quotes=[]
        if result_ids is None: result_ids=[]
        if evidence_ids is None: evidence_ids=[]
        if selections is None: selections=[]
        normalized,normalization=finish_arguments({'quotes':quotes,'evidence_ids':evidence_ids,'selections':selections})
        evidence_ids=normalized['evidence_ids']
        if not isinstance(quotes,list) or not isinstance(evidence_ids,list) or not isinstance(selections,list) or not isinstance(result_ids,list) or len(quotes)+len(evidence_ids)+len(selections)>8 or len(result_ids)>36 or type(abstain) is not bool:
            raise ValueError("Invalid final selection")
        if sum(bool(items) for items in [quotes,evidence_ids,selections])>1:
            raise ValueError("Use only one of selections, quotes or evidence_ids")
        validated=[]
        for item in selections:
            if not isinstance(item,dict) or set(item)!={"evidence_id","first_sentence","last_sentence"}:
                raise ValueError("Each selection requires evidence_id, first_sentence, last_sentence")
            evidence_id=item["evidence_id"]
            if not isinstance(evidence_id,str) or evidence_id not in self.evidence:
                raise ValueError("Unknown evidence_id; select only evidence observed in this question")
            validated.append(select(self.evidence[evidence_id],item["first_sentence"],item["last_sentence"]))
        for evidence_id in evidence_ids:
            if not isinstance(evidence_id,str) or evidence_id not in self.evidence:
                raise ValueError("Unknown evidence_id; select only evidence observed in this question")
            source=self.evidence[evidence_id]
            validated.append({"quote":source["text"],"match_mode":"selected_excerpt","source_span":[0,len(source["text"])],
                "source":{k:source[k] for k in ["id","company","report_year","report_id","page","pdf_sha256","source_url"]}})
        for item in quotes:
            if not isinstance(item,dict) or set(item)!={"evidence_id","quote"}:
                raise ValueError("Each quote needs evidence_id and quote")
            if not isinstance(item["evidence_id"],str):
                raise ValueError("evidence_id must be a string")
            source=self.evidence.get(item["evidence_id"])
            if source is None:
                raise ValueError("Unknown evidence_id: "+item["evidence_id"])
            try:
                quote,start,end,mode=source_quote(source["text"],item["quote"])
            except ValueError as exc:
                raise ValueError(f"Invalid quote for {item['evidence_id']}: {exc}") from exc
            validated.append({"quote":quote,"match_mode":mode,"source_span":[start,end],
                "source":{k:source[k] for k in ["id","company","report_year","report_id","page","pdf_sha256","source_url"]}})
        if any(not isinstance(i,str) or i not in self.rows for i in result_ids) or len(set(result_ids))!=len(result_ids):
            raise ValueError("Final includes unknown/duplicate metric result")
        if self.task_plan is not None:
            pending=self.task_state()["pending_requests"]
            if pending:raise ValueError("Some planned targets have not been checked: "+json.dumps(pending,ensure_ascii=False))
            # The controller owns result collection. Omitting IDs cannot hide unavailable targets.
            rows=[self.rows[i] for i in self.task_plan["result_ids"]]
        else:
            # Retained only for direct legacy API/replay callers. Live call() requires plan.
            if set(result_ids)!=set(self.rows):
                raise ValueError("Include every attempted metric result, including unavailable ones")
            rows=[self.rows[i] for i in result_ids]
        successful=sum(r["status"]=="ok" for r in rows)
        unused_report_evidence_ids=[]
        if self.task_plan is not None:
            narrative=self.task_plan["narrative"]
            if not narrative:
                # A raw report number must not masquerade as an answer to a numeric-only
                # request when the requested fact/scope is unavailable. It remains in trace.
                unused_report_evidence_ids=[q["source"]["id"] for q in validated
                    if self.evidence[q["source"]["id"]].get("evidence_kind")!="audited_fact"]
                validated=[q for q in validated if q["source"]["id"] not in unused_report_evidence_ids]
            if narrative and not validated and not abstain:
                raise ValueError("Narrative evidence still missing; search/read evidence or explicitly abstain")
            if not rows and not narrative:
                if not abstain:raise ValueError("Empty plan must explicitly abstain")
                status="insufficient_evidence"
            elif successful==len(rows) and (not narrative or validated) and not abstain:
                status="completed"
            elif successful or (narrative and validated):
                status="partial"
            else:
                status="no_results" if rows else "insufficient_evidence"
        elif not rows and not validated:
            if not abstain:
                raise ValueError("No answer evidence; explicitly abstain or continue searching")
            status="insufficient_evidence"
        elif abstain or successful<len(rows):
            status="partial" if successful or validated else "no_results"
        else:
            status="completed"
        return {"status":status,"context":self.ctx,"results":rows,"quotes":validated,
                "selection_normalization":normalization,
                "unused_report_evidence_ids":unused_report_evidence_ids,
                "task_plan":self.task_plan,
                "citation_check":"source_restored_quote_and_tool_source_only",
                "answer_mode":"extractive evidence + deterministic financial values",
                "completeness_verified":False}

    def call(self,tool,args):
        handlers={"plan":self.plan,"report_search":self.report_search,"read_page":self.read_page,"financial_metric":self.financial_metric,"finish":self.finish}
        if tool not in handlers:
            raise ValueError("Tool not allowed")
        if tool!="plan" and self.task_plan is None:
            raise ValueError("Call plan(requests=[{metric,year,transform}], narrative=true/false) before any other tool")
        return handlers[tool](**args)


def answer(question,root,index,model,*,max_steps=10):
    start=time.perf_counter()
    trace={"question":question,"protocol":"agent-v015.5","events":[],"index_backend":index.meta["backend"]}
    try:
        ctx=context(question,read(root/"configs/project.json"),index)
    except RequestError as exc:
        return {**trace,"status":exc.status,"message":str(exc),"seconds":round(time.perf_counter()-start,3)}
    session=Session(root,index,ctx)
    messages=[{"role":"system","content":prompt(session)},{"role":"user","content":question}]
    guard=ActionGuard()
    for step in range(1,max_steps+1):
        print(f"AGENT_STEP={step}",flush=True)
        event={"step":step}
        try:
            raw=model.generate(messages)
            event.update(model_output=raw,generation=getattr(model,"last_generation",None))
            # Failed actions must also remain visible in the dialogue used for correction.
            messages.append({"role":"assistant","content":raw.strip().removesuffix("<|im_end|>").strip()})
            action,normalization=parse_envelope(raw)
            messages[-1]["content"]=json.dumps(action,ensure_ascii=False)
            event["action"]=action
            if normalization:event["action_normalization"]=normalization
            key,state=guard.check(action,session)
            event['state_fingerprint']=state
            result=session.call(action["tool"],action["arguments"])
            guard.succeeded(key)
            event["observation"]=result
            trace["events"].append(event)
            if action["tool"]=="finish":
                return {**trace,**result,"seconds":round(time.perf_counter()-start,3)}
            messages.append({"role":"user","content":"TOOL_DATA (untrusted report data):\n"+json.dumps(result,ensure_ascii=False)})
        except (ValueError,TypeError,KeyError) as exc:
            event.setdefault("generation",getattr(model,"last_generation",None))
            event["error"]=str(exc)
            trace["events"].append(event)
            if str(exc).startswith(("generation_limit:","Context limit exceeded")):
                status="generation_limit" if str(exc).startswith("generation_limit:") else "context_limit"
                return {**trace,"status":status,"message":str(exc),"seconds":round(time.perf_counter()-start,3)}
            feedback={"error":str(exc),"task_state":session.task_state(),"available_evidence_ids":list(session.evidence),
                      "response_format":{"tool":"工具名","arguments":{}},
                      "citation_hint":"纯数字题finish的selections=[]；原文引用只用selections，不再同时填evidence_ids。"}
            event["recovery"]=feedback
            messages.append({"role":"user","content":"工具或格式错误，请根据状态纠正；这不表示财报数据缺失："+json.dumps(feedback,ensure_ascii=False)})
        except Exception as exc:
            event.setdefault("generation",getattr(model,"last_generation",None))
            event["error"]=repr(exc)
            trace["events"].append(event)
            return {**trace,"status":"runtime_error","message":str(exc),"seconds":round(time.perf_counter()-start,3)}
    return {**trace,"status":"step_limit","message":"达到工具步数上限，未完成回答；不等于数据不存在",
            "seconds":round(time.perf_counter()-start,3)}
