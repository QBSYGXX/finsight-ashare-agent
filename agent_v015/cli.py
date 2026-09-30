"""模块：操作入口；作用：建索引、常驻问答、评测与可下载报告；核心业务：否。"""
import argparse
from datetime import datetime
import html
import json
import os
from pathlib import Path
import sys
import uuid
import zipfile

from .storage import ROOT, read, save, sha, verify


def render(path,result):
    payload=json.dumps(result,ensure_ascii=False,indent=2)
    path.write_text('<!doctype html><meta charset="utf-8"><title>FinSight Agent v0.15</title>'
        '<style>body{max-width:1100px;margin:32px auto;font:16px/1.6 sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        '<h1>FinSight 财报 Agent 运行记录</h1><p>原文引用经过逐字核验；相关性与回答完整性仍需审阅。</p><pre>'+html.escape(payload)+'</pre>',encoding="utf-8")


class LazyModel:
    def __init__(self,args):
        self.args,self.model,self.last_generation=args,None,None
    def generate(self,messages):
        if self.model is None:
            from .model import Model
            self.model=Model(self.args.backend,self.args.thinking=="on",self.args.max_tokens)
        try:
            return self.model.generate(messages)
        finally:
            self.last_generation=self.model.last_generation


# 模块：结果导出；作用：连同两组诊断、逐题轨迹和参考指纹一次打包；核心业务：否。
def export_run(directory):
    target=ROOT/"exports"/("finsight-agent-v015-"+directory.name+".zip")
    target.parent.mkdir(exist_ok=True)
    if target.exists():raise ValueError("Export already exists; reuse the existing ZIP")
    files=sorted(p for p in directory.rglob("*") if p.is_file())
    manifest={p.relative_to(directory).as_posix():sha(p) for p in files}
    with zipfile.ZipFile(target,"x",zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(directory).as_posix())
        z.writestr("EXPORT_MANIFEST.json",json.dumps(manifest,indent=2))
    checksum=sha(target)
    target.with_suffix(".zip.sha256").write_text(checksum+"  "+target.name+"\n",encoding="ascii",newline="\n")
    print(f"EXPORT={target}\nSHA256={checksum}",flush=True)
    return target


def main():
    parser=argparse.ArgumentParser(description="Evidence-first RAG + financial tool Agent")
    parser.add_argument("command",choices=["doctor","index","retrieve","chat","ask","evaluate","export"])
    parser.add_argument("--split",choices=["validation","development"],default="validation")
    parser.add_argument("--retrieval",choices=["bm25","hybrid"],default="hybrid")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--backend",choices=["vllm","vllm-mtp"],default="vllm-mtp")
    parser.add_argument("--thinking",choices=["on","off"],default="off")
    parser.add_argument("--max-tokens",type=int)
    parser.add_argument("--question")
    parser.add_argument("--retrieval-only",action="store_true")
    parser.add_argument("--suite",choices=["diagnostic","natural","all"],default="diagnostic",
                        help="evaluate: original 13, natural 28, or both with one model load")
    parser.add_argument("--export-after",action="store_true",help="Export this evaluation run, including partial failures")
    parser.add_argument("--run",help="Run directory name printed by ask/evaluate (for export)")
    args=parser.parse_args()
    if args.max_tokens is not None and not 64<=args.max_tokens<=8192:
        parser.error("max-tokens must be 64..8192")
    if args.export_after and args.command!="evaluate":parser.error("--export-after requires evaluate")
    pin=verify()
    if args.command=="export":
        if not args.run or Path(args.run).name!=args.run:
            parser.error("--run must be a run directory name")
        directory=ROOT/"outputs/agent_v015/runs"/args.run
        if not directory.is_dir():
            parser.error("Run directory not found")
        export_run(directory)
        return
    if args.command=="doctor":
        import importlib.metadata
        receipt={"root":str(ROOT),"package_verified":True,"python":sys.version,"runtime":sys.executable,
                 "packages":{},"models":{},"index_exists":(ROOT/"outputs/agent_v015/index"/f"{args.split}-{args.retrieval}"/"metadata.json").exists()}
        for name in ["torch","transformers","numpy","vllm"]:
            try: receipt["packages"][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: receipt["packages"][name]=None
        for name in ["Qwen3.5-9B","bge-m3"]: receipt["models"][name]=(ROOT/"models"/name/"config.json").exists()
        import torch
        receipt["cuda_available"]=torch.cuda.is_available()
        receipt["visible_gpu_count"]=torch.cuda.device_count()
        save(ROOT/"outputs/agent_v015/doctor.json",receipt)
        print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
        if not all(receipt["models"].values()): raise RuntimeError("Missing local model")
        return
    from .retrieval import Index,build
    if args.command=="index":
        build(split=args.split,backend=args.retrieval,device=args.device)
        return
    index=Index(ROOT,ROOT/"outputs/agent_v015/index"/f"{args.split}-{args.retrieval}")
    tag=datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8]
    dest=ROOT/"outputs/agent_v015/runs"/tag
    save(dest/"run_config.json",{"args":vars(args),"package":pin,"index":index.meta,"model":"base Qwen3.5-9B; no LoRA"})
    print(f"RUN={tag}\nOUTPUT={dest}",flush=True)
    model=LazyModel(args)
    if args.command=="evaluate":
        from .evaluation import run
        from .natural_evaluation import run as run_natural
        if args.suite=="all":
            baseline=run(ROOT,index,model,dest/"diagnostic",args.retrieval_only)
            natural=(run_natural(ROOT,index,model,dest/"natural",args.retrieval_only)
                     if baseline["execution_status"]=="completed" else None)
            suites={"diagnostic":baseline,"natural":natural}
            summary={"protocol":"agent-v0155-combined","suites":suites,
                     "execution_status":"completed" if natural and natural["execution_status"]=="completed" else "failed",
                     "diagnostic_pass":{name:s["diagnostic_pass"] if s else None for name,s in suites.items()}}
            save(dest/"summary.json",summary)
        else:
            runner=run_natural if args.suite=="natural" else run
            summary=runner(ROOT,index,model,dest,args.retrieval_only)
        render(dest/"report.html",summary)
        save(dest/"completed.json",{"status":summary["execution_status"],"passed":summary["diagnostic_pass"]})
        if args.export_after:export_run(dest)
        if summary["execution_status"]!="completed":
            raise RuntimeError("Evaluation stopped after a runtime error; partial results and not_run list saved")
        return
    from .agent import answer,context,RequestError
    n=0
    while True:
        if args.command=="chat":
            try: question=input("财报问题（exit退出）> ").strip()
            except (EOFError,KeyboardInterrupt): break
            if question.lower() in {"exit","quit"}: break
            if not question: continue
        else:
            question=args.question
            if not question: parser.error("--question is required")
        print("收到问题："+question,flush=True)
        n+=1
        if args.command=="retrieve":
            try:
                ctx=context(question,read(ROOT/"configs/project.json"),index)
                result={"question":question,"context":ctx,"hits":index.search(question,ctx["company"],ctx["years"],6)}
            except RequestError as exc: result={"status":exc.status,"message":str(exc)}
        else: result=answer(question,ROOT,index,model)
        save(dest/f"question-{n}.json",result)
        render(dest/f"question-{n}.html",result)
        print(json.dumps({k:v for k,v in result.items() if k!="events"},ensure_ascii=False,indent=2),flush=True)
        print(f"TRACE={dest/f'question-{n}.json'}",flush=True)
        if args.command!="chat": break
    save(dest/"completed.json",{"status":"completed","questions":n})


if __name__=="__main__":
    main()
