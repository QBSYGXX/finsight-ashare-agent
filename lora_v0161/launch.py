"""模块：离线作业；作用：后台运行、持续日志、阶段门禁、失败导出及子进程清理；核心业务：否。"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vllm_v013')]
from runtime import BASE,ENV,process_env
from lora_v0161.common import ADDON,read,save,sha,verify,run_path


def export(dest,logdir):
    (ROOT/'exports').mkdir(exist_ok=True)
    target=ROOT/'exports'/f'finsight-lora-v0161-{dest.name}-{datetime.now():%Y%m%d-%H%M%S}.zip'
    manifest={}
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        def put(name,data):
            z.writestr(name,data);manifest[name]=hashlib.sha256(data).hexdigest()
        for path in sorted(dest.rglob('*.json')):
            if 'merged' in path.relative_to(dest).parts:continue
            put(path.relative_to(dest).as_posix(),path.read_bytes())
        for name in ['PACKAGE_MANIFEST.json','config.json','data/audit.json']:
            put('release/'+name,(ADDON/name).read_bytes())
        if (logdir/'run.log').exists():put('run.log',(logdir/'run.log').read_bytes())
        put('export_scope.json',json.dumps({'weights_included':False,'training_text_included':False,
             'note':'Adapters and merged weights remain on server. Progress exports do not imply success.'}).encode())
        z.writestr('EXPORT_MANIFEST.json',json.dumps(manifest,indent=2))
    checksum=sha(target)
    target.with_suffix('.zip.sha256').write_text(checksum+'  '+target.name+'\n',encoding='ascii',newline='\n')
    print(f'EXPORT={target}\nSHA256={checksum}',flush=True)
    return target


# 模块：进程监控；作用：stdout/stderr连续写盘，心跳标注距最新输出时长；核心业务：否。
def stage(stage_name,run,gpus,log):
    gpu_list=gpus.split(',')
    training=stage_name in ['probe','train']
    accelerated=stage_name in ['base','tuned']
    env=process_env(gpus if training else gpu_list[0],accelerated=accelerated)
    env.update(PYTHONPATH=str(ROOT),TOKENIZERS_PARALLELISM='false')
    python=(ENV if accelerated else BASE)/'bin/python'
    command=[str(python),'-u']
    if training and len(gpu_list)>1:
        command+=['-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc_per_node={len(gpu_list)}']
    command+=['-m','lora_v0161.worker',stage_name,'--run',run]
    print(f'STAGE_START={stage_name} GPUs={env["CUDA_VISIBLE_DEVICES"]} RUNTIME={python}',flush=True)
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    print('CHILD_PID='+str(child.pid),flush=True)
    last=[time.monotonic()]
    def pump():
        for block in iter(child.stdout.readline,b''):
            last[0]=time.monotonic();log.write(block);log.flush()
            sys.stdout.buffer.write(block);sys.stdout.buffer.flush()
    thread=threading.Thread(target=pump,daemon=True);thread.start()
    try:
        while child.poll() is None:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                quiet=time.monotonic()-last[0]
                print(f'PROCESS_RUNNING stage={stage_name} pid={child.pid} quiet_seconds={quiet:.0f}'+
                      ('; possible stall, inspect last output/GPU' if quiet>=180 else ''),flush=True)
    except (KeyboardInterrupt,SystemExit):
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        thread.join(timeout=10)
        if thread.is_alive():
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            thread.join(timeout=5)
    print(f'STAGE_END={stage_name} EXIT={child.returncode}',flush=True)
    if child.returncode:raise subprocess.CalledProcessError(child.returncode,command)


def main():
    parser=argparse.ArgumentParser(description='New-protocol LoRA; single GPU by default, optional 2/4-GPU DDP')
    parser.add_argument('command',choices=['all','prepare','probe','train','merge','evaluate','export'])
    parser.add_argument('--run')
    parser.add_argument('--background',action='store_true')
    args=parser.parse_args()
    verify()
    if args.command not in ['all','prepare'] and not args.run:parser.error('--run is required')
    run=args.run or datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    dest=run_path(run)
    if args.command in ['all','prepare'] and dest.exists():
        raise FileExistsError('Existing run is immutable; use a new run ID')
    if args.command!='export' and (dest/(args.command+'-status.json')).exists():
        raise FileExistsError('Stage status exists; preserve previous results')
    logdir=ROOT/'runs'/('lora-v0161-'+run);logdir.mkdir(parents=True,exist_ok=True)
    gpus=os.environ.get('FINSIGHT_GPUS','1')
    if any(not s.isdigit() for s in gpus.split(',')) or len(set(gpus.split(',')))!=len(gpus.split(',')) or len(gpus.split(',')) not in [1,2,4]:
        raise ValueError('FINSIGHT_GPUS must list 1, 2 or 4 distinct GPU IDs')
    print(f'RUN={run}\nLOG={logdir/"run.log"}\nGPUS={gpus}',flush=True)
    if args.background:
        with (logdir/'launcher.log').open('ab') as stream:
            child=subprocess.Popen([sys.executable,'-u',__file__,args.command,'--run',run],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(logdir/'pid.json',{'pid':child.pid,'run':run,'command':args.command})
        print(f'PID={child.pid}\nPROGRESS_LOG={logdir/"launcher.log"}\n'
              f'tail -n 60 -f {logdir/"launcher.log"}',flush=True)
        return
    if args.command=='export':export(dest,logdir);return
    sequences={'all':['prepare','probe','reload-probe','train','reload-full','merge','base','tuned','compare'],
               'prepare':['prepare'],'probe':['probe','reload-probe'],'train':['train','reload-full'],
               'merge':['merge'],'evaluate':['base','tuned','compare']}
    # SIGTERM to launcher also propagates through the active stage cleanup.
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    code=0
    try:
        with (logdir/'run.log').open('ab') as log:
            for step in sequences[args.command]:stage(step,run,gpus,log)
        save(dest/(args.command+'-status.json'),{'status':'completed','command':args.command})
    except BaseException as exc:
        code=getattr(exc,'returncode',1) or 1
        if not isinstance(code,int) or code<1:code=1
        save(dest/(args.command+'-status.json'),{'status':'failed','command':args.command,'error':repr(exc),'exit_code':code})
        print('ERROR='+repr(exc),flush=True)
    finally:
        if args.command=='all' or code:
            try:export(dest,logdir)
            except Exception as exc:print('EXPORT_FAILED='+repr(exc),flush=True);code=code or 1
        print('EXIT='+str(code),flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
