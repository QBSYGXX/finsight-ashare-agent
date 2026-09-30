"""模块：进程管理；作用：选择独立环境、连续日志、准确退出码；核心业务：否。"""
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"vllm_v013"))
from runtime import BASE,ENV,process_env,verify_package


def main():
    args=sys.argv[1:]
    if not args:
        raise SystemExit("Usage: bash agent_v015/run.sh doctor|index|chat|ask|evaluate|retrieve|export [options]")
    gpu=os.environ.get("FINSIGHT_GPU","1")
    accelerated=args[0] in {"chat","ask","doctor"} or (args[0]=="evaluate" and "--retrieval-only" not in args)
    env=process_env(gpu,accelerated=accelerated)
    env["FINSIGHT_ROOT"]=str(ROOT)
    env["PYTHONPATH"]=str(ROOT)
    verify_package()
    python=(ENV if accelerated else BASE)/"bin/python"
    logdir=ROOT/"runs"/(datetime.now().strftime("%Y%m%d-%H%M%S")+f"-agent-v015-{os.getpid()}")
    logdir.mkdir(parents=True,exist_ok=False)
    print(f"LOG={logdir/'run.log'}\nGPU={gpu} RUNTIME={python}",flush=True)
    with (logdir/"run.log").open("wb") as log:
        # Chat inherits the terminal; no background heartbeat corrupts input editing.
        if args[0]=="chat":
            print("交互日志保存每题JSON/HTML；终端输入期间不打印心跳。",flush=True)
            child=subprocess.Popen([str(python),"-u","-m","agent_v015.cli",*args],cwd=ROOT,env=env,start_new_session=os.name=="posix")
        else:
            child=subprocess.Popen([str(python),"-u","-m","agent_v015.cli",*args],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=os.name=="posix")
        print(f"PID={child.pid}",flush=True)
        thread=None
        try:
            if child.stdout is not None:
                import threading
                def pump():
                    for line in iter(child.stdout.readline,b""):
                        log.write(line); log.flush()
                        sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
                thread=threading.Thread(target=pump,daemon=True)
                thread.start()
                start=time.monotonic()
                while child.poll() is None:
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print(f"PROCESS_RUNNING pid={child.pid} seconds={time.monotonic()-start:.0f}; 查看上方最后阶段",flush=True)
            else: child.wait()
        except KeyboardInterrupt:
            if os.name=="posix": os.killpg(child.pid,signal.SIGTERM)
            else: child.terminate()
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: child.kill();child.wait()
        finally:
            # Only this child's dedicated process group; never kill other GPU jobs.
            if os.name=="posix":
                try: os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError: pass
            if thread is not None:
                thread.join(timeout=10)
                if thread.is_alive() and os.name=="posix":
                    try: os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    thread.join(timeout=5)
        code=child.returncode
        log.write(f"\nEXIT={code}\n".encode())
    print(f"EXIT={code}",flush=True)
    raise SystemExit(code)


if __name__=="__main__": main()
