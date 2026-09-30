"""Persistent logs, background processes, GPU mutual exclusion and status files."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight_ashare.common import now, read_json, write_json

GPU_COMMANDS = {"setup", "smoke", "train", "evaluate", "chat"}
COMMANDS = GPU_COMMANDS | {"doctor", "prepare", "export", "status"}


# 模块：任务运行器；作用：日志可见、退出码持久化、同项目GPU任务互斥；核心业务：否。
def gpu_sample(path):
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=8)
        if result.returncode == 0:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(result.stdout)
    except (OSError, subprocess.TimeoutExpired):
        pass


def worker(run_dir, args):
    lock = None
    log_path = run_dir / "run.log"
    status = {"command": args, "pid": os.getpid(), "started": now(), "status": "running", "exit_code": None}
    write_json(run_dir / "status.json", status)
    code = 1
    child = None
    with log_path.open("a", encoding="utf-8", buffering=1) as logfile:
        def emit(text):
            sys.stdout.write(text)
            sys.stdout.flush()
            logfile.write(text)
        try:
            if args[0] in GPU_COMMANDS:
                if os.name != "posix":
                    raise RuntimeError("GPU commands require the Linux server")
                import fcntl
                (ROOT / ".runtime").mkdir(exist_ok=True)
                lock = (ROOT / ".runtime/gpu.lock").open("a+")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONPATH=str(ROOT / "src"))
            command = [sys.executable, "-u", "-m", "finsight_ashare", *args]
            emit(f"START {now()} | {' '.join(args)}\n")
            # stdin remains attached for interactive chat; no SSH or port forwarding is needed.
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1, start_new_session=os.name == "posix")
            status["child_pid"] = child.pid
            write_json(run_dir / "status.json", status)
            output = queue.Queue()
            def pump():
                for line in child.stdout:
                    output.put(line)
                output.put(None)
            threading.Thread(target=pump, daemon=True).start()
            last_output = last_sample = time.monotonic()
            last_line = "process started"
            ended = False
            while not ended or child.poll() is None:
                try:
                    line = output.get(timeout=1)
                    if line is None:
                        ended = True
                    else:
                        emit(line)
                        last_output, last_line = time.monotonic(), line.strip()[-300:]
                except queue.Empty:
                    pass
                if time.monotonic() - last_sample >= 30:
                    gpu_sample(run_dir / "gpu.csv")
                    status.update(last_output=last_line, updated=now())
                    write_json(run_dir / "status.json", status)
                    last_sample = time.monotonic()
                if time.monotonic() - last_output >= 60:
                    emit(f"HEARTBEAT {now()}: no new output for 60s; possible download/compute stall. Last: {last_line}\n")
                    last_output = time.monotonic()
            code = child.wait()
        except (Exception, KeyboardInterrupt) as error:
            emit(f"ERROR {type(error).__name__}: {error}\n")
            if child and child.poll() is None:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    child.wait()
            code = 130 if isinstance(error, KeyboardInterrupt) else 1
        finally:
            status.update(status="completed" if code == 0 else "failed", exit_code=code, finished=now())
            write_json(run_dir / "status.json", status)
            emit(f"END {now()} EXIT={code} STATUS={status['status']}\nLOG={log_path}\n")
            if lock:
                lock.close()
    return code


def main():
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Received termination signal")
    signal.signal(signal.SIGTERM, interrupt)
    args = sys.argv[1:]
    if args and args[0] == "_worker":
        return worker(Path(args[1]), args[2:])
    if not args or args[0] not in COMMANDS:
        print("Usage: bash server/run.sh doctor|setup|smoke|prepare|train|evaluate|chat|status|export [options]")
        return 2
    if args[0] == "status":
        for path in sorted((ROOT / "runs").glob("*/status.json"))[-10:]:
            print(str(path) + "\n" + json.dumps(read_json(path), ensure_ascii=False, indent=2))
        return 0
    background = "--background" in args
    if background:
        args.remove("--background")
        if args[0] not in {"train", "evaluate", "setup", "prepare"}:
            raise ValueError("Background supported for setup/prepare/train/evaluate only")
    run_dir = ROOT / "runs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + args[0] + "-" + uuid.uuid4().hex[:6])
    run_dir.mkdir(parents=True)
    write_json(run_dir / "status.json", {"command": args, "status": "queued", "created": now()})
    if background:
        if os.name != "posix":
            raise ValueError("Background server tasks require Linux")
        with (run_dir / "launcher.log").open("ab") as stream:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "_worker", str(run_dir), *args],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Launched PID={process.pid}; this does NOT mean the task completed.\nLOG={run_dir / 'run.log'}\nSTATUS={run_dir / 'status.json'}\ntail -f '{run_dir / 'run.log'}'")
        return 0
    return worker(run_dir, args)


if __name__ == "__main__":
    raise SystemExit(main())
