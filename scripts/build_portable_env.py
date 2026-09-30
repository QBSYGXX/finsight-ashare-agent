"""Build and verify Linux Conda on this Windows machine using Docker/WSL2."""
import argparse
from datetime import datetime
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def run(command, log):
    print("RUN " + " ".join(map(str, command)), flush=True)
    log.write("RUN " + " ".join(map(str, command)) + "\n")
    process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    messages = queue.Queue()
    def pump():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)
    threading.Thread(target=pump, daemon=True).start()
    last = time.monotonic()
    last_line = "started"
    while True:
        try:
            line = messages.get(timeout=1)
            if line is None:
                break
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            last, last_line = time.monotonic(), line.strip()[-250:]
        except queue.Empty:
            if time.monotonic() - last >= 45:
                message = "HEARTBEAT: no new output for 45s. Last: " + last_line
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                last = time.monotonic()
    code = process.wait()
    print(f"EXIT={code}", flush=True)
    log.write(f"EXIT={code}\n")
    log.flush()
    if code:
        raise RuntimeError("Command failed: " + str(command[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    release = args.output_dir.resolve()
    release.mkdir(parents=True, exist_ok=True)
    log_path = release / ("finsight-env-build-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".log")
    name = "finsight-env-build-" + uuid.uuid4().hex[:8]
    print(f"LOG={log_path}\nCONTAINER={name}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        run(["docker", "build", "--progress=plain", "--platform", "linux/amd64", "-t", "finsight-conda-builder:0.1", "server/environment"], log)
        run(["docker", "run", "--name", name, "--init", "--platform", "linux/amd64",
             "--mount", "type=volume,source=finsight-conda-build-cache,target=/build",
             "--mount", f"type=bind,source={ROOT},target=/project,readonly",
             "--mount", f"type=bind,source={release},target=/exports", "finsight-conda-builder:0.1",
             "bash", "/project/server/environment/build_linux.sh"], log)
        run([sys.executable, ROOT / "scripts/register_env_artifact.py", "--output-dir", release], log)
        run([sys.executable, ROOT / "scripts/validate_portable_env.py", "--output-dir", release], log)
    print(f"BUILD_COMPLETE LOG={log_path}", flush=True)


if __name__ == "__main__":
    main()
