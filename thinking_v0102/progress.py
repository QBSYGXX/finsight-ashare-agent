"""Serialize heartbeat output with interactive input boundaries."""
import json
import threading
import time


# 模块：交互进度；作用：输入期间静默、工作期间保留心跳；核心业务：否。
class Progress:
    def __init__(self, interval=30, output=print, clock=time.perf_counter):
        self.interval, self.output, self.clock = interval, output, clock
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.waiting = False
        self.phase = "initializing"
        self.since = clock()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def set_phase(self, phase):
        with self.lock:
            self.phase, self.since = phase, self.clock()

    def tick(self):
        with self.lock:
            if self.waiting or self.finished.is_set():
                return False
            self.output(f"PROCESS_RUNNING phase={self.phase} phase_seconds={self.clock()-self.since:.0f}", flush=True)
            return True

    def _run(self):
        while not self.finished.wait(self.interval):
            self.tick()

    def read_question(self, reader=input):
        # Printing and pausing share the heartbeat's lock: no late heartbeat can
        # appear after the prompt. Other generation threads are joined by ask().
        with self.lock:
            self.waiting = True
            self.output("财务指标> ", end="", flush=True)
        try:
            question = reader().strip()
        finally:
            with self.lock:
                self.waiting = False
                self.phase, self.since = "processing_input", self.clock()
        if question:
            self.output("QUESTION=" + json.dumps(question, ensure_ascii=False), flush=True)
        return question

    def stop(self):
        self.finished.set()
        if self.thread.is_alive():
            self.thread.join()
