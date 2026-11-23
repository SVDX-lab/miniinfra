"""第 12 期：准入与放置。异速 worker + 本地队列 + 两种放置策略（本期核心机制）。

模型：
- worker = 一个执行节点：固定槽位数（并发上限）、速度系数（slow=0.5，用 sleep
  时长缩放模拟）、本地 FIFO 队列、自己的执行容器（exec 跑任务命令）；
- 放置策略在提交时决定任务去哪个 worker 的队列：
  random：均匀随机；
  cost-aware：选"预计排空时间"最小的 worker——(在跑剩余时长 + 排队时长之和)/槽位数，
  时长用任务声明的 duration_s（训练/评测批量任务可声明的成本口径）按 worker 速度折算；
- 总容量恒定：单 worker 基线 3 槽 × 1.0 速 = 双 worker（2 槽 ×1.0 + 2 槽 ×0.5）= 3。
"""
from __future__ import annotations

import collections
import json
import os
import random
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images

CHAPTER = "c12"

CONSTRAINTS = ["--network", "none", "--read-only", "--user", "65534:65534",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--memory", "256m", "--memory-swap", "256m",
               "--tmpfs", "/tmp:rw,size=16m,noexec"]


def sh(argv, timeout=120):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


class Worker:
    """执行节点：本地队列 + 槽位线程 + 速度系数 + 自己的执行容器。"""

    def __init__(self, name: str, slots: int, speed: float):
        self.name, self.slots, self.speed = name, slots, speed
        self.cv = threading.Condition()
        self.pending: collections.deque = collections.deque()
        self.running: list[dict] = []
        self.max_concurrent = 0
        self.container = f"mis-{CHAPTER}-node-{name}"
        r = sh(["docker", "run", "-d", "--name", self.container,
                f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
                f"--label=mis.worker={name}",
                *CONSTRAINTS, images.PINNED["alpine"], "sleep", "infinity"])
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:200])
        self._threads = [threading.Thread(target=self._slot_loop, daemon=True)
                         for _ in range(slots)]
        for t in self._threads:
            t.start()

    def _slot_loop(self):
        while True:
            with self.cv:
                while not self.pending:
                    self.cv.wait()
                item = self.pending.popleft()
            rec = item["rec"]
            start = time.perf_counter()
            rec["queue_wait_ms"] = round((start - rec["submit_ts"]) * 1000, 1)
            entry = {"t0": start, "dur": item["dur_scaled"]}
            with self.cv:
                self.running.append(entry)
                self.max_concurrent = max(self.max_concurrent, len(self.running))
            for cmd in item["commands"]:
                cmd = list(cmd)
                if cmd[0] == "sleep":  # 速度模拟：慢 worker 的 sleep 按比例拉长
                    cmd[1] = str(round(float(cmd[1]) / self.speed, 3))
                sh(["docker", "exec", self.container, *cmd])
            with self.cv:
                self.running.remove(entry)
            rec["done_ev"].set()

    def est_time_to_drain(self) -> float:
        """预计排空时间（秒）：在跑剩余 + 排队之和，按槽位数学并行折算。"""
        now = time.perf_counter()
        with self.cv:
            rem_running = sum(max(0.0, (e["t0"] + e["dur"]) - now)
                              for e in self.running)
            rem_pending = sum(i["dur_scaled"] for i in self.pending)
        return (rem_running + rem_pending) / self.slots

    def submit(self, rec: dict, commands: list, dur_base: float):
        item = {"rec": rec, "commands": commands, "dur_scaled": dur_base / self.speed}
        rec["submit_ts"] = time.perf_counter()
        with self.cv:
            self.pending.append(item)
            self.cv.notify()

    def snapshot(self) -> dict:
        with self.cv:
            return {"name": self.name, "slots": self.slots, "speed": self.speed,
                    "max_concurrent": self.max_concurrent}

    def destroy(self):
        sh(["docker", "rm", "-f", self.container])


class Router:
    def __init__(self, workers: list[Worker], policy: str, seed: int = 42):
        self.workers = workers
        self.policy = policy
        self.rng = random.Random(seed)
        self.records: list[dict] = []
        self.lock = threading.Lock()

    def _pick(self) -> Worker:
        if self.policy == "random":
            return self.rng.choice(self.workers)
        return min(self.workers, key=lambda w: w.est_time_to_drain())

    def run_task(self, payload: dict) -> dict:
        dur_base = float(payload.get("duration_s", 0.4))
        commands = payload["commands"]
        rec: dict = {"done_ev": threading.Event()}
        w = self._pick()
        rec["worker"] = w.name
        w.submit(rec, commands, dur_base)
        if not rec["done_ev"].wait(timeout=180):
            raise RuntimeError(f"task timeout on {w.name}")
        rec["task_ms"] = round((time.perf_counter() - rec["submit_ts"]) * 1000, 1)
        out = {"worker": w.name, "queue_wait_ms": rec["queue_wait_ms"],
               "task_ms": rec["task_ms"]}
        with self.lock:
            self.records.append(out)
        return out

    def stats(self) -> dict:
        return {"workers": [w.snapshot() for w in self.workers]}


def make_handler(router: Router):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                if self.path.strip("/") == "tasks":
                    n = int(self.headers.get("Content-Length") or 0)
                    payload = json.loads(self.rfile.read(n) or b"{}")
                    return self._send(200, router.run_task(payload))
                self._send(404, {"error": "no_route"})
            except Exception as e:
                self._send(500, {"error": "internal", "message": str(e)})

        def do_GET(self):
            if self.path.strip("/") == "stats":
                return self._send(200, router.stats())
            self._send(404, {"error": "no_route"})

        def log_message(self, *a):
            pass

    return Handler


def start_router(workers: list[Worker], policy: str):
    router = Router(workers, policy)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(router))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, router
