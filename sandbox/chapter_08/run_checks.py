"""第 08 期检查与实验：缓冲 vs 流式限额（内存上界、截断语义、客户端断开）。"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_08.sandbox_service import SandboxService, make_handler, CHAPTER
from common import dockerops, images, results, stats
from http.server import ThreadingHTTPServer

CHECKS = []
FLOOD_32M = ["sh", "-c", "yes hello | head -c 33554432"]
FLOOD_64M = ["sh", "-c", "yes hello | head -c 67108864"]
SLOW_FLOOD = ["sh", "-c",
              "for i in $(seq 1 640); do dd if=/dev/zero bs=1K count=100 2>/dev/null; "
              "sleep 0.05; done"]


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def api(method, url, body=None, timeout=180):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0
    return -1.0


class RssSampler:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.baseline = rss_mb()
        self.peak = self.baseline
        self._run = threading.Event()

    def start(self):
        def loop():
            while not self._run.is_set():
                self.peak = max(self.peak, rss_mb())
                time.sleep(self.interval)
        threading.Thread(target=loop, daemon=True).start()

    def stop(self) -> dict:
        self._run.set()
        time.sleep(self.interval)
        return {"baseline_mb": round(self.baseline, 1),
                "peak_mb": round(self.peak, 1),
                "peak_delta_mb": round(self.peak - self.baseline, 1)}


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)
    svc = SandboxService()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def new_session(mode):
        _, s = api("POST", f"{base}/sessions", {"output_mode": mode})
        return s["session_id"]

    # 1) 小输出回归：两种模式内容一致
    sid_a, sid_b = new_session("buffer"), new_session("stream")
    _, ea = api("POST", f"{base}/sessions/{sid_a}/exec", {"cmd": ["echo", "hello"]})
    _, eb = api("POST", f"{base}/sessions/{sid_b}/exec", {"cmd": ["echo", "hello"]})
    a_out = base64.b64decode(ea["stdout_b64"])
    b_out = base64.b64decode(eb["stdout"]["head_b64"])
    check("小输出：两种模式内容一致（无截断）",
          a_out == b_out == b"hello\n" and not eb["stdout"]["truncated"])

    # 2) 32MiB 洪泛：buffer 组
    sm = RssSampler()
    sm.start()
    _, ea = api("POST", f"{base}/sessions/{sid_a}/exec", {"cmd": FLOOD_32M, "timeout_ms": 60000})
    rss_a = sm.stop()
    check("buffer：全量 32MiB 进响应", ea["stdout_bytes"] == 33554432)
    check("buffer：服务 RSS 峰值增量 >60MiB（随输出规模增长）",
          rss_a["peak_delta_mb"] > 60, str(rss_a))

    # 3) 32MiB 洪泛：stream 组
    sm = RssSampler()
    sm.start()
    _, eb = api("POST", f"{base}/sessions/{sid_b}/exec", {"cmd": FLOOD_32M, "timeout_ms": 60000})
    rss_b = sm.stop()
    so = eb["stdout"]
    head, tail = base64.b64decode(so["head_b64"]), base64.b64decode(so["tail_b64"])
    check("stream：total=全量、kept≤1MiB、truncated=true",
          so["total_bytes"] == 33554432 and so["kept_bytes"] <= 1024 * 1024 + 1
          and so["truncated"])
    # 33554432 % 6 = 2：定长洪泛以 "he" 结尾，头尾须与重构的期望流精确一致
    expected_full = (b"hello\n" * (33554432 // 6 + 1))[:33554432]
    check("stream：头尾语义与期望流精确一致",
          head == expected_full[:len(head)] and tail == expected_full[-len(tail):])
    check("stream：退出码与 buffer 组一致", eb["exit_code"] == ea["exit_code"] == 0)
    check("stream：服务 RSS 峰值增量 <30MiB（内存上界）",
          rss_b["peak_delta_mb"] < 30, str(rss_b))
    check("stream 响应体远小于 buffer（传输也省）",
          len(json.dumps(eb)) < len(json.dumps(ea)) / 10,
          f"stream={len(json.dumps(eb))//1024}KiB buffer={len(json.dumps(ea))//1024}KiB")

    # 4) 64MiB 洪泛再压 stream 一轮
    _, eb2 = api("POST", f"{base}/sessions/{sid_b}/exec", {"cmd": FLOOD_64M, "timeout_ms": 60000})
    check("stream：64MiB 洪泛 total 正确且 kept 仍 ≤1MiB",
          eb2["stdout"]["total_bytes"] == 67108864 and eb2["stdout"]["kept_bytes"] <= 1024 * 1024 + 1)

    # 5) 客户端断开（读超时 3s，负载 ~32s，命令自身超时 8s tree 清理）
    abort_outcome = {}
    abort_sids = []

    def aborting_exec(mode, key):
        sid = new_session(mode)
        abort_sids.append(sid)
        try:
            api("POST", f"{base}/sessions/{sid}/exec",
                {"cmd": SLOW_FLOOD, "timeout_ms": 8000}, timeout=3)
            abort_outcome[key] = "unexpected-success"
        except Exception as e:
            abort_outcome[key] = f"client-aborted({type(e).__name__})"
        api("POST", f"{base}/sessions/{sid}/stop")

    for mode, key in (("buffer", "buf"), ("stream", "str")):
        t = threading.Thread(target=aborting_exec, args=(mode, key))
        t.start()
        t.join(timeout=30)
    time.sleep(6)  # 等命令超时清理完成
    code, h = api("GET", f"{base}/health")
    left = [sid for sid in abort_sids
            if dockerops.ps_ids(f"mis.session={sid}", include_stopped=False)]
    check("客户端断开后：服务健康、（仅断开会话）容器无残留",
          code == 200 and h.get("ok") and len(left) == 0,
          f"{abort_outcome.get('buf')}/{abort_outcome.get('str')} left={len(left)}")

    # 清理
    for sid in (sid_a, sid_b):
        api("POST", f"{base}/sessions/{sid}/stop")
    svc.shutdown()
    srv.shutdown()
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "rss": {"buffer_32mib": rss_a, "stream_32mib": rss_b},
           "client_abort": abort_outcome}
    path = results.write_result(
        os.path.dirname(os.path.abspath(__file__)), "backpressure_ab", doc,
        extra_meta={"image": images.PINNED["alpine"], "cache_state": "warm",
                    "flood_bytes": [33554432, 67108864],
                    "stream_keep": "head512K+tail512K"})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
