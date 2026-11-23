"""第 01 期 Benchmark：本机基线 vs 直连容器 vs 经 API 容器的单命令延迟。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_01.sandbox_service import SandboxService, make_handler, CONSTRAINT_ARGS, CHAPTER
from common import dockerops, images, results, stats
from http.server import ThreadingHTTPServer

WARMUP = 3
N = 20
CMD = ["echo", "ok"]


def timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def main():
    dockerops.cleanup_chapter(CHAPTER)
    images.ensure(images.PINNED["alpine"])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(SandboxService()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def api(method, url, body=None):
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(url, data=data, method=method)
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    s = api("POST", f"{base}/sessions", {})
    sid = s["session_id"]
    ws = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "data_local", "workspaces", sid)
    img = images.PINNED["alpine"]

    # 预热（声明为暖缓存）
    for _ in range(WARMUP):
        api("POST", f"{base}/sessions/{sid}/exec", {"cmd": CMD})

    # 1) 本机基线：固定可信命令，subprocess 直跑
    host = [timed(lambda: subprocess.run(CMD, capture_output=True, check=True))
            for _ in range(N)]

    # 2) 直连容器：不经 API，docker run 全参数
    def direct():
        r = subprocess.run(
            ["docker", "run", "--rm", f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
             *CONSTRAINT_ARGS, "-v", f"{ws}:/workspace", "-w", "/workspace", img, *CMD],
            capture_output=True, check=True, timeout=60)
    container_direct = [timed(direct) for _ in range(N)]

    # 3) 经 API 容器：完整服务路径
    via_api = [timed(lambda: api("POST", f"{base}/sessions/{sid}/exec", {"cmd": CMD}))
               for _ in range(N)]

    # 4) create→首条命令可用
    create_first = []
    for _ in range(10):
        t0 = time.perf_counter()
        ns = api("POST", f"{base}/sessions", {})
        e = api("POST", f"{base}/sessions/{ns['session_id']}/exec", {"cmd": ["echo", "ok"]})
        assert e["exit_code"] == 0
        create_first.append((time.perf_counter() - t0) * 1000.0)
        api("POST", f"{base}/sessions/{ns['session_id']}/stop")

    api("POST", f"{base}/sessions/{sid}/stop")
    srv.shutdown()

    data = {
        "config": {"cmd": CMD, "warmup": WARMUP, "n": N, "cache_state": "warm"},
        "host_baseline_ms": stats.summary(host),
        "container_direct_ms": stats.summary(container_direct),
        "container_via_api_ms": stats.summary(via_api),
        "create_to_first_command_ms": stats.summary(create_first),
        "raw": {"host": [round(x, 3) for x in host],
                "direct": [round(x, 3) for x in container_direct],
                "api": [round(x, 3) for x in via_api],
                "create_first": [round(x, 3) for x in create_first]},
        "overhead": {
            "container_minus_host_p50": round(stats.summary(container_direct)["p50"] - stats.summary(host)["p50"], 3),
            "api_minus_direct_p50": round(stats.summary(via_api)["p50"] - stats.summary(container_direct)["p50"], 3),
        },
    }
    path = results.write_result(
        os.path.dirname(os.path.abspath(__file__)), "benchmark_single_command", data,
        extra_meta={"image": img, "n": N, "warmup": WARMUP, "cache_state": "warm"})
    print(json.dumps({k: v for k, v in data.items() if k != "raw"}, indent=2))
    print(f"→ {path}")


if __name__ == "__main__":
    main()
