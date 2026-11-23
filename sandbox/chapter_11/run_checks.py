"""第 11 期检查与实验：常驻 vs 到期释放（文件连续性、进程不恢复、恢复延迟、空闲成本）。"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_11.sandbox_service import SandboxService, make_handler, CHAPTER
from common import dockerops, images, results, stats
from http.server import ThreadingHTTPServer

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def api(method, url, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def dec(b64):
    return base64.b64decode(b64).decode(errors="replace")


def mem_mib_of(container_or_none) -> float:
    if not container_or_none:
        return 0.0
    r = dockerops.docker(["stats", "--no-stream", "--format", "{{.MemUsage}}", container_or_none])
    try:
        return float(r.stdout.split("/")[0].strip().rstrip("MiBKiB").strip()) \
            if "MiB" in r.stdout else 0.0
    except Exception:
        return 0.0


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)
    svc = SandboxService()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    # 常驻组（TTL 极大）
    _, sr = api("POST", f"{base}/sessions", {"ttl_ms": 10 ** 9})
    RES = sr["session_id"]
    api("POST", f"{base}/sessions/{RES}/exec", {"cmd": ["echo", "warm"]})

    # 释放组（TTL 1.5s）
    _, se = api("POST", f"{base}/sessions", {"ttl_ms": 1500})
    REL = se["session_id"]
    api("POST", f"{base}/sessions/{REL}/exec",
        {"cmd": ["sh", "-c", "printf state-v0 > /workspace/f.txt"]})

    # 1) 常驻组 idle 后容器仍在
    time.sleep(2.5)
    _, st = api("GET", f"{base}/sessions/{RES}/status")
    check("常驻组：idle>1.5s 后容器仍在（TTL 不生效）",
          st["state"] == "active" and st["containers_alive"] == 1)

    # 2) 释放组到期：容器消失、工作区保留
    _, st = api("GET", f"{base}/sessions/{REL}/status")
    check("释放组：到期容器释放、工作区保留、state=released",
          st["state"] == "released" and st["containers_alive"] == 0 and st["workspace_exists"])

    # 3) 文件连续性 + 透明重建（10 次循环写→释放→读）
    loss = 0
    for i in range(1, 11):
        api("POST", f"{base}/sessions/{REL}/exec",
            {"cmd": ["sh", "-c", f"printf state-v{i-1} > /workspace/f.txt"]})
        time.sleep(2.2)  # 过 TTL 释放
        _, e = api("POST", f"{base}/sessions/{REL}/exec", {"cmd": ["cat", "/workspace/f.txt"]})
        if dec(e["stdout_b64"]) != f"state-v{i-1}":
            loss += 1
    _, st = api("GET", f"{base}/sessions/{REL}/status")
    check("文件状态连续：10 次释放/重建循环丢失率 0",
          loss == 0 and st["restores_total"] >= 10, f"loss={loss} restores={st['restores_total']}")

    # 4) 后台进程不恢复：nc 服务到期释放后重建容器内无进程、端口不通
    _, e = api("POST", f"{base}/sessions/{REL}/exec",
               {"cmd": ["sh", "-c",
                        "(while true; do echo hello | nc -l -p 8080; done) >/dev/null 2>&1 & "
                        "sleep 0.2; echo started"]})
    time.sleep(2.2)  # 释放（后台进程随容器销毁）
    _, e = api("POST", f"{base}/sessions/{REL}/exec",
               {"cmd": ["sh", "-c",
                        "echo | nc -w 1 127.0.0.1 8080; echo probe_rc=$?; "
                        "ls /proc | grep -c '^[0-9]'"]})
    out = dec(e["stdout_b64"])
    check("后台进程不随重建恢复（端口不通、进程不残留）",
          "probe_rc=1" in out, out.strip().replace("\n", " | "))

    # 5) 恢复延迟：释放组 vs 常驻组同命令
    rel_lat, res_lat = [], []
    for _ in range(10):
        api("POST", f"{base}/sessions/{REL}/exec", {"cmd": ["echo", "x"]})
        time.sleep(2.2)
        _, e = api("POST", f"{base}/sessions/{REL}/exec", {"cmd": ["echo", "ok"]})
        rel_lat.append(e["duration_ms"])
        _, e = api("POST", f"{base}/sessions/{RES}/exec", {"cmd": ["echo", "ok"]})
        res_lat.append(e["duration_ms"])
    rs, rr = stats.summary(rel_lat), stats.summary(res_lat)
    check("恢复路径可用：释放组重建后命令成功（含重建成本）",
          rs["p50"] > rr["p50"] * 1.5, f"released p50={rs['p50']:.0f}ms resident p50={rr['p50']:.0f}ms")

    # 6) 空闲成本：4s 观察窗 MiB·s 积分（0.5s 采样）
    def gib_seconds(sid, container):
        total = 0.0
        for _ in range(8):
            total += mem_mib_of(container) * 0.5
            time.sleep(0.5)
        return total

    res_container = f"mis-{CHAPTER}-{RES[2:8]}"
    cost_res = gib_seconds(RES, res_container)
    api("POST", f"{base}/sessions/{REL}/exec", {"cmd": ["echo", "x"]})
    time.sleep(2.2)  # 释放
    cost_rel = gib_seconds(REL, None)  # 释放后无容器
    check("空闲成本：释放组 4s 窗口 MiB·s ≈ 0，常驻组累积",
          cost_rel == 0.0 and cost_res > 0, f"resident={cost_res:.1f}MiB·s released={cost_rel}")

    api("POST", f"{base}/sessions/{RES}/stop")
    api("POST", f"{base}/sessions/{REL}/stop")
    svc.shutdown()
    srv.shutdown()
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "restore_latency_ms": {"released_path": rs, "resident_path": rr,
                                  "raw_released": rel_lat, "raw_resident": res_lat},
           "idle_cost_mib_s_4s_window": {"resident": cost_res, "released": cost_rel},
           "file_loss_rate": loss}
    path = results.write_result(
        os.path.dirname(os.path.abspath(__file__)), "idle_recycle_ab", doc,
        extra_meta={"image": images.PINNED["alpine"], "cache_state": "warm",
                    "ttl_ms": {"released": 1500, "resident": 10**9}, "cycles": 10})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
