"""第 03 期检查与对照实验：两种超时清理路径的孤儿进程、端口占用、取消延迟。"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_03.sandbox_service import SandboxService, make_handler, CHAPTER, count_procs
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


# 派生树负载：主进程前台 sleep；后台 nc 服务 + 3 个 sleep
SPAWN_CMD = ["sh", "-c",
             "(while true; do echo hello | nc -l -p 8080; done) >/dev/null 2>&1 & "
             "sleep 300 & sleep 300 & sleep 300 & "
             "sleep 300"]
PROBE_PORT = ["sh", "-c", "echo | nc -w 1 127.0.0.1 8080"]


def run_cancel_case(base, cancel_mode):
    _, s = api("POST", f"{base}/sessions", {"cancel_mode": cancel_mode, "ttl_ms": 60000})
    sid = s["session_id"]
    _, e = api("POST", f"{base}/sessions/{sid}/exec",
               {"cmd": SPAWN_CMD, "timeout_ms": 1000})
    _, st = api("GET", f"{base}/sessions/{sid}/status")
    _, probe = api("POST", f"{base}/sessions/{sid}/exec",
                   {"cmd": PROBE_PORT, "timeout_ms": 5000})
    _, nxt = api("POST", f"{base}/sessions/{sid}/exec", {"cmd": ["echo", "alive"]})
    return {"sid": sid, "exec": e, "status": st,
            "port_probe_exit": probe["exit_code"],
            "next_cmd_out": base64.b64decode(nxt["stdout_b64"]).decode()}


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)
    svc = SandboxService()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    # ---- 两种清理路径对照 ----
    mo = run_cancel_case(base, "main-only")
    rep_mo = mo["exec"]["cancel_report"]
    check("main-only：返回 timed_out 且带清理报告",
          mo["exec"]["timed_out"] and rep_mo["path"] == "main-only")
    check("main-only：遗留存活孤儿进程 >0",
          rep_mo["leftover_alive"] > 0,
          f"alive={rep_mo['leftover_alive']} zombie={rep_mo['leftover_zombie']}")
    check("main-only：8080 端口仍被占用（后台服务未死）",
          mo["port_probe_exit"] == 0, f"probe_exit={mo['port_probe_exit']}")
    check("main-only：会话仍可用",
          mo["next_cmd_out"] == "alive\n")

    tr = run_cancel_case(base, "tree")
    rep_tr = tr["exec"]["cancel_report"]
    check("tree：返回 timed_out 且带清理报告",
          tr["exec"]["timed_out"] and rep_tr["path"] == "tree")
    check("tree：存活与僵尸进程双归零（--init/tini 作 PID1 收割）",
          rep_tr["leftover_alive"] == 0 and rep_tr["leftover_zombie"] == 0,
          f"alive={rep_tr['leftover_alive']} zombie={rep_tr['leftover_zombie']}")
    check("tree：8080 端口已释放",
          tr["port_probe_exit"] != 0, f"probe_exit={tr['port_probe_exit']}")
    check("tree：会话仍可用", tr["next_cmd_out"] == "alive\n")

    # ---- 取消到归零延迟（tree，重复 10 次）----
    cancel_lat = []
    for i in range(10):
        _, s = api("POST", f"{base}/sessions",
                   {"cancel_mode": "tree", "ttl_ms": 60000})
        sid = s["session_id"]
        _, e = api("POST", f"{base}/sessions/{sid}/exec",
                   {"cmd": SPAWN_CMD, "timeout_ms": 800})
        cancel_lat.append(e["cancel_report"]["cancel_to_zero_ms"])
        api("POST", f"{base}/sessions/{sid}/stop")
    lat = stats.summary(cancel_lat)
    check("tree 取消到归零延迟已测（重复 10 次）", lat["n"] == 10 and lat["max"] < 3000,
          f"p50={lat['p50']:.0f}ms p95={lat['p95']:.0f}ms max={lat['max']:.0f}ms")

    # ---- 清理 ----
    for c in (mo, tr):
        api("POST", f"{base}/sessions/{c['sid']}/stop")
    left = dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}", include_stopped=False)
    check("全部停止后容器归零", len(left) == 0)

    svc.shutdown()
    srv.shutdown()
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "main_only": {"leftover_alive": rep_mo["leftover_alive"],
                         "leftover_zombie": rep_mo["leftover_zombie"],
                         "port_still_bound": mo["port_probe_exit"] == 0},
           "tree": {"leftover_alive": rep_tr["leftover_alive"],
                    "leftover_zombie": rep_tr["leftover_zombie"],
                    "port_released": tr["port_probe_exit"] != 0},
           "tree_cancel_to_zero_ms": lat, "tree_cancel_raw": cancel_lat,
           "note": "僵尸由容器 PID1(sleep infinity) 不收割导致，容器销毁时随命名空间消失；生产用 --init/tini"}
    path = results.write_result(
        os.path.dirname(os.path.abspath(__file__)), "cancel_paths_ab", doc,
        extra_meta={"image": images.PINNED["alpine"], "cache_state": "warm",
                    "repeats": 10})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
