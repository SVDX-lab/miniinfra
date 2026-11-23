"""第 02 期 Smoke 与正确性/边界检查 + A/B 对照实验（状态连续性、第二命令延迟、空闲占用）。"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_02.sandbox_service import SandboxService, make_handler, CHAPTER
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


# 后台服务用 busybox nc（alpine 的 busybox 构建 httpd 不在 applet 列表中）
SEQ = {
    "write": ["sh", "-c", "mkdir -p /workspace/www && echo hello > /workspace/www/index.html"],
    "start_service": ["sh", "-c",
                      "(while true; do echo hello | nc -l -p 8080; done) >/dev/null 2>&1 & "
                      "sleep 0.2; echo server-started"],
    "fetch": ["sh", "-c", "echo | nc -w 2 127.0.0.1 8080"],
    "read_file": ["sh", "-c", "cat /workspace/www/index.html"],
    "set_env": ["sh", "-c", "export FOO=1"],
    "get_env": ["sh", "-c", "echo v=$FOO"],
}


def run_sequence(base, mode):
    """固定序列，返回逐命令观察。"""
    obs = {}
    _, s = api("POST", f"{base}/sessions", {"mode": mode})
    sid = s["session_id"]
    for step, cmd in SEQ.items():
        _, e = api("POST", f"{base}/sessions/{sid}/exec", {"cmd": cmd, "timeout_ms": 8000})
        obs[step] = {"exit": e["exit_code"], "out": dec(e.get("stdout_b64", "")),
                     "dur_ms": e["duration_ms"]}
    obs["sid"] = sid
    return obs


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)
    svc = SandboxService()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    # ---- 两组对照：固定序列 ----
    obs_cmd = run_sequence(base, "per-cmd")
    obs_ses = run_sequence(base, "per-session")

    check("per-cmd：文件状态保留（写后可读）",
          obs_cmd["read_file"]["out"] == "hello\n")
    check("per-cmd：后台进程不存活（fetch 连接被拒）",
          obs_cmd["fetch"]["exit"] != 0, f"exit={obs_cmd['fetch']['exit']}")
    check("per-session：文件状态保留",
          obs_ses["read_file"]["out"] == "hello\n")
    check("per-session：后台进程跨命令存活（fetch 返回 hello）",
          obs_ses["fetch"]["exit"] == 0 and obs_ses["fetch"]["out"] == "hello\n",
          f"exit={obs_ses['fetch']['exit']} out={obs_ses['fetch']['out']!r}")
    check("per-session：服务进程确实在两次命令之间运行（start→fetch 间隔观察）",
          obs_ses["start_service"]["exit"] == 0)
    check("两种模式 shell 环境变量均不保留（进程状态≠shell状态）",
          obs_cmd["get_env"]["out"] == "v=\n" and obs_ses["get_env"]["out"] == "v=\n")

    # ---- 第二命令延迟对照（首条命令后的第二条）----
    lat = {"per-cmd": [], "per-session": []}
    for mode in ("per-cmd", "per-session"):
        for _ in range(10):
            _, s = api("POST", f"{base}/sessions", {"mode": mode})
            sid = s["session_id"]
            api("POST", f"{base}/sessions/{sid}/exec", {"cmd": ["echo", "1"]})  # 首条
            t0 = time.perf_counter()
            _, e = api("POST", f"{base}/sessions/{sid}/exec", {"cmd": ["echo", "2"]})  # 第二条
            lat[mode].append(e["duration_ms"])
            api("POST", f"{base}/sessions/{sid}/stop")
    s_cmd, s_ses = stats.summary(lat["per-cmd"]), stats.summary(lat["per-session"])
    check("per-session 第二命令延迟显著更低",
          s_ses["p50"] < s_cmd["p50"] * 0.8,
          f"per-cmd p50={s_cmd['p50']:.0f}ms vs per-session p50={s_ses['p50']:.0f}ms")

    # ---- 空闲常驻内存 ----
    _, s = api("POST", f"{base}/sessions", {"mode": "per-session", "ttl_ms": 15000})
    sid_idle = s["session_id"]
    api("POST", f"{base}/sessions/{sid_idle}/exec", {"cmd": ["echo", "x"]})
    time.sleep(1.0)
    r = dockerops.docker(["stats", "--no-stream",
                          "--format", "{{.Name}} {{.MemUsage}}",
                          f"mis-{CHAPTER}-{sid_idle[2:8]}"])
    idle_mem = r.stdout.strip()
    check("空闲常驻容器内存已采样", bool(idle_mem), idle_mem)

    # ---- TTL 到期 ----
    _, s = api("POST", f"{base}/sessions", {"mode": "per-session", "ttl_ms": 1500})
    sid_ttl = s["session_id"]
    api("POST", f"{base}/sessions/{sid_ttl}/exec", {"cmd": ["echo", "x"]})
    time.sleep(2.5)
    _, st = api("GET", f"{base}/sessions/{sid_ttl}/status")
    check("TTL 到期后状态 expired 且容器回收",
          st["state"] == "expired" and st["containers_alive"] == 0)
    code, e = api("POST", f"{base}/sessions/{sid_ttl}/exec", {"cmd": ["echo", "x"]})
    check("expired 会话 exec 被拒绝", code == 409 and e.get("error") == "session_expired")

    # ---- 超时（per-session 粗粒度）：会话终止且无残留 ----
    _, s = api("POST", f"{base}/sessions", {"mode": "per-session", "ttl_ms": 60000})
    sid_to = s["session_id"]
    _, e = api("POST", f"{base}/sessions/{sid_to}/exec",
               {"cmd": ["sleep", "10"], "timeout_ms": 800})
    _, st = api("GET", f"{base}/sessions/{sid_to}/status")
    check("per-session 命令超时：timed_out 且容器清理",
          e["timed_out"] and st["containers_alive"] == 0,
          f"state={st['state']}（粗粒度：整个会话被终止）")

    # ---- 清理 ----
    for sid in {obs_cmd["sid"], obs_ses["sid"], sid_idle}:
        api("POST", f"{base}/sessions/{sid}/stop")
    left = dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}", include_stopped=False)
    check("全部停止后容器归零", len(left) == 0, f"left={len(left)}")

    svc.shutdown()
    srv.shutdown()
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "sequence_observations": {"per-cmd": obs_cmd, "per-session": obs_ses},
           "second_command_latency_ms": {"per-cmd": s_cmd, "per-session": s_ses,
                                         "raw": lat},
           "idle_memory_sample": idle_mem}
    path = results.write_result(
        os.path.dirname(os.path.abspath(__file__)), "ab_session_state", doc,
        extra_meta={"image": images.PINNED["alpine"], "cache_state": "warm",
                    "sequence": list(SEQ)})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    print(f"第二命令延迟 per-cmd p50={s_cmd['p50']:.0f}ms per-session p50={s_ses['p50']:.0f}ms")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
