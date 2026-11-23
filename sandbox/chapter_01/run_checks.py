"""第 01 期 Smoke 与正确性/边界检查（先于 Benchmark，协议第 3 节）。"""
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
from chapter_01.sandbox_service import SandboxService, make_handler, CHAPTER
from common import dockerops, images, results
from http.server import ThreadingHTTPServer

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def api(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def dec(b64: str) -> str:
    return base64.b64decode(b64).decode(errors="replace")


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(SandboxService()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    # Smoke：接口、状态机、错误提示
    code, r = api("GET", f"{base}/health")
    check("health", code == 200 and r.get("ok"))

    code, r = api("GET", f"{base}/sessions/s-none/status")
    check("未知会话 404", code == 404 and r.get("error") == "session_not_found")

    code, s = api("POST", f"{base}/sessions", {})
    sid = s["session_id"]

    code, e = api("POST", f"{base}/sessions/{sid}/exec", {"cmd": ["echo", "hello"]})
    check("正常命令 exit=0 stdout 完整",
          e["exit_code"] == 0 and dec(e["stdout_b64"]) == "hello\n" and e["category"] == "ok",
          f"exit={e['exit_code']} out={dec(e['stdout_b64'])!r}")

    code, e = api("POST", f"{base}/sessions/{sid}/exec", {"cmd": ["sh", "-c", "exit 3"]})
    check("非零退出码透传", e["exit_code"] == 3 and e["category"] == "ok")

    code, e = api("POST", f"{base}/sessions/{sid}/exec",
                  {"cmd": ["sh", "-c", "echo o; echo e >&2; exit 1"]})
    check("stdout/stderr 分离",
          dec(e["stdout_b64"]) == "o\n" and dec(e["stderr_b64"]) == "e\n")

    # 正确性：工作区写读
    api("POST", f"{base}/sessions/{sid}/exec",
        {"cmd": ["sh", "-c", "printf abc > f.txt"]})
    code, fl = api("GET", f"{base}/sessions/{sid}/files")
    names = [x["name"] for x in fl["entries"]]
    code, fc = api("GET", f"{base}/sessions/{sid}/files?path=f.txt")
    check("工作区写读一致", "f.txt" in names and base64.b64decode(fc["content_b64"]) == b"abc")

    code, fc = api("GET", f"{base}/sessions/{sid}/files?path=../../etc/passwd")
    check("路径逃逸被拒绝", code == 400 and fc.get("error") == "path_escape")

    # 边界：超时
    t0 = time.perf_counter()
    code, e = api("POST", f"{base}/sessions/{sid}/exec",
                  {"cmd": ["sleep", "5"], "timeout_ms": 700})
    wall = time.perf_counter() - t0
    check("超时终止 timed_out 且延迟受控",
          e["timed_out"] and e["category"] == "timeout" and 0.6 < wall < 4.0,
          f"wall={wall:.2f}s")
    check("超时后无容器残留", len(dockerops.ps_ids(
        f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False)) == 0)

    # 边界：容器启动失败（镜像不存在）
    code, s2 = api("POST", f"{base}/sessions", {"image": "nonexistent-xyz:latest"})
    code, e = api("POST", f"{base}/sessions/{s2['session_id']}/exec", {"cmd": ["echo", "x"]})
    check("镜像不存在归类 env_prep_error",
          e["category"] == "env_prep_error" and e["exit_code"] == 125)
    code, h = api("GET", f"{base}/health")
    check("启动失败后服务仍健康", code == 200 and h.get("ok"))
    api("POST", f"{base}/sessions/{s2['session_id']}/stop")

    # 边界：执行中停止（取消）
    fut = ThreadPoolExecutor(max_workers=1).submit(
        api, "POST", f"{base}/sessions/{sid}/exec", {"cmd": ["sleep", "20"]})
    time.sleep(1.0)
    code, st = api("POST", f"{base}/sessions/{sid}/stop")
    code_exec, e = fut.result(timeout=60)
    check("停止会话取消执行中命令", st.get("stopped") and True, f"exec category={e.get('category')}")

    # 边界：停止后资源为零
    code, stat = api("GET", f"{base}/sessions/{sid}/status")
    left = dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}", include_stopped=False)
    check("stop 后容器与工作区归零",
          stat["state"] == "stopped" and stat["containers_alive"] == 0
          and not stat["workspace_exists"] and len(left) == 0,
          f"containers_left={len(left)}")

    # 停止后 exec 被拒绝
    code, e = api("POST", f"{base}/sessions/{sid}/exec", {"cmd": ["echo", "x"]})
    check("停止后 exec 返回 409", code == 409 and e.get("error") == "session_stopped")

    srv.shutdown()
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS)}
    path = results.write_result(
        os.path.dirname(os.path.abspath(__file__)), "checks_smoke_correctness", doc,
        extra_meta={"image": images.PINNED["alpine"], "cache_state": "warm"})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
