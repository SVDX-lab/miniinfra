"""第 16 期检查与实验：同步 vs 异步（首响应、进度重连、取消、用户隔离、空闲占用）。"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_16.sandbox_service import TaskService, make_handler, CHAPTER
from common import dockerops, images, results, stats
from http.server import ThreadingHTTPServer

CHECKS = []
USER_A = "fake-token-userA-111"
USER_B = "fake-token-userB-222"

TRAJ = [
    ["echo", "step-1"],
    ["sh", "-c", "echo working > /workspace/note.txt"],
    ["sleep", "1"],
    ["sh", "-c", "echo 'final artifact' > /workspace/result.txt"],
    ["sleep", "1"],
]


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def api(method, path, body=None, token=None, timeout=120):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method=method)
    if token:
        req.add_header("X-User-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


PORT = 0


def main() -> int:
    global PORT
    dockerops.cleanup_chapter(CHAPTER)
    svc = TaskService()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    PORT = srv.server_address[1]

    # ---- 异步：首响应 + 进度重连 + 产物 ----
    acks, completions = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        code, r = api("POST", "/tasks", {"user_token": USER_A, "trajectory": TRAJ})
        acks.append((time.perf_counter() - t0) * 1000)
        tid = r["task_id"]
        progresses = []
        t_submit = time.perf_counter()
        while True:
            code, v = api("GET", f"/tasks/{tid}", token=USER_A)  # 断线重连=再次轮询
            progresses.append(v["progress"]["done_steps"])
            if v["state"] in ("done", "failed"):
                break
            time.sleep(0.2)
        completions.append((time.perf_counter() - t_submit) * 1000)
        code, art = api("GET", f"/tasks/{tid}/artifact?path=result.txt", token=USER_A)
        ok_art = base64.b64decode(art["content_b64"]) == b"final artifact\n"
        check(f"异步任务完成且产物正确（{tid[:8]}）",
              v["state"] == "done" and ok_art and progresses[-1] == 5,
              f"progress_seq={progresses}")
    monotone = all(b >= a for a, b in zip(progresses, progresses[1:]))
    check("进度随轮询单调不减（重连语义）", monotone)

    ack_s = stats.summary(acks)
    # ---- 同步：同一轨迹阻塞 ----
    sync_first = []
    for _ in range(3):
        t0 = time.perf_counter()
        code, r = api("POST", "/tasks/sync", {"user_token": USER_A, "trajectory": TRAJ})
        sync_first.append((time.perf_counter() - t0) * 1000)
        assert r.get("completed") and r["state"] == "done"
    sync_s = stats.summary(sync_first)
    check("异步首响应 << 同步阻塞（P50 差 10 倍以上）",
          ack_s["p50"] < sync_s["p50"] / 10,
          f"async_ack p50={ack_s['p50']:.1f}ms sync_block p50={sync_s['p50']:.0f}ms")

    # ---- 用户隔离 ----
    code, r = api("POST", "/tasks", {"user_token": USER_A, "trajectory": TRAJ})
    tidA = r["task_id"]
    time.sleep(0.5)
    code, v = api("GET", f"/tasks/{tidA}", token=USER_B)
    check("用户 B 无法查询 A 的任务（404）", code == 404)
    code, v = api("GET", f"/tasks/{tidA}/artifact?path=result.txt", token=USER_B)
    check("用户 B 无法下载 A 的产物（404）", code == 404)
    code, v = api("POST", f"/tasks/{tidA}/cancel", token=USER_B)
    check("用户 B 无法取消 A 的任务（404）", code == 404)

    # ---- 取消 ----
    prog_before = None
    deadline = time.time() + 20
    while time.time() < deadline:
        code, v = api("GET", f"/tasks/{tidA}", token=USER_A)
        if v["progress"]["done_steps"] >= 2:
            prog_before = v["progress"]["done_steps"]
            break
        time.sleep(0.1)
    code, c = api("POST", f"/tasks/{tidA}/cancel", token=USER_A)
    time.sleep(1.0)
    code, v = api("GET", f"/tasks/{tidA}", token=USER_A)
    alive = dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}", include_stopped=False)
    check("取消运行中任务：状态 cancelled 且容器清理",
          c.get("state") == "cancelled" and v["state"] == "cancelled" and len(alive) == 0,
          f"progress_stopped_at={prog_before}")

    # 清理
    dockerops.cleanup_chapter(CHAPTER)
    check("实验后容器归零", len(dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}")) == 0)

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "async_ack_ms": ack_s, "async_completion_ms": stats.summary(completions),
           "sync_blocked_ms": sync_s}
    path = results.write_result(
        os.path.dirname(os.path.abspath(__file__)), "async_agent_ab", doc,
        extra_meta={"image": images.PINNED["alpine"], "steps": len(TRAJ),
                    "users": 2, "tokens": "course-local-fake"})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
