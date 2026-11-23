"""第 13 期检查与实验：内存 vs journal（幂等、kill -9 重启、响应丢失、对账）。"""
from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images, results, stats

CHAPTER = "c13"
CHAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKS = []
APPEND_CMD = ["sh", "-c", "echo append >> /workspace/log.txt"]


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def api(base, method, path, body=None, timeout=120):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(f"{base}{path}", data=data, method=method)

    def _parse(raw: bytes, code: int):
        try:
            return code, json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return code, {"http_error": code, "body_raw": raw[:100].decode(errors="replace")}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _parse(r.read(), r.status)
    except urllib.error.HTTPError as e:
        return _parse(e.read(), e.code)


def _safe_api(base, method, path, body=None, timeout=120):
    try:
        return api(base, method, path, body, timeout)
    except Exception as e:  # 连接被杀/重置等：响应丢失
        return -1, {"transport_error": type(e).__name__}


def count_log_lines(workspaces, sid) -> int:
    p = os.path.join(workspaces, sid, "log.txt")
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def start_service(state_store, port):
    proc = subprocess.Popen(
        [sys.executable, "-m", "chapter_13.sandbox_service", str(port), state_store],
        cwd=os.path.dirname(CHAPTER_DIR), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            code, r = api(f"http://127.0.0.1:{port}", "GET", "/health", timeout=2)
            if code == 200:
                return proc, f"http://127.0.0.1:{port}"
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError("service did not start")


def main() -> int:
    # 防御：上次异常退出可能留下残留服务进程（占用固定端口会让新服务起不来，
    # 测试打到旧进程的状态上——实测踩过的坑）
    subprocess.run(["pkill", "-f", "chapter_13.sandbox_service"], capture_output=True)
    time.sleep(0.5)
    import random
    ports = random.Random(int(time.time())).sample(range(19000, 19500), 4)
    dockerops.cleanup_chapter(CHAPTER)
    ws_root = os.path.join(CHAPTER_DIR, "data_local", "workspaces")
    os.makedirs(ws_root, exist_ok=True)

    # ============ journal 组 ============
    jpath = os.path.join(CHAPTER_DIR, "data_local", "journal.jsonl")
    if os.path.exists(jpath):
        os.remove(jpath)
    proc, base = start_service("journal", ports[0])

    # 1) 执行一次 + 同 request_id 重发（幂等）
    _, s = api(base, "POST", "/sessions", {"ttl_ms": 600000})
    J1 = s["session_id"]
    _, e1 = api(base, "POST", f"/sessions/{J1}/exec",
                {"cmd": APPEND_CMD, "request_id": "req-1"})
    _, e2 = api(base, "POST", f"/sessions/{J1}/exec",
                {"cmd": APPEND_CMD, "request_id": "req-1"})
    n1 = count_log_lines(ws_root, J1)
    check("journal：同 request_id 重发不重复执行（幂等）",
          e2.get("deduplicated") is True and e1["command_id"] == e2["command_id"]
          and n1 == 1, f"lines={n1} e1cat={e1.get('category')} e2dedup={e2.get('deduplicated')}")

    # 2) 响应丢失场景：执行成功但客户端丢弃，再重发
    _, e_lost = api(base, "POST", f"/sessions/{J1}/exec",
                    {"cmd": APPEND_CMD, "request_id": "req-lost"})
    del e_lost  # 模拟响应在网络中丢失
    _, e_again = api(base, "POST", f"/sessions/{J1}/exec",
                     {"cmd": APPEND_CMD, "request_id": "req-lost"})
    check("journal：响应丢失后重发返回记录不重复执行",
          e_again.get("deduplicated") is True and count_log_lines(ws_root, J1) == 2)

    # 3) kill -9 于命令执行中 → 重启恢复
    _, s = api(base, "POST", "/sessions", {"ttl_ms": 600000})
    J2 = s["session_id"]
    fut = ThreadPoolExecutor(max_workers=1).submit(
        _safe_api, base, "POST", f"/sessions/{J2}/exec",
        {"cmd": ["sh", "-c", "sleep 3 && echo append >> /workspace/log.txt"],
         "request_id": "req-crash", "timeout_ms": 10000})
    time.sleep(1.2)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)
    fut.result(timeout=30)  # 客户端拿到连接错误/空响应（响应丢失）
    proc2, base2 = start_service("journal", ports[1])
    code, st = api(base2, "GET", f"/sessions/{J2}/status")
    check("journal：重启后会话状态可查（不 404）", code == 200 and st["state"] in
          ("active", "released"), f"state={st.get('state')}")

    # 对账：kill 时命令可能已写 journal_accepted？interrupted 语义由重放+标记给出。
    # 该命令没有 completed 记录 → 重放后 req-crash 不可见；显式标记后拒绝原 id 重放：
    api(base2, "POST", "/admin/mark_interrupted", {"request_id": "req-crash", "sid": J2})
    code, e = api(base2, "POST", f"/sessions/{J2}/exec",
                  {"cmd": APPEND_CMD, "request_id": "req-crash"})
    check("journal：interrupted 命令用原 id 重发被明确拒绝（不盲目重试）",
          code == 409 and e.get("error") == "request_interrupted")
    code, e = api(base2, "POST", f"/sessions/{J2}/exec",
                  {"cmd": APPEND_CMD, "request_id": "req-crash-2"})
    n2 = count_log_lines(ws_root, J2)
    check("journal：换新 request_id 正常执行（副作用恰好再 +1）",
          code == 200 and n2 == 1, f"lines={n2}（旧命令是否完成取决于崩溃点，不虚报）")

    # 4) 孤儿容器对账
    orphan = subprocess.run(
        ["docker", "run", "-d", "--name", "mis-c13-orphan-1",
         f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
         "--label=mis.session=s-orphannorec", images.PINNED["alpine"], "sleep", "infinity"],
        capture_output=True, text=True).stdout.strip()
    _, rec = api(base2, "POST", "/admin/reconcile")
    still = dockerops.docker(["ps", "-aq", "--filter", "id=" + orphan])
    check("journal：孤儿容器（有 label 无记录）被对账清理",
          rec.get("orphans_removed") and len(still.stdout.split()) == 0,
          str(rec))

    # 5) journal 开销
    lat_j = []
    for i in range(10):
        t0 = time.perf_counter()
        api(base2, "POST", f"/sessions/{J1}/exec",
            {"cmd": ["echo", "x"], "request_id": f"perf-{i}"})
        lat_j.append((time.perf_counter() - t0) * 1000)
    proc2.terminate()
    proc2.wait(timeout=10)
    for sid in (J1, J2):
        try:
            api(base2, "POST", f"/sessions/{sid}/stop", timeout=5)
        except Exception:
            pass
    dockerops.cleanup_chapter(CHAPTER)

    # ============ memory 组（基线，同场景关键子集） ============
    proc3, base3 = start_service("memory", ports[2])
    _, s = api(base3, "POST", "/sessions", {"ttl_ms": 600000})
    M1 = s["session_id"]
    _, m1 = api(base3, "POST", f"/sessions/{M1}/exec",
                {"cmd": APPEND_CMD, "request_id": "req-1"})
    _, m2 = api(base3, "POST", f"/sessions/{M1}/exec",
                {"cmd": APPEND_CMD, "request_id": "req-1"})
    check("memory：同 request_id 重发被重复执行（基线缺陷演示）",
          m2.get("deduplicated") is None and count_log_lines(ws_root, M1) == 2,
          f"lines={count_log_lines(ws_root, M1)}")
    proc3.send_signal(signal.SIGKILL)
    proc3.wait(timeout=10)
    proc4, base4 = start_service("memory", ports[3])
    code, _ = api(base4, "GET", f"/sessions/{M1}/status")
    check("memory：重启后状态丢失（会话 404）", code == 404)
    proc4.terminate()
    proc4.wait(timeout=10)
    dockerops.cleanup_chapter(CHAPTER)

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "journal_exec_overhead_ms": stats.summary(lat_j),
           "note": "interrupted 判定：崩溃时无 completed 记录的 request_id 显式标记后拒绝原 id 重放"}
    path = results.write_result(
        CHAPTER_DIR, "journal_ab", doc,
        extra_meta={"image": images.PINNED["alpine"], "state_stores": ["journal", "memory"],
                    "crash": "SIGKILL during exec"})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
