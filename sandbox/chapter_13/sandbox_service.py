"""第 13 期：持久状态机与幂等性。唯一变量：内存状态 vs journal 持久化 + request_id 去重。

journal（JSONL，每行一次状态转换，写后 flush）：
  session_created / command_accepted / command_completed(result) / command_interrupted* / session_stopped
启动时重放恢复会话表与去重表。exec 接受 request_id：
  - 已有 completed 结果 → 原样返回，不执行（幂等）；
  - 上次 interrupted → 拒绝并说明（不盲目重放有副作用的命令）；
  - 新 id → 执行并在完成后记录。
restart 后 POST /admin/reconcile 清理"有 label 无记录"的孤儿容器。
state_store=memory 时不落盘、不去重（基线组）。
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images

CHAPTER = "c13"

CONSTRAINT_ARGS = [
    "--network", "none", "--read-only", "--user", "65534:65534",
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
    "--memory", "256m", "--memory-swap", "256m", "--cpus", "1", "--pids-limit", "64",
    "--tmpfs", "/tmp:rw,size=16m,noexec",
]
TREE_KILL = ('p=$$; for d in /proc/[0-9]*; do x=${d#/proc/}; '
             '[ "$x" = 1 ] || [ "$x" = "$p" ] || kill -9 "$x" 2>/dev/null; done; exit 0')


class ApiError(Exception):
    def __init__(self, http_code, code, message):
        super().__init__(message)
        self.http_code, self.code, self.message = http_code, code, message


def _dex(container, argv, timeout=30):
    return subprocess.run(["docker", "exec", container, *argv],
                          capture_output=True, text=True, timeout=timeout)


class SandboxService:
    def __init__(self, base_dir=None, state_store="journal", ttl_ms=60000):
        chapter_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir or os.path.join(chapter_dir, "data_local", "workspaces")
        os.makedirs(self.base_dir, exist_ok=True)
        self.journal_path = os.path.join(os.path.dirname(self.base_dir), "journal.jsonl")
        self.state_store = state_store
        self.ttl_ms = ttl_ms
        self.sessions: dict[str, dict] = {}
        self.dedup: dict[str, dict] = {}       # request_id -> result（completed）
        self.interrupted: dict[str, dict] = {}  # request_id -> 记录（interrupted）
        self.lock = threading.Lock()
        self._stop_evt = threading.Event()
        if state_store == "journal":
            self._replay_journal()
        threading.Thread(target=self._ttl_loop, daemon=True).start()

    # ---- journal ----
    def _append(self, kind: str, **fields):
        if self.state_store != "journal":
            return
        rec = {"ts": time.time(), "kind": kind, **fields}
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _replay_journal(self):
        if not os.path.exists(self.journal_path):
            return
        with open(self.journal_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 崩溃时的半行：忽略（append+fsync 前崩溃的转换视为未发生）
                k = rec["kind"]
                if k == "session_created":
                    self.sessions[rec["sid"]] = {"workspace": rec["workspace"],
                                                 "created_at": rec["ts"], "state": "active",
                                                 "container": None, "ttl_ms": rec["ttl_ms"],
                                                 "last_used": rec["ts"], "in_flight": 0}
                elif k == "session_stopped":
                    if rec["sid"] in self.sessions:
                        self.sessions[rec["sid"]]["state"] = "stopped"
                elif k == "command_accepted":
                    s = self.sessions.get(rec["sid"])
                    if s:
                        s["container"] = rec.get("container")
                elif k == "command_completed":
                    self.dedup[rec["request_id"]] = rec["result"]
                elif k == "command_interrupted":
                    self.interrupted[rec["request_id"]] = rec

    # ---- TTL（含 in_flight 防护，见第 11 期）----
    def _ttl_loop(self):
        while not self._stop_evt.wait(0.25):
            now = time.time()
            with self.lock:
                due = [sid for sid, s in self.sessions.items()
                       if s["state"] == "active" and s["in_flight"] == 0
                       and now - s["last_used"] > s["ttl_ms"] / 1000.0]
                for sid in due:
                    self.sessions[sid]["state"] = "released"
            for sid in due:
                self._kill_container(sid)

    def _kill_container(self, sid):
        ids = dockerops.ps_ids(f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False)
        dockerops.kill_ids(ids)
        if ids:
            dockerops.docker(["rm", "-f", *ids])

    def _start_container(self, sid, ws):
        name = f"mis-{CHAPTER}-{sid[2:8]}"
        r = subprocess.run(
            ["docker", "run", "-d", "--name", name,
             f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
             f"--label={dockerops.LABEL_SESSION}={sid}",
             *CONSTRAINT_ARGS, "-v", f"{ws}:/workspace", "-w", "/workspace",
             images.PINNED["alpine"], "sleep", "infinity"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise ApiError(500, "env_prep_error", r.stderr.strip()[:200])
        return name

    # ---- API ----
    def create(self, payload: dict) -> dict:
        sid = "s-" + uuid.uuid4().hex[:12]
        ws = os.path.join(self.base_dir, sid)
        os.makedirs(ws, exist_ok=True)
        os.chmod(ws, 0o777)
        ttl = int(payload.get("ttl_ms", self.ttl_ms))
        with self.lock:
            self.sessions[sid] = {"workspace": ws, "created_at": time.time(),
                                  "state": "active", "container": None, "ttl_ms": ttl,
                                  "last_used": time.time(), "in_flight": 0}
        self._append("session_created", sid=sid, workspace=ws, ttl_ms=ttl)
        self.sessions[sid]["container"] = self._start_container(sid, ws)
        return {"session_id": sid, "ttl_ms": ttl}

    def _get(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        if s["state"] == "stopped":
            raise ApiError(409, "session_stopped", "session is stopped")
        return s

    def exec(self, sid, payload: dict) -> dict:
        s = self._get(sid)
        cmd = payload.get("cmd")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            raise ApiError(400, "bad_cmd", "cmd must be a non-empty list of strings")
        request_id = payload.get("request_id")
        if request_id and self.state_store == "journal":
            # 幂等/打断语义只在持久模式下存在；memory 基线=纯进程内状态（重启即失、无去重）
            if request_id in self.dedup:
                r = dict(self.dedup[request_id])
                r["deduplicated"] = True
                return r  # 已完成：返回记录，不重复执行
            if request_id in self.interrupted:
                raise ApiError(409, "request_interrupted",
                               "上次执行被打断且副作用未知；请用新 request_id 显式决定是否重试")
        timeout_ms = int(payload.get("timeout_ms", 10000))
        cid = "c-" + uuid.uuid4().hex[:12]
        with self.lock:
            s["last_used"] = time.time()
            s["in_flight"] += 1
        t0 = time.perf_counter()
        try:
            if s["state"] == "released" or not s["container"] \
                    or not dockerops.ps_ids(f"mis.session={sid}", include_stopped=False):
                self._kill_container(sid)
                s["container"] = self._start_container(sid, s["workspace"])
                with self.lock:
                    s["state"] = "active"
            p = subprocess.Popen(["docker", "exec", s["container"], *cmd],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = p.communicate(timeout=timeout_ms / 1000.0)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                _dex(s["container"], ["sh", "-c", TREE_KILL])
                out, err = p.communicate(timeout=10)
            duration_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            with self.lock:
                s["in_flight"] -= 1
                s["last_used"] = time.time()
        out, err = (out or b"")[:8 * 1024 * 1024], (err or b"")[:8 * 1024 * 1024]
        result = {"command_id": cid, "session_id": sid, "exit_code": p.returncode,
                  "timed_out": timed_out, "category": "timeout" if timed_out else "ok",
                  "duration_ms": round(duration_ms, 3),
                  "stdout_b64": base64.b64encode(out).decode(),
                  "stderr_b64": base64.b64encode(err).decode(),
                  "stdout_bytes": len(out), "stderr_bytes": len(err)}
        if request_id and self.state_store == "journal":
            self._append("command_completed", request_id=request_id, sid=sid, result=result)
            self.dedup[request_id] = result
        return result

    def status(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        return {"session_id": sid, "state": s["state"],
                "containers_alive": len(dockerops.ps_ids(
                    f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False)),
                "workspace_exists": os.path.isdir(s["workspace"]),
                "dedup_entries": len(self.dedup),
                "interrupted_entries": len(self.interrupted)}

    def stop(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        self._kill_container(sid)
        if os.path.isdir(s["workspace"]) and s["state"] != "released_keep":
            shutil.rmtree(s["workspace"], ignore_errors=True)
        with self.lock:
            s["state"] = "stopped"
        self._append("session_stopped", sid=sid)
        return {"stopped": True}

    def reconcile(self) -> dict:
        """清理有本章 label 但无状态记录的孤儿容器（重启后对账）。"""
        known = {f"{dockerops.LABEL_SESSION}={sid}" for sid in self.sessions}
        orphans = []
        r = dockerops.docker(["ps", "-q", "--filter", f"label={dockerops.LABEL_CHAPTER}={CHAPTER}"])
        for cid in [x for x in r.stdout.split() if x]:
            ins = dockerops.docker(["inspect", "--format",
                                    "{{range .Config.Labels}}{{.}} {{end}}", cid])
            # 通过逐 label 查询判断归属
            matched = False
            for lab in known:
                rr = dockerops.docker(["ps", "-q", "--filter", f"id={cid}",
                                       "--filter", lab])
                if rr.stdout.strip():
                    matched = True
                    break
            if not matched:
                orphans.append(cid)
        for cid in orphans:
            dockerops.kill_ids([cid])
            dockerops.docker(["rm", "-f", cid])
        return {"orphans_removed": orphans}

    def mark_interrupted(self, request_id, sid):
        """由外部（崩溃模拟前的 accepted 记录）标记：执行中被打断。"""
        self.interrupted[request_id] = {"sid": sid, "ts": time.time()}
        self._append("command_interrupted", request_id=request_id, sid=sid)

    def shutdown(self):
        self._stop_evt.set()


def _iso(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat(timespec="seconds")


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            try:
                if self.path == "/health":
                    return self._send(200, {"ok": True})
                parts = self.path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "status":
                    return self._send(200, service.status(parts[1]))
                self._send(404, {"error": "no_route"})
            except ApiError as e:
                self._send(e.http_code, {"error": e.code, "message": e.message})
            except Exception as e:
                self._send(500, {"error": "internal", "message": str(e)})

        def do_POST(self):
            try:
                parts = self.path.strip("/").split("/")
                if parts == ["sessions"]:
                    return self._send(200, service.create(self._body()))
                if parts == ["admin", "reconcile"]:
                    return self._send(200, service.reconcile())
                if parts == ["admin", "mark_interrupted"]:
                    b = self._body()
                    service.mark_interrupted(b["request_id"], b["sid"])
                    return self._send(200, {"marked": True})
                if len(parts) == 3 and parts[0] == "sessions":
                    if parts[2] == "exec":
                        return self._send(200, service.exec(parts[1], self._body()))
                    if parts[2] == "stop":
                        return self._send(200, service.stop(parts[1]))
                self._send(404, {"error": "no_route"})
            except ApiError as e:
                self._send(e.http_code, {"error": e.code, "message": e.message})
            except Exception as e:
                self._send(500, {"error": "internal", "message": str(e)})

        def log_message(self, *a):
            pass

    return Handler


def serve(port: int = 8113, state_store: str = "journal"):
    if state_store == "journal":
        dockerops.cleanup_chapter(CHAPTER)
    images.ensure(images.PINNED["alpine"])
    srv = ThreadingHTTPServer(("127.0.0.1", port),
                              make_handler(SandboxService(state_store=state_store)))
    print(f"chapter_13 service({state_store}) on http://127.0.0.1:{srv.server_address[1]}",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8113
    store = sys.argv[2] if len(sys.argv) > 2 else "journal"
    serve(port, store)
