"""第 11 期：空闲会话回收。唯一变量：始终常驻 vs 到期释放并保留工作区供重建。

基座 = per-session + tree 取消 + loose 配额。本期核心机制：
- TTL 到期：删容器、保工作区、state=released（不删用户数据）；
- released 会话的 exec：透明重建容器（挂同一工作区），state 回 active，记录 restore_ms；
- 后台进程不随重建恢复（工作区恢复≠进程恢复）。
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

CHAPTER = "c11"
DEFAULT_TTL_MS = 1500  # 本期实验默认开启到期释放

CONSTRAINT_ARGS = [
    "--network", "none", "--read-only", "--user", "65534:65534",
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
    "--memory", "256m", "--memory-swap", "256m", "--cpus", "1", "--pids-limit", "64",
    "--tmpfs", "/tmp:rw,size=16m,noexec",
]

PID_WRAP = 'echo $$ > /tmp/mis_main.pid; exec "$@"'
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
    def __init__(self, base_dir=None, ttl_ms=DEFAULT_TTL_MS):
        chapter_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir or os.path.join(chapter_dir, "data_local", "workspaces")
        os.makedirs(self.base_dir, exist_ok=True)
        self.ttl_ms = ttl_ms
        self.sessions: dict[str, dict] = {}
        self.lock = threading.Lock()
        self._stop_evt = threading.Event()
        threading.Thread(target=self._ttl_loop, daemon=True).start()

    # ---- TTL：到期释放（删容器、保工作区）----
    # 释放条件 = "空闲且无在途命令"：exec 开始即更新 last_used 并计数在途，
    # reaper 跳过有在途命令的会话，kill 前持锁复查——杜绝 TOCTOU 窗口。
    def _ttl_loop(self):
        while not self._stop_evt.wait(0.25):
            now = time.time()
            with self.lock:
                due = [sid for sid, s in self.sessions.items()
                       if s["state"] == "active" and s["container"]
                       and s["in_flight"] == 0
                       and now - s["last_used"] > s["ttl_ms"] / 1000.0]
                for sid in due:
                    self.sessions[sid]["state"] = "released"
            for sid in due:
                with self.lock:
                    s = self.sessions.get(sid)
                    # 复查：标记与 kill 之间可能有命令进入（重建），避免误杀新容器
                    if not s or s["state"] != "released" or s["in_flight"] > 0:
                        continue
                self._kill_container(sid)

    def _kill_container(self, sid):
        ids = dockerops.ps_ids(f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False)
        dockerops.kill_ids(ids)
        if ids:
            dockerops.docker(["rm", "-f", *ids])

    def _start_container(self, s) -> None:
        name = f"mis-{CHAPTER}-{s['sid'][2:8]}"
        r = subprocess.run(
            ["docker", "run", "-d", "--name", name,
             f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
             f"--label={dockerops.LABEL_SESSION}={s['sid']}",
             *CONSTRAINT_ARGS, "-v", f"{s['workspace']}:/workspace", "-w", "/workspace",
             images.PINNED["alpine"], "sleep", "infinity"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise ApiError(500, "env_prep_error", r.stderr.strip()[:200])
        s["container"] = name

    def create(self, payload: dict) -> dict:
        sid = "s-" + uuid.uuid4().hex[:12]
        ws = os.path.join(self.base_dir, sid)
        os.makedirs(ws, exist_ok=True)
        os.chmod(ws, 0o777)
        s = {"sid": sid, "workspace": ws, "created_at": time.time(), "state": "active",
             "ttl_ms": int(payload.get("ttl_ms", self.ttl_ms)),
             "last_used": time.time(), "container": None, "restores": 0,
             "in_flight": 0}
        with self.lock:
            self.sessions[sid] = s
        self._start_container(s)
        return {"session_id": sid, "created_at": _iso(s["created_at"]),
                "ttl_ms": s["ttl_ms"]}

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
        timeout_ms = int(payload.get("timeout_ms", 10000))
        cid = "c-" + uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        with self.lock:
            s["last_used"] = time.time()
            s["in_flight"] += 1
        try:
            return self._exec_inner(s, payload, cmd, timeout_ms, cid, t0)
        finally:
            with self.lock:
                s["in_flight"] -= 1
                s["last_used"] = time.time()

    def _exec_inner(self, s, payload, cmd, timeout_ms, cid, t0) -> dict:
        sid = s["sid"]
        # 透明重建：released 会话下次命令自动恢复容器（工作区不变）
        restored = False
        if s["state"] == "released":
            self._kill_container(s["sid"])  # 防御：清理可能的同名残留
            self._start_container(s)
            with self.lock:
                s["state"] = "active"
            s["restores"] += 1
            restored = True
        restore_ms = (time.perf_counter() - t0) * 1000.0 if restored else 0.0
        argv = ["docker", "exec", s["container"], "sh", "-c", PID_WRAP, "mis", *cmd]
        timed_out = False
        try:
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = p.communicate(timeout=timeout_ms / 1000.0)
            except subprocess.TimeoutExpired:
                timed_out = True
                _dex(s["container"], ["sh", "-c", TREE_KILL])
                out, err = p.communicate(timeout=10)
            duration_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            return {"command_id": cid, "session_id": sid, "category": "internal",
                    "error": str(e)}
        s["last_used"] = time.time()
        out, err = (out or b"")[:8 * 1024 * 1024], (err or b"")[:8 * 1024 * 1024]
        return {"command_id": cid, "session_id": sid, "exit_code": p.returncode,
                "timed_out": timed_out, "category": "timeout" if timed_out else "ok",
                "duration_ms": round(duration_ms, 3),
                "restored_from_released": restored,
                "restore_ms": round(restore_ms, 1),
                "restores_total": s["restores"],
                "stdout_b64": base64.b64encode(out).decode(),
                "stderr_b64": base64.b64encode(err).decode(),
                "stdout_bytes": len(out), "stderr_bytes": len(err)}

    def stop(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        self._kill_container(sid)
        if os.path.isdir(s["workspace"]):
            shutil.rmtree(s["workspace"], ignore_errors=True)
        with self.lock:
            s["state"] = "stopped"
        return {"stopped": True}

    def status(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        return {"session_id": sid, "state": s["state"],
                "containers_alive": len(dockerops.ps_ids(
                    f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False)),
                "workspace_exists": os.path.isdir(s["workspace"]),
                "restores_total": s["restores"]}


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


def serve(port: int = 8111):
    dockerops.cleanup_chapter(CHAPTER)
    images.ensure(images.PINNED["alpine"])
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(SandboxService()))
    print(f"chapter_11 service on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8111)
