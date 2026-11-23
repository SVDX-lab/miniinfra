"""第 02 期：会话状态。唯一变量：每命令新容器（per-cmd）vs 每会话常驻容器（per-session）。

per-session 模式：create 时启动 `sleep infinity` 常驻容器（同一约束集），
exec 用 docker exec；后台进程因此跨命令存活。
TTL：默认 60s，到期回收容器（工作区恢复是第 11 期的变量）。
超时处理本期为粗粒度（杀整个容器），如实记录；精细进程树清理是第 03 期。
API 与第 01 期 v1 兼容，create 增加 mode 参数（本期实验变量）。
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
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images

CHAPTER = "c02"
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024
DEFAULT_TTL_MS = 60000

CONSTRAINT_ARGS = [
    "--network", "none",
    "--read-only",
    "--user", "65534:65534",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--memory", "256m",
    "--memory-swap", "256m",
    "--cpus", "1",
    "--pids-limit", "64",
    "--tmpfs", "/tmp:rw,size=16m,noexec",
]


class ApiError(Exception):
    def __init__(self, http_code: int, code: str, message: str):
        super().__init__(message)
        self.http_code, self.code, self.message = http_code, code, message


class SandboxService:
    def __init__(self, base_dir: str | None = None, image: str | None = None,
                 ttl_ms: int = DEFAULT_TTL_MS):
        chapter_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir or os.path.join(chapter_dir, "data_local", "workspaces")
        os.makedirs(self.base_dir, exist_ok=True)
        self.image = image or images.PINNED["alpine"]
        self.ttl_ms = ttl_ms
        self.sessions: dict[str, dict] = {}
        self.lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._ttl_thread = threading.Thread(target=self._ttl_loop, daemon=True)
        self._ttl_thread.start()

    # ---- TTL 回收 ----
    def _ttl_loop(self):
        while not self._stop_evt.wait(0.25):
            now = time.time()
            with self.lock:
                expired = [sid for sid, s in self.sessions.items()
                           if s["state"] == "active"
                           and s["mode"] == "per-session"
                           and now - s["last_used"] > s["ttl_ms"] / 1000.0]
                for sid in expired:
                    self.sessions[sid]["state"] = "expired"
            for sid in expired:
                self._kill_session_container(sid)

    def _kill_session_container(self, sid: str) -> int:
        ids = dockerops.ps_ids(f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False)
        dockerops.kill_ids(ids)
        if ids:
            dockerops.docker(["rm", "-f", *ids])
        return len(ids)

    # ---- 会话生命周期 ----
    def create(self, payload: dict) -> dict:
        mode = payload.get("mode", "per-cmd")
        if mode not in ("per-cmd", "per-session"):
            raise ApiError(400, "bad_mode", "mode must be per-cmd|per-session")
        sid = "s-" + uuid.uuid4().hex[:12]
        ws = os.path.join(self.base_dir, sid)
        os.makedirs(ws, exist_ok=True)
        os.chmod(ws, 0o777)
        s = {"workspace": ws, "created_at": time.time(), "state": "active",
             "image": payload.get("image") or self.image, "mode": mode,
             "ttl_ms": int(payload.get("ttl_ms", self.ttl_ms)),
             "last_used": time.time(), "container": None}
        with self.lock:
            self.sessions[sid] = s
        if mode == "per-session":
            name = f"mis-{CHAPTER}-{sid[2:8]}"
            r = subprocess.run(
                ["docker", "run", "-d", "--name", name,
                 f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
                 f"--label={dockerops.LABEL_SESSION}={sid}",
                 *CONSTRAINT_ARGS,
                 "-v", f"{ws}:/workspace", "-w", "/workspace",
                 s["image"], "sleep", "infinity"],
                capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                s["state"] = "broken"
                raise ApiError(500, "env_prep_error", r.stderr.strip()[:200])
            s["container"] = name
        return {"session_id": sid, "created_at": _iso(s["created_at"]),
                "mode": mode, "ttl_ms": s["ttl_ms"]}

    def _get_active(self, sid: str) -> dict:
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        if s["state"] != "active":
            raise ApiError(409, f"session_{s['state']}", f"session is {s['state']}")
        return s

    # ---- 执行 ----
    def exec(self, sid: str, payload: dict) -> dict:
        s = self._get_active(sid)
        cmd = payload.get("cmd")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            raise ApiError(400, "bad_cmd", "cmd must be a non-empty list of strings")
        timeout_ms = int(payload.get("timeout_ms", 10000))
        cid = "c-" + uuid.uuid4().hex[:12]
        t0 = time.perf_counter()

        if s["mode"] == "per-session":
            argv = ["docker", "exec", s["container"], *cmd]
        else:
            name = f"mis-{CHAPTER}-{sid[2:8]}-{cid[2:8]}"
            argv = ["docker", "run", "--rm", "--name", name,
                    f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
                    f"--label={dockerops.LABEL_SESSION}={sid}",
                    *CONSTRAINT_ARGS,
                    "-v", f"{s['workspace']}:/workspace", "-w", "/workspace",
                    s["image"], *cmd]

        timed_out = False
        try:
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = p.communicate(timeout=timeout_ms / 1000.0)
            except subprocess.TimeoutExpired:
                if s["mode"] == "per-session":
                    # 粗粒度超时：杀整个会话容器（本期已知限制，第 03 期细化）
                    dockerops.kill_ids([s["container"]])
                    dockerops.docker(["rm", "-f", s["container"]])
                    s["container"] = None
                    with self.lock:
                        s["state"] = "stopped"
                    s["stop_reason"] = "command_timeout_coarse"
                else:
                    subprocess.run(["docker", "kill", name], capture_output=True, timeout=60)
                out, err = p.communicate()
                timed_out = True
            duration_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            return {"command_id": cid, "session_id": sid, "category": "internal",
                    "error": str(e), "duration_ms": (time.perf_counter() - t0) * 1000.0}

        s["last_used"] = time.time()
        out, err = (out or b"")[:MAX_OUTPUT_BYTES], (err or b"")[:MAX_OUTPUT_BYTES]
        exit_code = p.returncode
        category = ("timeout" if timed_out
                    else "env_prep_error" if exit_code == 125 else "ok")
        return {"command_id": cid, "session_id": sid, "exit_code": exit_code,
                "timed_out": timed_out, "category": category,
                "duration_ms": round(duration_ms, 3),
                "stdout_b64": base64.b64encode(out).decode(),
                "stderr_b64": base64.b64encode(err).decode(),
                "stdout_bytes": len(out), "stderr_bytes": len(err)}

    # ---- 文件 ----
    def list_files(self, sid: str) -> dict:
        s = self.sessions.get(sid) or self._raise404(sid)
        root = os.path.realpath(s["workspace"])
        entries = [{"name": n, "size": os.stat(os.path.join(root, n)).st_size,
                    "mtime": os.stat(os.path.join(root, n)).st_mtime,
                    "is_dir": os.path.isdir(os.path.join(root, n))}
                   for n in sorted(os.listdir(root))]
        return {"entries": entries}

    def read_file(self, sid: str, rel: str) -> dict:
        s = self.sessions.get(sid) or self._raise404(sid)
        root = os.path.realpath(s["workspace"])
        target = os.path.realpath(os.path.join(root, rel))
        if target != root and not target.startswith(root + os.sep):
            raise ApiError(400, "path_escape", "path escapes workspace")
        if not os.path.isfile(target):
            raise ApiError(404, "not_a_file", rel)
        if os.path.getsize(target) > MAX_FILE_BYTES:
            raise ApiError(413, "file_too_large", "over 1MiB")
        with open(target, "rb") as f:
            data = f.read(MAX_FILE_BYTES)
        return {"name": rel, "size": len(data), "content_b64": base64.b64encode(data).decode()}

    def _raise404(self, sid: str):
        raise ApiError(404, "session_not_found", sid)

    # ---- 停止与状态 ----
    def stop(self, sid: str) -> dict:
        s = self.sessions.get(sid) or self._raise404(sid)
        n = self._kill_session_container(sid)
        ws = s["workspace"]
        if os.path.isdir(ws):
            shutil.rmtree(ws, ignore_errors=True)
        with self.lock:
            s["state"] = "stopped"
        return {"stopped": True, "containers_killed": n,
                "workspace_removed": not os.path.isdir(ws)}

    def status(self, sid: str) -> dict:
        s = self.sessions.get(sid) or self._raise404(sid)
        return {"session_id": sid, "state": s["state"], "mode": s["mode"],
                "created_at": _iso(s["created_at"]),
                "containers_alive": len(dockerops.ps_ids(
                    f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False)),
                "workspace_exists": os.path.isdir(s["workspace"]),
                "idle_seconds": round(time.time() - s["last_used"], 1)}

    def shutdown(self):
        self._stop_evt.set()


def _iso(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat(timespec="seconds")


def make_handler(service: SandboxService):
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
                u = urlparse(self.path)
                if u.path == "/health":
                    return self._send(200, {"ok": True})
                parts = u.path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "files":
                    q = parse_qs(u.query)
                    if "path" in q:
                        return self._send(200, service.read_file(parts[1], q["path"][0]))
                    return self._send(200, service.list_files(parts[1]))
                if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "status":
                    return self._send(200, service.status(parts[1]))
                self._send(404, {"error": "no_route"})
            except ApiError as e:
                self._send(e.http_code, {"error": e.code, "message": e.message})
            except Exception as e:
                self._send(500, {"error": "internal", "message": str(e)})

        def do_POST(self):
            try:
                u = urlparse(self.path)
                parts = u.path.strip("/").split("/")
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


def serve(port: int = 8102):
    dockerops.cleanup_chapter(CHAPTER)
    images.ensure(images.PINNED["alpine"])
    srv = ThreadingHTTPServer(("127.0.0.1", port),
                              make_handler(SandboxService()))
    print(f"chapter_02 service on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8102)
