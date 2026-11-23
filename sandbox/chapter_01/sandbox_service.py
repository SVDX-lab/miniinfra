"""第 01 期：最小 Agent 执行闭环（一次性容器 baseline）。

API v1 见 API.md。每条 exec 命令在独立的一次性受约束容器中运行；
会话 = 宿主侧工作区 + 记录，无常驻容器（常驻是第 02 期的变量）。
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

CHAPTER = "c01"
MAX_OUTPUT_BYTES = 8 * 1024 * 1024  # 服务侧硬上限，防止内存事故；机制性限额是第 08 期变量
MAX_FILE_BYTES = 1024 * 1024

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


class SandboxService:
    def __init__(self, base_dir: str | None = None, image: str | None = None):
        chapter_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir or os.path.join(chapter_dir, "data_local", "workspaces")
        os.makedirs(self.base_dir, exist_ok=True)
        self.image = image or images.PINNED["alpine"]
        self.sessions: dict[str, dict] = {}
        self.lock = threading.Lock()

    # ---- 会话生命周期 ----
    def create(self, payload: dict) -> dict:
        sid = "s-" + uuid.uuid4().hex[:12]
        ws = os.path.join(self.base_dir, sid)
        os.makedirs(ws, exist_ok=True)
        os.chmod(ws, 0o777)  # 容器内 uid 65534 需要写权限；课程测试目录
        with self.lock:
            self.sessions[sid] = {
                "workspace": ws,
                "created_at": time.time(),
                "state": "active",
                "image": payload.get("image") or self.image,
            }
        return {"session_id": sid, "created_at": _iso(self.sessions[sid]["created_at"])}

    def _get(self, sid: str) -> dict:
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", f"unknown session {sid}")
        return s

    # ---- 执行 ----
    def exec(self, sid: str, payload: dict) -> dict:
        s = self._get(sid)
        if s["state"] != "active":
            raise ApiError(409, "session_stopped", "session is stopped")
        cmd = payload.get("cmd")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            raise ApiError(400, "bad_cmd", "cmd must be a non-empty list of strings")
        timeout_ms = int(payload.get("timeout_ms", 10000))
        cid = "c-" + uuid.uuid4().hex[:12]
        name = f"mis-{CHAPTER}-{sid[2:8]}-{cid[2:8]}"
        argv = ["docker", "run", "--rm", "--name", name,
                f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
                f"--label={dockerops.LABEL_SESSION}={sid}",
                *CONSTRAINT_ARGS,
                "-v", f"{s['workspace']}:/workspace", "-w", "/workspace",
                s["image"], *cmd]
        t0 = time.perf_counter()
        timed_out = False
        try:
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = p.communicate(timeout=timeout_ms / 1000.0)
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "kill", name], capture_output=True, timeout=60)
                out, err = p.communicate()
                timed_out = True
            duration_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            return {"command_id": cid, "session_id": sid, "category": "internal",
                    "error": str(e), "duration_ms": (time.perf_counter() - t0) * 1000.0}
        exit_code = p.returncode
        out = (out or b"")[:MAX_OUTPUT_BYTES]
        err = (err or b"")[:MAX_OUTPUT_BYTES]
        if timed_out:
            category = "timeout"
        elif exit_code == 125:
            category = "env_prep_error"  # docker 守护进程/镜像层错误，非命令本身
        else:
            category = "ok"
        return {
            "command_id": cid, "session_id": sid, "exit_code": exit_code,
            "timed_out": timed_out, "category": category,
            "duration_ms": round(duration_ms, 3),
            "stdout_b64": base64.b64encode(out).decode(),
            "stderr_b64": base64.b64encode(err).decode(),
            "stdout_bytes": len(out), "stderr_bytes": len(err),
        }

    # ---- 文件 ----
    def list_files(self, sid: str) -> dict:
        s = self._get(sid)
        root = os.path.realpath(s["workspace"])
        entries = []
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            st = os.stat(p)
            entries.append({"name": name, "size": st.st_size, "mtime": st.st_mtime,
                            "is_dir": os.path.isdir(p)})
        return {"entries": entries}

    def read_file(self, sid: str, rel: str) -> dict:
        s = self._get(sid)
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

    # ---- 停止与状态 ----
    def stop(self, sid: str) -> dict:
        s = self._get(sid)
        ids = dockerops.ps_ids(f"{dockerops.LABEL_SESSION}={sid}")
        dockerops.kill_ids(ids)
        if ids:
            dockerops.docker(["rm", "-f", *ids])
        ws = s["workspace"]
        removed = False
        if os.path.isdir(ws):
            shutil.rmtree(ws, ignore_errors=True)
            removed = True
        with self.lock:
            s["state"] = "stopped"
        return {"stopped": True, "containers_killed": len(ids), "workspace_removed": removed}

    def status(self, sid: str) -> dict:
        s = self._get(sid)
        return {
            "session_id": sid, "state": s["state"],
            "created_at": _iso(s["created_at"]),
            "containers_alive": len(dockerops.ps_ids(
                f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False)),
            "workspace_exists": os.path.isdir(s["workspace"]),
        }


class ApiError(Exception):
    def __init__(self, http_code: int, code: str, message: str):
        super().__init__(message)
        self.http_code, self.code, self.message = http_code, code, message


def _iso(ts: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat(timespec="seconds")


def make_handler(service: SandboxService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
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
                self._send(404, {"error": "no_route", "path": u.path})
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
                self._send(404, {"error": "no_route", "path": u.path})
            except ApiError as e:
                self._send(e.http_code, {"error": e.code, "message": e.message})
            except Exception as e:
                self._send(500, {"error": "internal", "message": str(e)})

        def log_message(self, *a):
            pass

    return Handler


def serve(port: int = 8101) -> ThreadingHTTPServer:
    dockerops.cleanup_chapter(CHAPTER)
    images.ensure(images.PINNED["alpine"])
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(SandboxService()))
    print(f"chapter_01 service on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8101)
