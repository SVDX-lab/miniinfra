"""第 16 期：在线云端 Agent。唯一变量：同步等待返回 vs 后台异步任务。

异步任务线程：per-session 容器（主线约束）→ 逐步执行录制轨迹并更新进度 →
产物写工作区 → 完成后可下载。取消走第 03 期 tree 清理语义（杀容器）。
任务归属：task 记录 user token，查询/取消/下载均校验（跨用户 404）。
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

CHAPTER = "c16"

CONSTRAINTS = ["--network", "none", "--read-only", "--user", "65534:65534",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--memory", "128m", "--cpus", "0.5", "--pids-limit", "32",
               "--tmpfs", "/tmp:rw,size=8m,noexec"]


class ApiError(Exception):
    def __init__(self, http_code, code, message):
        super().__init__(message)
        self.http_code, self.code, self.message = http_code, code, message


def sh(argv, timeout=60):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def new_session(user: str, ws_root: str) -> tuple[str, str]:
    sid = f"s-{uuid.uuid4().hex[:12]}"
    ws = os.path.join(ws_root, sid)
    os.makedirs(ws, exist_ok=True)
    os.chmod(ws, 0o777)
    r = sh(["docker", "run", "-d", "--name", f"mis-{CHAPTER}-{sid[2:10]}",
            f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
            f"--label={dockerops.LABEL_SESSION}={sid}",
            *CONSTRAINTS, "-v", f"{ws}:/workspace", "-w", "/workspace",
            images.PINNED["alpine"], "sleep", "infinity"])
    if r.returncode != 0:
        raise ApiError(500, "env_prep_error", r.stderr.strip()[:200])
    return sid, ws


def kill_session(sid: str):
    ids = dockerops.ps_ids(f"{dockerops.LABEL_SESSION}={sid}")
    dockerops.kill_ids(ids)
    if ids:
        dockerops.docker(["rm", "-f", *ids])


class TaskService:
    def __init__(self, base_dir=None):
        chapter_dir = os.path.dirname(os.path.abspath(__file__))
        self.ws_root = base_dir or os.path.join(chapter_dir, "data_local", "workspaces")
        os.makedirs(self.ws_root, exist_ok=True)
        self.tasks: dict[str, dict] = {}
        self.lock = threading.Lock()

    def _run_trajectory(self, tid: str, user: str, steps: list[list[str]]):
        t = self.tasks[tid]
        sid = None
        try:
            sid, ws = new_session(user, self.ws_root)
            with self.lock:
                t.update({"session": sid, "state": "running"})
            container = f"mis-{CHAPTER}-{sid[2:10]}"
            for i, cmd in enumerate(steps):
                with self.lock:
                    if t["state"] == "cancelled":
                        return
                    t["current_step"] = i
                r = sh(["docker", "exec", container, *cmd])
                with self.lock:
                    t["progress"] = {"done_steps": i + 1, "total_steps": len(steps)}
                    t["last_exit"] = r.returncode
                    if t["state"] == "cancelled":
                        return
            with self.lock:
                if t["state"] == "cancelled":
                    return
                t["state"] = "done"
                t["workspace"] = ws
        except Exception as e:
            with self.lock:
                t.update({"state": "failed", "error": str(e)})
        finally:
            # 完成/失败/取消后都释放容器；工作区保留供产物下载（空闲成本归零）
            if sid:
                kill_session(sid)

    def submit(self, payload: dict, mode: str) -> dict:
        user = payload.get("user_token", "")
        steps = payload.get("trajectory")
        if not user or not isinstance(steps, list) or not steps:
            raise ApiError(400, "bad_request", "user_token and trajectory required")
        tid = "t-" + uuid.uuid4().hex[:12]
        with self.lock:
            self.tasks[tid] = {"user": user, "state": "queued",
                               "progress": {"done_steps": 0, "total_steps": len(steps)},
                               "submitted": time.time(), "mode": mode}
        if mode == "sync":
            self._run_trajectory(tid, user, steps)  # 阻塞到完成
            t = self.tasks[tid]
            if t["state"] != "done":
                raise ApiError(500, "task_failed", t.get("error", t["state"]))
            return self._view(tid, user) | {"completed": True}
        threading.Thread(target=self._run_trajectory, args=(tid, user, steps),
                         daemon=True).start()
        return {"task_id": tid, "state": "queued", "mode": mode}

    def _get_owned(self, tid: str, user: str) -> dict:
        t = self.tasks.get(tid)
        if not t or t["user"] != user:
            # 跨用户与不存在统一 404，不泄露任务存在性
            raise ApiError(404, "task_not_found", tid)
        return t

    def _view(self, tid: str, user: str) -> dict:
        t = self._get_owned(tid, user)
        return {"task_id": tid, "state": t["state"], "progress": t["progress"]}

    def cancel(self, tid: str, user: str) -> dict:
        t = self._get_owned(tid, user)
        with self.lock:
            t["state"] = "cancelled"
        if t.get("session"):
            kill_session(t["session"])
        return {"task_id": tid, "state": "cancelled"}

    def artifact(self, tid: str, user: str, path: str) -> dict:
        t = self._get_owned(tid, user)
        if t["state"] != "done":
            raise ApiError(409, "not_done", f"state={t['state']}")
        root = os.path.realpath(t["workspace"])
        target = os.path.realpath(os.path.join(root, path))
        if target != root and not target.startswith(root + os.sep):
            raise ApiError(400, "path_escape", "path escapes workspace")
        if not os.path.isfile(target):
            raise ApiError(404, "no_artifact", path)
        with open(target, "rb") as f:
            data = f.read(1024 * 1024)
        return {"task_id": tid, "path": path, "size": len(data),
                "content_b64": base64.b64encode(data).decode()}


def make_handler(service: TaskService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _auth(self):
            return self.headers.get("X-User-Token", "")

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            try:
                u = urlparse(self.path)
                if u.path == "/health":
                    return self._send(200, {"ok": True})
                parts = u.path.strip("/").split("/")
                if len(parts) == 2 and parts[0] == "tasks":
                    return self._send(200, service._view(parts[1], self._auth()))
                if len(parts) == 3 and parts[0] == "tasks" and parts[2] == "artifact":
                    q = parse_qs(u.query)
                    path = q.get("path", ["result.txt"])[0]
                    return self._send(200, service.artifact(parts[1], self._auth(), path))
                self._send(404, {"error": "no_route"})
            except ApiError as e:
                self._send(e.http_code, {"error": e.code, "message": e.message})
            except Exception as e:
                self._send(500, {"error": "internal", "message": str(e)})

        def do_POST(self):
            try:
                parts = self.path.strip("/").split("/")
                if parts == ["tasks", "sync"]:
                    return self._send(200, service.submit(self._body(), "sync"))
                if parts == ["tasks"]:
                    return self._send(200, service.submit(self._body(), "async"))
                if len(parts) == 3 and parts[0] == "tasks" and parts[2] == "cancel":
                    return self._send(200, service.cancel(parts[1], self._auth()))
                self._send(404, {"error": "no_route"})
            except ApiError as e:
                self._send(e.http_code, {"error": e.code, "message": e.message})
            except Exception as e:
                self._send(500, {"error": "internal", "message": str(e)})

        def log_message(self, *a):
            pass

    return Handler


def serve(port: int = 8116):
    dockerops.cleanup_chapter(CHAPTER)
    images.ensure(images.PINNED["alpine"])
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(TaskService()))
    print(f"chapter_16 service on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8116)
