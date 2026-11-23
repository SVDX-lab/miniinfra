"""第 04 期：资源配额。唯一变量：固定宽松额度 vs 按任务类型的配额 profile。

基座 = 第 03 期（per-session 常驻容器 + tree 取消）。本期把容器约束从
固定常量改为按 create 时选择的 quota profile 生成；每次扫描只改一类资源。
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

CHAPTER = "c04"
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
DEFAULT_TTL_MS = 60000

# 配额 profile：两组都保留绝对上限（≤2 cpu、≤512m、≤64 pids）
QUOTA_PROFILES = {
    "loose":     {"memory": "256m", "cpus": "1.0", "pids": 64},
    "light":     {"memory": "64m",  "cpus": "0.25", "pids": 32},
    "cpu_heavy": {"memory": "256m", "cpus": "2.0", "pids": 64},
    "mem_heavy": {"memory": "512m", "cpus": "1.0", "pids": 64},
    "half":      {"memory": "128m", "cpus": "0.5", "pids": 48},
}

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


def _constraint_args(profile: dict, cpuset: str | None = None) -> list[str]:
    args = [
        "--network", "none", "--read-only", "--user", "65534:65534",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", profile["memory"], "--memory-swap", profile["memory"],
        "--cpus", profile["cpus"], "--pids-limit", str(profile["pids"]),
        "--tmpfs", "/tmp:rw,size=16m,noexec",
    ]
    if cpuset:
        args += ["--cpuset-cpus", cpuset]
    return args


def oom_kill_count(container: str) -> int:
    r = _dex(container, ["cat", "/sys/fs/cgroup/memory.events"])
    for line in (r.stdout or "").splitlines():
        if line.startswith("oom_kill"):
            return int(line.split()[1])
    return -1


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

    def _ttl_loop(self):
        while not self._stop_evt.wait(0.25):
            now = time.time()
            with self.lock:
                dead = [sid for sid, s in self.sessions.items()
                        if s["state"] == "active" and now - s["last_used"] > s["ttl_ms"] / 1000.0]
                for sid in dead:
                    self.sessions[sid]["state"] = "expired"
            for sid in dead:
                self._kill_container(sid)

    def _kill_container(self, sid):
        ids = dockerops.ps_ids(f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False)
        dockerops.kill_ids(ids)
        if ids:
            dockerops.docker(["rm", "-f", *ids])

    def create(self, payload: dict) -> dict:
        profile_name = payload.get("quota_profile", "loose")
        if profile_name not in QUOTA_PROFILES:
            raise ApiError(400, "bad_profile", f"unknown quota profile {profile_name}")
        cpuset = payload.get("cpuset")  # 可选：cpuset 硬分区（如 "0,1"），配额之外的隔离手段
        image = payload.get("image") or images.PINNED["alpine"]
        sid = "s-" + uuid.uuid4().hex[:12]
        ws = os.path.join(self.base_dir, sid)
        os.makedirs(ws, exist_ok=True)
        os.chmod(ws, 0o777)
        s = {"workspace": ws, "created_at": time.time(), "state": "active",
             "image": image, "quota_profile": profile_name,
             "quota": QUOTA_PROFILES[profile_name], "cpuset": cpuset,
             "ttl_ms": int(payload.get("ttl_ms", self.ttl_ms)),
             "last_used": time.time(), "container": None}
        with self.lock:
            self.sessions[sid] = s
        name = f"mis-{CHAPTER}-{sid[2:8]}"
        r = subprocess.run(
            ["docker", "run", "-d", "--name", name,
             f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
             f"--label={dockerops.LABEL_SESSION}={sid}",
             *_constraint_args(s["quota"], cpuset),
             "-v", f"{ws}:/workspace", "-w", "/workspace",
             image, "sleep", "infinity"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            s["state"] = "broken"
            raise ApiError(500, "env_prep_error", r.stderr.strip()[:200])
        s["container"] = name
        return {"session_id": sid, "created_at": _iso(s["created_at"]),
                "quota_profile": profile_name, "quota": s["quota"],
                "ttl_ms": s["ttl_ms"]}

    def _get_active(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        if s["state"] != "active":
            raise ApiError(409, f"session_{s['state']}", f"session is {s['state']}")
        return s

    def _cancel_tree(self, s):
        t0 = time.perf_counter()
        _dex(s["container"], ["sh", "-c", TREE_KILL])
        return {"path": "tree", "cancel_to_zero_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def exec(self, sid, payload: dict) -> dict:
        s = self._get_active(sid)
        cmd = payload.get("cmd")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            raise ApiError(400, "bad_cmd", "cmd must be a non-empty list of strings")
        timeout_ms = int(payload.get("timeout_ms", 10000))
        cid = "c-" + uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        oom_before = oom_kill_count(s["container"])
        argv = ["docker", "exec", s["container"], "sh", "-c", PID_WRAP, "mis", *cmd]
        timed_out = False
        try:
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = p.communicate(timeout=timeout_ms / 1000.0)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._cancel_tree(s)
                out, err = p.communicate(timeout=10)
            duration_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            return {"command_id": cid, "session_id": sid, "category": "internal",
                    "error": str(e), "duration_ms": (time.perf_counter() - t0) * 1000.0}
        s["last_used"] = time.time()
        out, err = (out or b"")[:MAX_OUTPUT_BYTES], (err or b"")[:MAX_OUTPUT_BYTES]
        exit_code = p.returncode
        oom_after = oom_kill_count(s["container"])
        res = {"command_id": cid, "session_id": sid, "exit_code": exit_code,
               "timed_out": timed_out,
               "category": "timeout" if timed_out else ("env_prep_error" if exit_code == 125 else "ok"),
               "duration_ms": round(duration_ms, 3),
               "stdout_b64": base64.b64encode(out).decode(),
               "stderr_b64": base64.b64encode(err).decode(),
               "stdout_bytes": len(out), "stderr_bytes": len(err),
               "oom_kill_delta": (oom_after - oom_before) if oom_before >= 0 and oom_after >= 0 else None}
        if timed_out:
            res["cancel_report"] = self._cancel_tree.__name__  # 已在超时路径执行
        return res

    def list_files(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        root = os.path.realpath(s["workspace"])
        return {"entries": [{"name": n, "size": os.stat(os.path.join(root, n)).st_size,
                             "mtime": os.stat(os.path.join(root, n)).st_mtime,
                             "is_dir": os.path.isdir(os.path.join(root, n))}
                            for n in sorted(os.listdir(root))]}

    def read_file(self, sid, rel):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        root = os.path.realpath(s["workspace"])
        target = os.path.realpath(os.path.join(root, rel))
        if target != root and not target.startswith(root + os.sep):
            raise ApiError(400, "path_escape", "path escapes workspace")
        if not os.path.isfile(target):
            raise ApiError(404, "not_a_file", rel)
        with open(target, "rb") as f:
            data = f.read(1024 * 1024)
        return {"name": rel, "size": len(data), "content_b64": base64.b64encode(data).decode()}

    def stop(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        self._kill_container(sid)
        if os.path.isdir(s["workspace"]):
            shutil.rmtree(s["workspace"], ignore_errors=True)
        with self.lock:
            s["state"] = "stopped"
        return {"stopped": True, "workspace_removed": not os.path.isdir(s["workspace"])}

    def status(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        return {"session_id": sid, "state": s["state"],
                "quota_profile": s["quota_profile"],
                "created_at": _iso(s["created_at"]),
                "containers_alive": len(dockerops.ps_ids(
                    f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False)),
                "workspace_exists": os.path.isdir(s["workspace"])}

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


def serve(port: int = 8104):
    dockerops.cleanup_chapter(CHAPTER)
    images.ensure(images.PINNED["alpine"])
    images.ensure(images.PINNED["python"])
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(SandboxService()))
    print(f"chapter_04 service on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8104)
