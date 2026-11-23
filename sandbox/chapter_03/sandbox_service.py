"""第 03 期：进程树和取消。唯一变量：超时清理路径（main-only vs tree）。

基座为第 02 期 per-session 常驻容器。exec 统一用 sh 包装记录主进程 PID；
超时时按会话的 cancel_mode 选择清理路径，会话本身保留（对比第 02 期粗粒度杀容器）：
- main-only: docker exec kill <主PID>（孤儿留在容器里）
- tree: docker exec 杀除 PID 1 与清理 shell 自身外的全部进程，并轮询确认归零
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images

CHAPTER = "c03"
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
DEFAULT_TTL_MS = 60000

CONSTRAINT_ARGS = [
    "--network", "none", "--read-only", "--user", "65534:65534",
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
    "--memory", "256m", "--memory-swap", "256m", "--cpus", "1", "--pids-limit", "64",
    "--tmpfs", "/tmp:rw,size=16m,noexec",
]

# 主进程 PID 包装：exec "$@" 后 PID 不变，/tmp/mis_main.pid 即主进程 PID
PID_WRAP = 'echo $$ > /tmp/mis_main.pid; exec "$@"'
# 主进程保护：PID1(tini) 与其子进程 `sleep infinity` 是容器基底（含僵尸收割者），
# 不属任务残留；tree 清理与进程清点都必须跳过它们，否则会误杀主进程导致容器退出。
MAIN_GUARD = ('pp=$(grep -m1 "^PPid" $d/status 2>/dev/null | awk "{print \\$2}"); '
              'if [ "$pp" = 1 ]; then c=$(tr "\\0" " " < $d/cmdline 2>/dev/null); '
              'case "$c" in "sleep infinity"|"sleep infinity ") continue;; esac; fi; ')
# 进程树清理：杀除 PID1、清理 shell 自身与容器基底外的全部进程
TREE_KILL = ('p=$$; for d in /proc/[0-9]*; do x=${d#/proc/}; '
             '[ "$x" = 1 ] && continue; [ "$x" = "$p" ] && continue; ' + MAIN_GUARD +
             'kill -9 "$x" 2>/dev/null; done; exit 0')


class ApiError(Exception):
    def __init__(self, http_code, code, message):
        super().__init__(message)
        self.http_code, self.code, self.message = http_code, code, message


def _dex(container: str, argv: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "exec", container, *argv],
                          capture_output=True, text=True, timeout=timeout)


def count_procs(container: str) -> tuple[int, int]:
    """返回 (存活任务进程数, 僵尸进程数)。

    统计范围排除容器基底：PID1（--init 的 tini，负责收割孤儿/僵尸）
    与主进程 `sleep infinity`（tini 的子进程）。清点脚本自身也排除。
    """
    r = _dex(container, ["sh", "-c",
                         'a=0; z=0; for d in /proc/[0-9]*; do x=${d#/proc/}; '
                         '[ "$x" = 1 ] && continue; [ "$x" = "$$" ] && continue; '
                         + MAIN_GUARD +
                         'read st < $d/stat || continue; s=${st#*) }; '
                         'case $s in Z*) z=$((z+1));; *) a=$((a+1));; esac; done; '
                         'echo $a $z'])
    try:
        a, z = r.stdout.split()
        return int(a), int(z)
    except ValueError:
        return -1, -1


class SandboxService:
    def __init__(self, base_dir=None, image=None, ttl_ms=DEFAULT_TTL_MS):
        chapter_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir or os.path.join(chapter_dir, "data_local", "workspaces")
        os.makedirs(self.base_dir, exist_ok=True)
        self.image = image or images.PINNED["alpine"]
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
        mode = payload.get("cancel_mode", "tree")
        if mode not in ("main-only", "tree"):
            raise ApiError(400, "bad_cancel_mode", "cancel_mode must be main-only|tree")
        sid = "s-" + uuid.uuid4().hex[:12]
        ws = os.path.join(self.base_dir, sid)
        os.makedirs(ws, exist_ok=True)
        os.chmod(ws, 0o777)
        s = {"workspace": ws, "created_at": time.time(), "state": "active",
             "image": payload.get("image") or self.image, "cancel_mode": mode,
             "ttl_ms": int(payload.get("ttl_ms", self.ttl_ms)),
             "last_used": time.time(), "container": None}
        with self.lock:
            self.sessions[sid] = s
        name = f"mis-{CHAPTER}-{sid[2:8]}"
        r = subprocess.run(
            ["docker", "run", "-d", "--name", name, "--init",
             f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
             f"--label={dockerops.LABEL_SESSION}={sid}",
             *CONSTRAINT_ARGS, "-v", f"{ws}:/workspace", "-w", "/workspace",
             s["image"], "sleep", "infinity"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            s["state"] = "broken"
            raise ApiError(500, "env_prep_error", r.stderr.strip()[:200])
        s["container"] = name
        return {"session_id": sid, "created_at": _iso(s["created_at"]),
                "cancel_mode": mode, "ttl_ms": s["ttl_ms"]}

    def _get_active(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        if s["state"] != "active":
            raise ApiError(409, f"session_{s['state']}", f"session is {s['state']}")
        return s

    # ---- 超时清理路径（本期唯一变量） ----
    def _cancel_main_only(self, s) -> dict:
        r = _dex(s["container"], ["cat", "/tmp/mis_main.pid"])
        pid = r.stdout.strip()
        if pid.isdigit():
            _dex(s["container"], ["kill", "-9", pid])
        time.sleep(0.15)
        alive, zombie = count_procs(s["container"])
        return {"path": "main-only", "killed_pid": pid,
                "leftover_alive": alive, "leftover_zombie": zombie}

    def _cancel_tree(self, s) -> dict:
        t0 = time.perf_counter()
        _dex(s["container"], ["sh", "-c", TREE_KILL])
        for _ in range(40):  # 轮询确认存活进程归零（僵尸由容器销毁带走）
            alive, zombie = count_procs(s["container"])
            if alive == 0:
                break
            time.sleep(0.05)
        return {"path": "tree",
                "cancel_to_zero_ms": round((time.perf_counter() - t0) * 1000, 1),
                "leftover_alive": alive, "leftover_zombie": zombie}

    def exec(self, sid, payload: dict) -> dict:
        s = self._get_active(sid)
        cmd = payload.get("cmd")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            raise ApiError(400, "bad_cmd", "cmd must be a non-empty list of strings")
        timeout_ms = int(payload.get("timeout_ms", 10000))
        cid = "c-" + uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        argv = ["docker", "exec", s["container"], "sh", "-c", PID_WRAP, "mis", *cmd]
        timed_out = False
        cleanup = None
        try:
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = p.communicate(timeout=timeout_ms / 1000.0)
            except subprocess.TimeoutExpired:
                timed_out = True
                cleanup = (self._cancel_main_only(s) if s["cancel_mode"] == "main-only"
                           else self._cancel_tree(s))
                out, err = p.communicate(timeout=10)
            duration_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            return {"command_id": cid, "session_id": sid, "category": "internal",
                    "error": str(e), "duration_ms": (time.perf_counter() - t0) * 1000.0}
        s["last_used"] = time.time()
        out, err = (out or b"")[:MAX_OUTPUT_BYTES], (err or b"")[:MAX_OUTPUT_BYTES]
        exit_code = p.returncode
        res = {"command_id": cid, "session_id": sid, "exit_code": exit_code,
               "timed_out": timed_out,
               "category": "timeout" if timed_out else ("env_prep_error" if exit_code == 125 else "ok"),
               "duration_ms": round(duration_ms, 3),
               "stdout_b64": base64.b64encode(out).decode(),
               "stderr_b64": base64.b64encode(err).decode(),
               "stdout_bytes": len(out), "stderr_bytes": len(err)}
        if cleanup:  # 失败不默默报成功：清理报告随命令返回
            res["cancel_report"] = cleanup
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
        import shutil
        if os.path.isdir(s["workspace"]):
            shutil.rmtree(s["workspace"], ignore_errors=True)
        with self.lock:
            s["state"] = "stopped"
        return {"stopped": True, "workspace_removed": not os.path.isdir(s["workspace"])}

    def status(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        return {"session_id": sid, "state": s["state"], "cancel_mode": s["cancel_mode"],
                "created_at": _iso(s["created_at"]),
                "containers_alive": len(dockerops.ps_ids(
                    f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False)),
                "workspace_exists": os.path.isdir(s["workspace"]),
                "procs_in_container": (count_procs(s["container"])
                                       if s.get("container") and s["state"] == "active" else None)}

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


def serve(port: int = 8103):
    dockerops.cleanup_chapter(CHAPTER)
    images.ensure(images.PINNED["alpine"])
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(SandboxService()))
    print(f"chapter_03 service on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8103)
