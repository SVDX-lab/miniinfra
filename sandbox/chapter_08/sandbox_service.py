"""第 08 期：输出与回压。唯一变量：进程内缓冲（buffer）vs 流式限额（stream）。

stream 模式核心语义：
- 增量排水 stdout/stderr（排水线程永不停止读取 → 命令不会因管道满而阻塞）；
- 内存只保留头 512KiB + 尾 512KiB，全量只记 total_bytes；
- truncated 标记 + 头尾可拼接核对；退出码照常返回；
- 超时/取消走第 03 期 tree 清理。
"""
from __future__ import annotations

import base64
import collections
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

CHAPTER = "c08"
DEFAULT_TTL_MS = 60000
STREAM_KEEP_HEAD = 512 * 1024
STREAM_KEEP_TAIL = 512 * 1024

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


class _BoundedCapture:
    """持续排水一个管道，只保留头/尾字节与全量计数（本期核心机制）。"""

    def __init__(self):
        self.total = 0
        self.head = bytearray()
        self.tail = collections.deque(maxlen=STREAM_KEEP_TAIL)
        self.tail_bytes = 0
        self.lock = threading.Lock()

    def drain(self, pipe):
        while True:
            chunk = pipe.read(65536)
            if not chunk:
                return
            with self.lock:
                self.total += len(chunk)
                if len(self.head) < STREAM_KEEP_HEAD:
                    self.head.extend(chunk[:STREAM_KEEP_HEAD - len(self.head)])
                self.tail.append(chunk)
                self.tail_bytes += len(chunk)
                if self.tail_bytes > STREAM_KEEP_TAIL:
                    drop = self.tail_bytes - STREAM_KEEP_TAIL
                    first = self.tail[0]
                    if len(first) <= drop:
                        self.tail.popleft()
                        self.tail_bytes -= len(first)
                    else:
                        self.tail[0] = first[drop:]
                        self.tail_bytes -= drop

    def result(self) -> dict:
        with self.lock:
            tail = b"".join(self.tail)[-STREAM_KEEP_TAIL:]
            head = bytes(self.head)
        return {"total_bytes": self.total,
                "head_b64": base64.b64encode(head).decode(),
                "tail_b64": base64.b64encode(tail).decode(),
                "kept_bytes": len(head) + len(tail),
                "truncated": self.total > len(head) + len(tail)}


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
        output_mode = payload.get("output_mode", "buffer")
        if output_mode not in ("buffer", "stream"):
            raise ApiError(400, "bad_output_mode", "output_mode must be buffer|stream")
        sid = "s-" + uuid.uuid4().hex[:12]
        ws = os.path.join(self.base_dir, sid)
        os.makedirs(ws, exist_ok=True)
        os.chmod(ws, 0o777)
        s = {"workspace": ws, "created_at": time.time(), "state": "active",
             "output_mode": output_mode, "ttl_ms": int(payload.get("ttl_ms", self.ttl_ms)),
             "last_used": time.time(), "container": None}
        with self.lock:
            self.sessions[sid] = s
        name = f"mis-{CHAPTER}-{sid[2:8]}"
        r = subprocess.run(
            ["docker", "run", "-d", "--name", name,
             f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
             f"--label={dockerops.LABEL_SESSION}={sid}",
             *CONSTRAINT_ARGS, "-v", f"{ws}:/workspace", "-w", "/workspace",
             images.PINNED["alpine"], "sleep", "infinity"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            s["state"] = "broken"
            raise ApiError(500, "env_prep_error", r.stderr.strip()[:200])
        s["container"] = name
        return {"session_id": sid, "created_at": _iso(s["created_at"]),
                "output_mode": output_mode, "ttl_ms": s["ttl_ms"]}

    def _get_active(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        if s["state"] != "active":
            raise ApiError(409, f"session_{s['state']}", f"session is {s['state']}")
        return s

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
        try:
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if s["output_mode"] == "stream":
                cap_out, cap_err = _BoundedCapture(), _BoundedCapture()
                threads = [threading.Thread(target=c.drain, args=(pipe,), daemon=True)
                           for c, pipe in ((cap_out, p.stdout), (cap_err, p.stderr))]
                for t in threads:
                    t.start()
                try:
                    p.wait(timeout=timeout_ms / 1000.0)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _dex(s["container"], ["sh", "-c", TREE_KILL])
                    p.wait(timeout=10)
                for t in threads:
                    t.join(timeout=10)
                duration_ms = (time.perf_counter() - t0) * 1000.0
                res = {"command_id": cid, "session_id": sid,
                       "exit_code": p.returncode, "timed_out": timed_out,
                       "category": "timeout" if timed_out else "ok",
                       "duration_ms": round(duration_ms, 3),
                       "output_mode": "stream",
                       "stdout": cap_out.result(), "stderr": cap_err.result()}
            else:
                try:
                    out, err = p.communicate(timeout=timeout_ms / 1000.0)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _dex(s["container"], ["sh", "-c", TREE_KILL])
                    out, err = p.communicate(timeout=10)
                duration_ms = (time.perf_counter() - t0) * 1000.0
                res = {"command_id": cid, "session_id": sid,
                       "exit_code": p.returncode, "timed_out": timed_out,
                       "category": "timeout" if timed_out else "ok",
                       "duration_ms": round(duration_ms, 3),
                       "output_mode": "buffer",
                       "stdout_b64": base64.b64encode(out or b"").decode(),
                       "stderr_b64": base64.b64encode(err or b"").decode(),
                       "stdout_bytes": len(out or b""), "stderr_bytes": len(err or b"")}
        except Exception as e:
            return {"command_id": cid, "session_id": sid, "category": "internal",
                    "error": str(e)}
        s["last_used"] = time.time()
        return res

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
                "output_mode": s["output_mode"],
                "containers_alive": len(dockerops.ps_ids(
                    f"{dockerops.LABEL_SESSION}={sid}", include_stopped=False))}

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


def serve(port: int = 8108):
    dockerops.cleanup_chapter(CHAPTER)
    images.ensure(images.PINNED["alpine"])
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(SandboxService()))
    print(f"chapter_08 service on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8108)
