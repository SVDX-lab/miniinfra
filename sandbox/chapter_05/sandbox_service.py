"""第 05 期：文件与产物。唯一变量：flat（单可写目录+拷贝输入）vs structured（ro 输入+工作区+导出）。

基座 = 第 04 期（per-session、tree 取消、loose 配额）。
新增 export API：白名单路径从工作区拷贝到导出区，realpath 校验防逃逸，记录 sha256。
"""
from __future__ import annotations

import base64
import hashlib
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

CHAPTER = "c05"
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
DEFAULT_TTL_MS = 60000
SEED = 42
INPUT_SIZE_BYTES = 32 * 1024 * 1024  # 32 MiB 确定性输入树

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


def build_shared_inputs(base: str) -> dict:
    """生成确定性共享输入树（种子固定，跨运行同哈希）。"""
    d = os.path.join(base, "shared_input")
    os.makedirs(d, exist_ok=True)
    dat = os.path.join(d, "input.dat")
    if not os.path.exists(dat) or os.path.getsize(dat) != INPUT_SIZE_BYTES:
        block = bytes(range(256)) * (INPUT_SIZE_BYTES // 256)
        with open(dat, "wb") as f:
            f.write(block)
    h = hashlib.sha256(open(dat, "rb").read()).hexdigest()
    with open(os.path.join(d, "input.txt"), "w") as f:
        f.write(f"seed={SEED}\ninput.dat.sha256={h}\n")
    return {"dir": d, "input_dat_sha256": h, "bytes": INPUT_SIZE_BYTES}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SandboxService:
    def __init__(self, base_dir=None, ttl_ms=DEFAULT_TTL_MS):
        chapter_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir or os.path.join(chapter_dir, "data_local", "workspaces")
        os.makedirs(self.base_dir, exist_ok=True)
        self.exports_dir = os.path.join(os.path.dirname(self.base_dir), "exports")
        os.makedirs(self.exports_dir, exist_ok=True)
        self.shared_input = build_shared_inputs(os.path.join(chapter_dir, "data_local"))
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
        layout = payload.get("layout", "flat")
        if layout not in ("flat", "structured"):
            raise ApiError(400, "bad_layout", "layout must be flat|structured")
        t0 = time.perf_counter()
        sid = "s-" + uuid.uuid4().hex[:12]
        ws = os.path.join(self.base_dir, sid)
        os.makedirs(ws, exist_ok=True)
        os.chmod(ws, 0o777)
        mounts = [f"{ws}:/workspace"]
        if layout == "flat":
            # 把共享输入树拷贝进工作区（每会话一份可变副本）
            shutil.copytree(self.shared_input["dir"], os.path.join(ws, "inputs"))
        else:
            # 共享输入只读挂载，零拷贝
            mounts.append(f"{self.shared_input['dir']}:/inputs:ro")
        s = {"workspace": ws, "created_at": time.time(), "state": "active",
             "layout": layout, "ttl_ms": int(payload.get("ttl_ms", self.ttl_ms)),
             "last_used": time.time(), "container": None,
             "prep_ms": (time.perf_counter() - t0) * 1000.0}
        with self.lock:
            self.sessions[sid] = s
        name = f"mis-{CHAPTER}-{sid[2:8]}"
        r = subprocess.run(
            ["docker", "run", "-d", "--name", name,
             f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
             f"--label={dockerops.LABEL_SESSION}={sid}",
             *CONSTRAINT_ARGS,
             *sum((["-v", m] for m in mounts), []),
             "-w", "/workspace", images.PINNED["alpine"], "sleep", "infinity"],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            s["state"] = "broken"
            raise ApiError(500, "env_prep_error", r.stderr.strip()[:200])
        s["container"] = name
        return {"session_id": sid, "created_at": _iso(s["created_at"]),
                "layout": layout, "prep_ms": round(s["prep_ms"], 3),
                "input_dat_sha256": self.shared_input["input_dat_sha256"],
                "ttl_ms": s["ttl_ms"]}

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
        out, err = (out or b"")[:MAX_OUTPUT_BYTES], (err or b"")[:MAX_OUTPUT_BYTES]
        exit_code = p.returncode
        return {"command_id": cid, "session_id": sid, "exit_code": exit_code,
                "timed_out": timed_out,
                "category": "timeout" if timed_out else ("env_prep_error" if exit_code == 125 else "ok"),
                "duration_ms": round(duration_ms, 3),
                "stdout_b64": base64.b64encode(out).decode(),
                "stderr_b64": base64.b64encode(err).decode(),
                "stdout_bytes": len(out), "stderr_bytes": len(err)}

    # ---- 导出（本期核心机制） ----
    def export(self, sid, payload: dict) -> dict:
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        paths = payload.get("paths")
        if not isinstance(paths, list) or not paths or not all(isinstance(x, str) for x in paths):
            raise ApiError(400, "bad_paths", "paths must be a list of strings")
        root = os.path.realpath(s["workspace"])
        out_dir = os.path.join(self.exports_dir, sid)
        os.makedirs(out_dir, exist_ok=True)
        manifest, rejected = [], []
        t0 = time.perf_counter()
        for rel in paths:
            target = os.path.realpath(os.path.join(root, rel))
            inside = target == root or target.startswith(root + os.sep)
            # 符号链接解析后必须仍在工作区内，且必须是普通文件
            if not inside or os.path.islink(os.path.join(root, rel)) and not inside:
                rejected.append({"path": rel, "reason": "path_escape"})
                continue
            if not inside:
                rejected.append({"path": rel, "reason": "path_escape"})
                continue
            if not os.path.isfile(target):
                rejected.append({"path": rel, "reason": "not_a_regular_file"})
                continue
            dest = os.path.join(out_dir, os.path.relpath(target, root))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(target, dest)
            manifest.append({"path": rel, "bytes": os.path.getsize(dest),
                             "sha256": sha256_file(dest)})
        return {"session_id": sid, "export_dir": out_dir,
                "manifest": manifest, "rejected": rejected,
                "export_ms": round((time.perf_counter() - t0) * 1000, 1)}

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
        return {"session_id": sid, "state": s["state"], "layout": s["layout"],
                "created_at": _iso(s["created_at"]), "prep_ms": round(s["prep_ms"], 1),
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
                    if parts[2] == "export":
                        return self._send(200, service.export(parts[1], self._body()))
                self._send(404, {"error": "no_route"})
            except ApiError as e:
                self._send(e.http_code, {"error": e.code, "message": e.message})
            except Exception as e:
                self._send(500, {"error": "internal", "message": str(e)})

        def log_message(self, *a):
            pass

    return Handler


def serve(port: int = 8105):
    dockerops.cleanup_chapter(CHAPTER)
    images.ensure(images.PINNED["alpine"])
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(SandboxService()))
    print(f"chapter_05 service on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8105)
