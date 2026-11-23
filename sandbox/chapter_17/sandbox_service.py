"""第 17 期：综合验收系统。组合第 03/04/08/11/13 期机制，不新增隔离原语。

机制清单：配额 profile、tree 取消、流式限额输出、TTL 释放+透明恢复、
journal 持久化 + request_id 幂等。全部可按 create 参数独立开关（阶段扫描用）。
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images

CHAPTER = "c17"
STREAM_KEEP_HEAD = 512 * 1024
STREAM_KEEP_TAIL = 512 * 1024

PROFILES = {"loose": {"memory": "256m", "cpus": "1.0", "pids": 64},
            "light": {"memory": "64m", "cpus": "0.25", "pids": 32}}
# 主进程保护：PID1(tini) 与其子进程 `sleep infinity` 是容器基底，不属任务残留；
# tree 清理必须跳过它们，否则会误杀主进程导致容器退出（与第 03 期同一正解）。
MAIN_GUARD = ('pp=$(grep -m1 "^PPid" $d/status 2>/dev/null | awk "{print \\$2}"); '
              'if [ "$pp" = 1 ]; then c=$(tr "\\0" " " < $d/cmdline 2>/dev/null); '
              'case "$c" in "sleep infinity"|"sleep infinity ") continue;; esac; fi; ')
TREE_KILL = ('p=$$; for d in /proc/[0-9]*; do x=${d#/proc/}; '
             '[ "$x" = 1 ] && continue; [ "$x" = "$p" ] && continue; ' + MAIN_GUARD +
             'kill -9 "$x" 2>/dev/null; done; exit 0')


class ApiError(Exception):
    def __init__(self, http_code, code, message):
        super().__init__(message)
        self.http_code, self.code, self.message = http_code, code, message


def sh(argv, timeout=60):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


class _Bounded:
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

    def result(self):
        with self.lock:
            tail = b"".join(self.tail)[-STREAM_KEEP_TAIL:]
            head = bytes(self.head)
        return {"total_bytes": self.total, "kept_bytes": len(head) + len(tail),
                "truncated": self.total > len(head) + len(tail),
                "head_b64": base64.b64encode(head).decode(),
                "tail_b64": base64.b64encode(tail).decode()}


class SystemService:
    def __init__(self, base_dir=None, state_store="journal"):
        chapter_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir or os.path.join(chapter_dir, "data_local", "workspaces")
        os.makedirs(self.base_dir, exist_ok=True)
        self.journal_path = os.path.join(os.path.dirname(self.base_dir), "journal.jsonl")
        self.state_store = state_store
        self.sessions: dict[str, dict] = {}
        self.dedup: dict[str, dict] = {}
        self.lock = threading.Lock()
        self._stop_evt = threading.Event()
        if state_store == "journal":
            self._replay()
        threading.Thread(target=self._ttl_loop, daemon=True).start()

    # ---- journal ----
    def _append(self, kind, **f):
        if self.state_store != "journal":
            return
        with open(self.journal_path, "a") as fp:
            fp.write(json.dumps({"ts": time.time(), "kind": kind, **f}, default=str) + "\n")
            fp.flush()
            os.fsync(fp.fileno())

    def _replay(self):
        if not os.path.exists(self.journal_path):
            return
        with open(self.journal_path) as fp:
            for line in fp:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec["kind"] == "session_created":
                    self.sessions[rec["sid"]] = {
                        "sid": rec["sid"],
                        "workspace": rec["workspace"], "state": "active",
                        "container": None, "ttl_ms": rec["ttl_ms"],
                        "profile": rec.get("profile", "loose"),
                        "output_mode": rec.get("output_mode", "buffer"),
                        "last_used": rec["ts"], "in_flight": 0, "restores": 0}
                elif rec["kind"] == "command_completed":
                    self.dedup[rec["request_id"]] = rec["result"]

    # ---- TTL 释放（含 in_flight 防护与复查，第 11 期语义）----
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
                with self.lock:
                    s = self.sessions.get(sid)
                    if not s or s["state"] != "released" or s["in_flight"] > 0:
                        continue
                self._kill(sid)

    def _kill(self, sid):
        # 含停止态：外部 docker kill 后容器处于 stopped，重建前必须清掉（同名冲突）
        ids = dockerops.ps_ids(f"{dockerops.LABEL_SESSION}={sid}", include_stopped=True)
        dockerops.kill_ids(ids)
        if ids:
            dockerops.docker(["rm", "-f", *ids])

    def _constraint_args(self, profile):
        p = PROFILES[profile]
        return ["--network", "none", "--read-only", "--user", "65534:65534",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--memory", p["memory"], "--memory-swap", p["memory"],
                "--cpus", p["cpus"], "--pids-limit", str(p["pids"]),
                "--tmpfs", "/tmp:rw,size=16m,noexec"]

    def _start(self, s):
        name = f"mis-{CHAPTER}-{s['sid'][2:10]}"
        r = sh(["docker", "run", "-d", "--name", name, "--init",
                f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
                f"--label={dockerops.LABEL_SESSION}={s['sid']}",
                *self._constraint_args(s["profile"]),
                "-v", f"{s['workspace']}:/workspace", "-w", "/workspace",
                images.PINNED["alpine"], "sleep", "infinity"])
        if r.returncode != 0:
            raise ApiError(500, "env_prep_error", r.stderr.strip()[:200])
        s["container"] = name

    def create(self, payload: dict) -> dict:
        sid = "s-" + uuid.uuid4().hex[:12]
        ws = os.path.join(self.base_dir, sid)
        os.makedirs(ws, exist_ok=True)
        os.chmod(ws, 0o777)
        s = {"sid": sid, "workspace": ws, "state": "active",
             "profile": payload.get("profile", "loose"),
             "output_mode": payload.get("output_mode", "buffer"),
             "ttl_ms": int(payload.get("ttl_ms", 10 ** 9)),
             "last_used": time.time(), "in_flight": 0, "restores": 0, "container": None}
        with self.lock:
            self.sessions[sid] = s
        self._append("session_created", sid=sid, workspace=ws, ttl_ms=s["ttl_ms"],
                     profile=s["profile"], output_mode=s["output_mode"])
        self._start(s)
        return {"session_id": sid, "ttl_ms": s["ttl_ms"], "profile": s["profile"],
                "output_mode": s["output_mode"]}

    def _get(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        if s["state"] == "stopped":
            raise ApiError(409, "session_stopped", sid)
        return s

    def exec(self, sid, payload: dict) -> dict:
        s = self._get(sid)
        cmd = payload.get("cmd")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
            raise ApiError(400, "bad_cmd", "cmd must be a non-empty list of strings")
        request_id = payload.get("request_id")
        if request_id and self.state_store == "journal" and request_id in self.dedup:
            r = dict(self.dedup[request_id])
            r["deduplicated"] = True
            return r
        timeout_ms = int(payload.get("timeout_ms", 15000))
        cid = "c-" + uuid.uuid4().hex[:12]
        with self.lock:
            s["last_used"] = time.time()
            s["in_flight"] += 1
        t0 = time.perf_counter()
        restored = False
        try:
            # 容器不在（TTL 释放或被外部杀死）→ 透明重建（第 11/13 期语义）
            if not s.get("container") or not dockerops.ps_ids(
                    f"mis.session={sid}", include_stopped=False):
                self._kill(sid)
                self._start(s)
                restored = True
                s["restores"] += 1
            argv = ["docker", "exec", s["container"], *cmd]
            timed_out = False
            p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if s["output_mode"] == "stream":
                co, ce = _Bounded(), _Bounded()
                ts = [threading.Thread(target=c.drain, args=(pp,), daemon=True)
                      for c, pp in ((co, p.stdout), (ce, p.stderr))]
                for t in ts:
                    t.start()
                try:
                    p.wait(timeout=timeout_ms / 1000.0)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    sh(["docker", "exec", s["container"], "sh", "-c", TREE_KILL])
                    p.wait(timeout=10)
                for t in ts:
                    t.join(timeout=10)
                res = {"command_id": cid, "session_id": sid, "exit_code": p.returncode,
                       "timed_out": timed_out, "category": "timeout" if timed_out else "ok",
                       "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
                       "output_mode": "stream", "stdout": co.result(), "stderr": ce.result(),
                       "restored": restored}
            else:
                try:
                    out, err = p.communicate(timeout=timeout_ms / 1000.0)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    sh(["docker", "exec", s["container"], "sh", "-c", TREE_KILL])
                    out, err = p.communicate(timeout=10)
                res = {"command_id": cid, "session_id": sid, "exit_code": p.returncode,
                       "timed_out": timed_out, "category": "timeout" if timed_out else "ok",
                       "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
                       "output_mode": "buffer",
                       "stdout_b64": base64.b64encode(out or b"").decode(),
                       "stderr_b64": base64.b64encode(err or b"").decode(),
                       "stdout_bytes": len(out or b""), "restored": restored}
            if request_id and self.state_store == "journal":
                self._append("command_completed", request_id=request_id, sid=sid, result=res)
                self.dedup[request_id] = res
            return res
        finally:
            with self.lock:
                s["in_flight"] -= 1
                s["last_used"] = time.time()

    def stop(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        self._kill(sid)
        if os.path.isdir(s["workspace"]):
            shutil.rmtree(s["workspace"], ignore_errors=True)
        with self.lock:
            s["state"] = "stopped"
        self._append("session_stopped", sid=sid)
        return {"stopped": True}

    def status(self, sid):
        s = self.sessions.get(sid)
        if not s:
            raise ApiError(404, "session_not_found", sid)
        return {"session_id": sid, "state": s["state"], "restores": s["restores"],
                "containers_alive": len(dockerops.ps_ids(
                    f"mis.session={sid}", include_stopped=False))}

    def shutdown(self):
        self._stop_evt.set()


def make_handler(service):
    class H(BaseHTTPRequestHandler):
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
                if len(parts) == 3 and parts[2] == "status":
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
                if len(parts) == 3 and parts[2] == "exec":
                    return self._send(200, service.exec(parts[1], self._body()))
                if len(parts) == 3 and parts[2] == "stop":
                    return self._send(200, service.stop(parts[1]))
                self._send(404, {"error": "no_route"})
            except ApiError as e:
                self._send(e.http_code, {"error": e.code, "message": e.message})
            except Exception as e:
                self._send(500, {"error": "internal", "message": str(e)})

        def log_message(self, *a):
            pass

    return H


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8117
    dockerops.cleanup_chapter(CHAPTER)
    images.ensure(images.PINNED["alpine"])
    srv = ThreadingHTTPServer(("127.0.0.1", port),
                              make_handler(SystemService()))
    print(f"chapter_17 system on http://127.0.0.1:{srv.server_address[1]}", flush=True)
    srv.serve_forever()
