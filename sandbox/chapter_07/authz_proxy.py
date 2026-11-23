"""第 07 期授权代理：运行在边界外容器（dual-homed），持有凭据并执行动作允许清单。

- 沙箱请求形如 GET http://authz/secret/report —— 代理校验 (resource, verb) 白名单；
- 允许：GET /secret/report；默认拒绝其余一切（含 /secret/payroll、POST /admin/delete）；
- 转发时代理注入真实 Token（沙箱永远拿不到）；
- 每个请求写一行 JSONL 审计（会话、动作、判定、UTC 时间）。
"""
import datetime
import json
import socket
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = ("mis-c07-api", 8080)
AUDIT_PATH = "/audit/audit.jsonl"
ALLOW = {("GET", "/secret/report")}


def audit(session, method, path, decision, detail=""):
    entry = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds"),
             "session": session, "method": method, "path": path,
             "decision": decision, "detail": detail}
    with open(AUDIT_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


class AuthzHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def _relay(self, method):
        u = urllib.parse.urlparse(self.path)
        path = u.path
        session = self.headers.get("X-Mis-Session", "unknown")
        if (method, path) not in ALLOW:
            audit(session, method, path, "denied", "not_in_allowlist")
            body = json.dumps({"error": "forbidden", "reason": "not_in_allowlist"}).encode()
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        # 代理持有凭据并注入（沙箱不可见）
        token = "TEST_TOKEN_secret_7f3a"
        try:
            up = socket.create_connection(UPSTREAM, timeout=5)
        except OSError:
            audit(session, method, path, "error", "upstream_unreachable")
            return
        req = (f"{method} {path} HTTP/1.0\r\nHost: {UPSTREAM[0]}\r\n"
               f"Authorization: Bearer {token}\r\n\r\n").encode()
        up.sendall(req)
        while True:
            chunk = up.recv(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        up.close()
        audit(session, method, path, "allowed")

    def do_GET(self):
        self._relay("GET")

    def do_POST(self):
        self._relay("POST")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    import os
    os.makedirs("/audit", exist_ok=True)
    ThreadingHTTPServer(("0.0.0.0", 3129), AuthzHandler).serve_forever()
