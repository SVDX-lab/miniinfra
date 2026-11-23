"""第 07 期假 API：本地测试服务（python:3.12-alpine 容器内运行）。

- /secret/<name>：需要 Bearer SECRET_TOKEN
- /admin/delete：需要 Bearer ADMIN_TOKEN
- 全部为本课程生成的假凭据与假数据（协议第 2 节）。
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SECRET_TOKEN = "TEST_TOKEN_secret_7f3a"
ADMIN_TOKEN = "TEST_TOKEN_admin_0c91"
SECRETS = {"report": "REPORT-DATA-42", "payroll": "PAYROLL-FAKE-99"}


class ApiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def _reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/secret/"):
            name = self.path.split("/")[2]
            if self.headers.get("Authorization") != f"Bearer {SECRET_TOKEN}":
                return self._reply(401, {"error": "unauthorized"})
            if name not in SECRETS:
                return self._reply(404, {"error": "no_such_secret"})
            return self._reply(200, {"secret": SECRETS[name]})
        return self._reply(404, {"error": "no_route"})

    def do_POST(self):
        if self.path == "/admin/delete":
            if self.headers.get("Authorization") != f"Bearer {ADMIN_TOKEN}":
                return self._reply(401, {"error": "unauthorized"})
            return self._reply(200, {"deleted": "fake-resource"})
        return self._reply(404, {"error": "no_route"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), ApiHandler).serve_forever()
