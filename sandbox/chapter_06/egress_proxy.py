"""第 06 期出站转发代理：运行在边界外容器中（python:3.12-alpine）。

要点：
- 允许列表匹配的是请求 URI 的目标 (host, port)，不是 Host 头（防字符串匹配绕过）；
- 域名由代理自己解析再连接（沙箱无法影响解析）；
- 显式拒绝链路本地/环回段（元数据服务防线的演示子集）；
- 拒绝响应携带 policy_version，策略变化可审计。
"""
import json
import socket
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

POLICY_PATH = "/policy.json"


def load_policy() -> dict:
    with open(POLICY_PATH) as f:
        return json.load(f)


def is_denied_range(ip: str) -> bool:
    # 生产应拒绝 RFC1918/元数据等全部内网段；课程假源位于 docker 内网段，
    # 此处仅演示链路本地（169.254.0.0/16，云元数据）与环回的显式拒绝。
    return ip.startswith("169.254.") or ip.startswith("127.")


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def _deny(self, host: str, version: int):
        body = json.dumps({"denied": host, "policy_version": version}).encode()
        self.send_response(403)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        pol = load_policy()
        version = pol.get("version", 0)
        u = urllib.parse.urlparse(self.path)
        if not u.scheme.startswith("http"):
            return self._deny(str(u.path), version)
        host, port = u.hostname, u.port or 80
        allowed = any(e["host"] == host and e.get("port", 80) == port
                      for e in pol.get("allow", []))
        if not allowed:
            return self._deny(f"{host}:{port}", version)
        try:
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
            ip = infos[0][4][0]
        except socket.gaierror:
            return self._deny(f"dns:{host}", version)
        if is_denied_range(ip):
            return self._deny(f"range:{ip}", version)
        try:
            upstream = socket.create_connection((ip, port), timeout=5)
        except OSError:
            return self._deny(f"connect:{host}", version)
        req = (f"GET {u.path or '/'} HTTP/1.0\r\nHost: {host}\r\n"
               f"X-Mis-Proxied: true\r\n\r\n").encode()
        upstream.sendall(req)
        while True:
            chunk = upstream.recv(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        upstream.close()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", 3128), ProxyHandler)
    srv.serve_forever()
