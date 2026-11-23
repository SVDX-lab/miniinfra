"""第 06 期检查与实验：断网 vs 允许列表代理（拓扑自建，含伪造头与 DNS 变化注入）。

服务本体沿用第 05 期语义（per-session、tree 取消、loose 配额），
本期核心机制在本文件的网络拓扑与代理中，故不复制整个 service 文件；
每会话容器按 net_mode 选择 --network none 或 internal 网络 + 代理。
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images, results, stats

CHAPTER = "c06"
BASE = None  # 占位：本期不经 HTTP API 驱动，直接以 docker 语义驱动（机制期）

CHECKS = []
CHAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(CHAPTER_DIR, "data_local")

NET_EGRESS = "mis-c06-net"
NET_INT = "mis-c06-int"

CONSTRAINTS = ["--read-only", "--user", "65534:65534", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--memory", "256m",
               "--memory-swap", "256m", "--cpus", "1", "--pids-limit", "64",
               "--tmpfs", "/tmp:rw,size=16m,noexec"]


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def sh(argv: list[str], timeout=60) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def dex(container: str, cmd: list[str], timeout=30):
    return sh(["docker", "exec", container, *cmd], timeout=timeout)


def setup_topology():
    """建立：egress 网络（可 NAT）+ internal 网络（无外联）+ 两个假源 + dual-homed 代理。"""
    os.makedirs(os.path.join(DATA, "www_allowed"), exist_ok=True)
    os.makedirs(os.path.join(DATA, "www_denied"), exist_ok=True)
    with open(os.path.join(DATA, "www_allowed", "health"), "w") as f:
        f.write("OK-ALLOWED-v1")
    with open(os.path.join(DATA, "www_denied", "health"), "w") as f:
        f.write("OK-DENIED-SOURCE")
    with open(os.path.join(DATA, "policy.json"), "w") as f:
        json.dump({"version": 1, "allow": [{"host": "allowed-origin", "port": 80}]}, f)

    sh(["docker", "network", "create", f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
        "--driver", "bridge", NET_EGRESS])
    sh(["docker", "network", "create", f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
        "--internal", NET_INT])

    for name, www in (("allowed-origin", "www_allowed"), ("denied-origin", "www_denied")):
        sh(["docker", "run", "-d", "--name", name,
            f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
            "--network", NET_EGRESS,
            "-v", f"{os.path.join(DATA, www)}:/www:ro", "-w", "/www",
            images.PINNED["python"], "python", "-m", "http.server", "80"])
    sh(["docker", "run", "-d", "--name", "mis-c06-proxy",
        f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
        "--network", NET_INT,
        "-v", f"{os.path.join(CHAPTER_DIR, 'egress_proxy.py')}:/egress_proxy.py:ro",
        "-v", f"{os.path.join(DATA, 'policy.json')}:/policy.json:ro",
        images.PINNED["python"], "python", "/egress_proxy.py"])
    sh(["docker", "network", "connect", NET_EGRESS, "mis-c06-proxy"])


def new_sandbox(net_mode: str) -> str:
    """创建一个会话容器：deny-all（network none）或 allowlist（internal 网+代理）。"""
    name = f"mis-{CHAPTER}-sb-{net_mode}-{int(time.time()*1000)%100000}"
    args = ["docker", "run", "-d", "--name", name,
            f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}", *CONSTRAINTS]
    if net_mode == "deny-all":
        args += ["--network", "none"]
    else:
        args += ["--network", NET_INT,
                 "-e", "http_proxy=http://mis-c06-proxy:3128", "-e", "no_proxy="]
    args += [images.PINNED["alpine"], "sleep", "infinity"]
    r = sh(args)
    assert r.returncode == 0, r.stderr
    return name


def timed_wget(container: str, url: str, timeout_s=6) -> dict:
    t0 = time.perf_counter()
    r = dex(container, ["sh", "-c", f"wget -T {timeout_s} -qO- '{url}' 2>&1; echo rc=$?"],
            timeout=timeout_s + 20)
    ms = (time.perf_counter() - t0) * 1000
    out = r.stdout.strip()
    rc = int(out.rsplit("rc=", 1)[-1].splitlines()[0]) if "rc=" in out else -1
    body = out.rsplit("rc=", 1)[0].strip()
    return {"rc": rc, "body": body[:200], "ms": round(ms, 1)}


def raw_proxy_request(container: str, url: str) -> str:
    """busybox wget 在 4xx 时不回显响应体，用 nc 原生请求拿完整响应（含 JSON 拒绝体）。"""
    r = dex(container, ["sh", "-c",
                        f"printf 'GET {url} HTTP/1.0\\r\\n\\r\\n' | nc -w 3 mis-c06-proxy 3128"])
    return r.stdout.strip()


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)
    setup_topology()
    time.sleep(2.0)  # 等源与代理就绪

    sb_none = new_sandbox("deny-all")
    sb_alw = new_sandbox("allowlist")

    # R6 断网组：任意请求不可达
    r6 = timed_wget(sb_none, "http://allowed-origin/health", timeout_s=3)
    check("断网组：请求不可达（连接错误）", r6["rc"] != 0, f"rc={r6['rc']}")

    # R1 允许请求
    r1 = timed_wget(sb_alw, "http://allowed-origin/health")
    check("允许列表：allowed-origin 返回 200 + 预期内容",
          r1["rc"] == 0 and r1["body"].endswith("OK-ALLOWED-v1"), f"rc={r1['rc']} body={r1['body']!r}")

    # R2 拒绝请求（带 policy_version）——403 响应体用 nc 原生请求验证
    raw2 = raw_proxy_request(sb_alw, "http://denied-origin/health")
    check("允许列表：denied-origin 被拒（403 JSON 带 policy_version）",
          "403" in raw2 and "policy_version" in raw2, raw2.splitlines()[0])

    # R3 元数据地址
    raw3 = raw_proxy_request(sb_alw, "http://169.254.169.254/latest/meta-data/")
    check("元数据地址被拒", "403" in raw3 and "policy_version" in raw3, raw3.splitlines()[0])

    # R4 伪造 Host 头：URI 目标是 denied-origin，Host 头声称 allowed-origin
    r = dex(sb_alw, ["sh", "-c",
                     "printf 'GET http://denied-origin/health HTTP/1.0\\r\\n"
                     "Host: allowed-origin\\r\\n\\r\\n' | nc -w 3 mis-c06-proxy 3128"])
    forged = r.stdout.strip()
    check("伪造 Host 头不能绕过（按 URI 目标判 403）",
        "403" in forged and "denied-origin" in forged, forged.splitlines()[0] if forged else "empty")

    # R5 目标重建（DNS/IP 变化）：代理按请求重新解析。Docker 的 IP 池可能复用同一地址，
    # 故断言语义（销毁重建后仍放行）并以代理侧解析结果为证据记录实际 IP。
    def origin_ip():
        r = sh(["docker", "inspect", "-f",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", "allowed-origin"])
        return r.stdout.strip()

    def proxy_resolve():
        r = sh(["docker", "exec", "mis-c06-proxy", "python", "-c",
                "import socket; print(socket.gethostbyname('allowed-origin'))"])
        return r.stdout.strip()

    old_ip = origin_ip()
    sh(["docker", "rm", "-f", "allowed-origin"])
    sh(["docker", "run", "-d", "--name", "allowed-origin",
        f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}", "--network", NET_EGRESS,
        "-v", f"{os.path.join(DATA, 'www_allowed')}:/www:ro", "-w", "/www",
        images.PINNED["python"], "python", "-m", "http.server", "80"])
    time.sleep(1.0)
    new_ip = origin_ip()
    resolved = proxy_resolve()
    r5 = timed_wget(sb_alw, "http://allowed-origin/health")
    check("目标销毁重建后按域名仍正确放行（代理请求时解析）",
          r5["rc"] == 0 and r5["body"].endswith("OK-ALLOWED-v1") and resolved == new_ip,
          f"ip {old_ip}→{new_ip}（Docker 可能复用），代理解析={resolved}")

    # 延迟对照：经代理 vs 直连（直连从 egress 网络内的 denied-origin 容器发起，消除沙箱侧差异）
    proxy_lat = [timed_wget(sb_alw, "http://allowed-origin/health")["ms"] for _ in range(5)]
    direct_lat = []
    for _ in range(5):
        t0 = time.perf_counter()
        dex("denied-origin", ["sh", "-c", "wget -T 4 -qO- http://allowed-origin/health >/dev/null"], 10)
        direct_lat.append((time.perf_counter() - t0) * 1000)

    # 清理
    for c in (sb_none, sb_alw):
        sh(["docker", "rm", "-f", c])
    dockerops.cleanup_chapter(CHAPTER)
    left = dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}")
    check("实验后容器归零", len(left) == 0)

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "latency_ms": {"via_proxy": stats.summary(proxy_lat),
                          "direct_egress_net": stats.summary(direct_lat),
                          "raw": {"proxy": proxy_lat, "direct": [round(x, 1) for x in direct_lat]}}}
    path = results.write_result(
        CHAPTER_DIR, "egress_ab", doc,
        extra_meta={"images": [images.PINNED["alpine"], images.PINNED["python"]],
                    "policy_version": 1, "cache_state": "warm"})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
