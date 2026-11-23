"""第 07 期检查与实验：注入凭据 vs 边界外授权代理（拓扑自建）。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images, results, stats

CHAPTER = "c07"
CHAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(CHAPTER_DIR, "data_local")
NET_EGRESS = "mis-c07-net"
NET_INT = "mis-c07-int"

CONSTRAINTS = ["--read-only", "--user", "65534:65534", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--memory", "256m",
               "--memory-swap", "256m", "--cpus", "1", "--pids-limit", "64",
               "--tmpfs", "/tmp:rw,size=16m,noexec"]

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def sh(argv, timeout=60):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def dex(container, cmd, timeout=30):
    return sh(["docker", "exec", container, *cmd], timeout=timeout)


def setup_topology():
    os.makedirs(DATA, exist_ok=True)
    sh(["docker", "network", "create", f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
        "--driver", "bridge", NET_EGRESS])
    sh(["docker", "network", "create", f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
        "--internal", NET_INT])
    sh(["docker", "run", "-d", "--name", "mis-c07-api",
        f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}", "--network", NET_EGRESS,
        "-v", f"{os.path.join(CHAPTER_DIR, 'fake_api.py')}:/fake_api.py:ro",
        images.PINNED["python"], "python", "/fake_api.py"])
    sh(["docker", "run", "-d", "--name", "mis-c07-authz",
        f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}", "--network", NET_INT,
        "-v", f"{os.path.join(CHAPTER_DIR, 'authz_proxy.py')}:/authz_proxy.py:ro",
        "-v", f"{DATA}:/audit",
        images.PINNED["python"], "python", "/authz_proxy.py"])
    sh(["docker", "network", "connect", NET_EGRESS, "mis-c07-authz"])


def new_sandbox(mode: str) -> str:
    name = f"mis-{CHAPTER}-sb-{mode}-{int(time.time()*1000)%100000}"
    args = ["docker", "run", "-d", "--name", name,
            f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}", *CONSTRAINTS]
    if mode == "injected":
        # 基线组：静态凭据直接注入，且可直连假 API（演示暴露面）
        args += ["--network", NET_EGRESS, "-e", "SECRET_TOKEN=TEST_TOKEN_secret_7f3a"]
    else:
        # 实验组：无凭据，只见代理
        args += ["--network", NET_INT]
    args += [images.PINNED["alpine"], "sleep", "infinity"]
    r = sh(args)
    assert r.returncode == 0, r.stderr
    return name


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)
    setup_topology()
    time.sleep(2.0)

    sb_inj = new_sandbox("injected")
    sb_prx = new_sandbox("proxy")

    # C1：A 组 env 泄露（注入即暴露的演示）
    r = dex(sb_inj, ["sh", "-c", "env | grep -c TOKEN; env | grep TOKEN | cut -c1-30"])
    check("A组(注入)：Agent 可直接读出 Token（暴露演示）",
          "TEST_TOKEN" in r.stdout, r.stdout.strip().splitlines()[-1] if r.stdout else "")

    # C2：B 组无凭据痕迹
    r = dex(sb_prx, ["sh", "-c", "env | grep -ci token; true"])
    env_hits = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "0"
    r = dex(sb_prx, ["sh", "-c",
                     "grep -rl TEST_TOKEN /workspace /tmp /etc 2>/dev/null | wc -l"],
            timeout=60)
    fs_hits = r.stdout.strip()
    check("B组(代理)：环境与文件系统均无长期 Token", env_hits == "0" and fs_hits == "0",
          f"env_hits={env_hits} fs_hits={fs_hits}")

    # C3/C4/C5：经代理的允许与越权操作
    r3 = dex(sb_prx, ["sh", "-c",
                      "wget -T 5 -qO- --header 'X-Mis-Session: sb_prx' "
                      "http://mis-c07-authz:3129/secret/report; echo rc=$?"])
    ok3 = "REPORT-DATA-42" in r3.stdout and "rc=0" in r3.stdout
    check("B组：允许操作 GET /secret/report 成功", ok3, r3.stdout.strip()[:80])

    # busybox wget 不回显 4xx 响应体，用 nc 原生请求验证拒绝体
    r4 = dex(sb_prx, ["sh", "-c",
                      "printf 'GET http://mis-c07-authz:3129/secret/payroll HTTP/1.0\\r\\n"
                      "X-Mis-Session: sb_prx\\r\\n\\r\\n' | nc -w 3 mis-c07-authz 3129"])
    check("B组：越权资源 GET /secret/payroll 被拒",
          "403" in r4.stdout and "not_in_allowlist" in r4.stdout,
          r4.stdout.strip().splitlines()[0] if r4.stdout.strip() else "")

    r5 = dex(sb_prx, ["sh", "-c",
                      "printf 'POST http://mis-c07-authz:3129/admin/delete HTTP/1.0\\r\\n"
                      "X-Mis-Session: sb_prx\\r\\nContent-Length: 0\\r\\n\\r\\n' "
                      "| nc -w 3 mis-c07-authz 3129"])
    check("B组：越权动作 POST /admin/delete 被拒",
          "403" in r5.stdout and "not_in_allowlist" in r5.stdout,
          r5.stdout.strip().splitlines()[0] if r5.stdout.strip() else "")

    # 审计日志
    r = sh(["docker", "exec", "mis-c07-authz", "cat", "/audit/audit.jsonl"])
    entries = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
    decisions = [(e["path"], e["decision"]) for e in entries]
    check("审计日志完整（allowed + 2×denied，含会话与UTC时间）",
          ("/secret/report", "allowed") in decisions
          and ("/secret/payroll", "denied") in decisions
          and ("/admin/delete", "denied") in decisions
          and all("utc" in e and "session" in e for e in entries),
          str(decisions))

    # A 组对照延迟：直连（带注入 Token）
    lat_direct = []
    for _ in range(5):
        t0 = time.perf_counter()
        dex(sb_inj, ["sh", "-c",
                     "wget -T 5 -qO- --header 'Authorization: Bearer TEST_TOKEN_secret_7f3a' "
                     "http://mis-c07-api:8080/secret/report >/dev/null"])
        lat_direct.append((time.perf_counter() - t0) * 1000)
    lat_proxy = []
    for _ in range(5):
        t0 = time.perf_counter()
        dex(sb_prx, ["sh", "-c",
                     "wget -T 5 -qO- --header 'X-Mis-Session: sb_prx' "
                     "http://mis-c07-authz:3129/secret/report >/dev/null"])
        lat_proxy.append((time.perf_counter() - t0) * 1000)

    for c in (sb_inj, sb_prx):
        sh(["docker", "rm", "-f", c])
    dockerops.cleanup_chapter(CHAPTER)
    check("实验后容器归零", len(dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}")) == 0)

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "latency_ms": {"direct_with_token": stats.summary(lat_direct),
                          "via_authz_proxy": stats.summary(lat_proxy)},
           "audit_entries": entries}
    path = results.write_result(
        CHAPTER_DIR, "credential_ab", doc,
        extra_meta={"images": [images.PINNED["alpine"], images.PINNED["python"]],
                    "cache_state": "warm", "credentials": "course-local-fake"})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
