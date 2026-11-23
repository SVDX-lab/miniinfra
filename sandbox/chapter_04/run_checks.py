"""第 04 期检查：内存超限 OOM、进程自身失败对照、PID 压力、CPU 扫描、邻居干扰。"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_04.sandbox_service import SandboxService, make_handler, CHAPTER, oom_kill_count
from common import dockerops, images, results, stats
from http.server import ThreadingHTTPServer

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def api(method, url, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def dec(b64):
    return base64.b64decode(b64).decode(errors="replace")


PY_ALLOC_200M = ["python", "-c",
                 "d = bytearray(200*1024*1024)\n"
                 "for i in range(0, len(d), 4096): d[i] = 1\n"
                 "print('allocated')"]
# 自身失败对照：ulimit -v 128MiB 下分配 200MiB → MemoryError → exit 3
PY_ALLOC_SELFFAIL = ["sh", "-c",
                     "ulimit -v 131072; "
                     "python -c \"import sys\n"
                     "try:\n"
                     "    d = bytearray(200*1024*1024)\n"
                     "    for i in range(0, len(d), 4096): d[i] = 1\n"
                     "except MemoryError:\n"
                     "    print('self-fail MemoryError'); sys.exit(3)\""]
# ash 非交互模式在 fork 失败时会直接退出脚本，用外层 shell 捕获内层退出码
PID_STRESS = ["sh", "-c",
              "sh -c 'for i in $(seq 1 50); do sleep 10 & done' 2>/tmp/forkerr; "
              "echo spawner_rc=$?; echo fork_err_lines=$(grep -c . /tmp/forkerr)"]
CPU_WORK = ["sh", "-c", "dd if=/dev/zero bs=1M count=1024 2>/dev/null | sha256sum"]


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)
    svc = SandboxService()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    # 1) 内存超限（light=64m，分配 200MiB）→ OOM kill
    _, s = api("POST", f"{base}/sessions", {"quota_profile": "light", "image": images.PINNED["python"]})
    sid = s["session_id"]
    _, e = api("POST", f"{base}/sessions/{sid}/exec", {"cmd": PY_ALLOC_200M, "timeout_ms": 30000})
    check("mem 超限：退出码 137（SIGKILL/OOM）", e["exit_code"] == 137,
          f"exit={e['exit_code']}")
    check("mem 超限：cgroup oom_kill 计数增加",
          (e["oom_kill_delta"] or 0) > 0, f"delta={e['oom_kill_delta']}")
    check("mem 超限后容器仍存活（会话可继续）",
          len(dockerops.ps_ids(f"mis.session={sid}", include_stopped=False)) == 1)
    api("POST", f"{base}/sessions/{sid}/stop")

    # 2) 进程自身失败对照（同量分配，ulimit 触发 MemoryError）
    _, s = api("POST", f"{base}/sessions", {"quota_profile": "mem_heavy", "image": images.PINNED["python"]})
    sid = s["session_id"]
    _, e = api("POST", f"{base}/sessions/{sid}/exec", {"cmd": PY_ALLOC_SELFFAIL, "timeout_ms": 30000})
    check("自身失败：退出码 3 且无 oom_kill",
          e["exit_code"] == 3 and (e["oom_kill_delta"] or 0) == 0,
          f"exit={e['exit_code']} oom_delta={e['oom_kill_delta']} out={dec(e['stdout_b64']).strip()!r}")
    api("POST", f"{base}/sessions/{sid}/stop")

    # 3) PID 压力：light(32) 内层派生器死亡；loose(64) 全部成功
    _, s = api("POST", f"{base}/sessions", {"quota_profile": "light"})
    sid = s["session_id"]
    _, e = api("POST", f"{base}/sessions/{sid}/exec", {"cmd": PID_STRESS, "timeout_ms": 30000})
    fail_light = dec(e["stdout_b64"]).strip()
    api("POST", f"{base}/sessions/{sid}/stop")
    _, s = api("POST", f"{base}/sessions", {"quota_profile": "loose"})
    sid = s["session_id"]
    _, e = api("POST", f"{base}/sessions/{sid}/exec", {"cmd": PID_STRESS, "timeout_ms": 30000})
    fail_loose = dec(e["stdout_b64"]).strip()
    api("POST", f"{base}/sessions/{sid}/stop")
    check("PID 压力：light(32) 派生器死亡且 fork 报错",
          "spawner_rc=2" in fail_light and "fork_err_lines=1" in fail_light,
          fail_light.replace("\n", " | "))
    check("PID 压力：loose(64) 50 个并发 fork 全部成功",
          "spawner_rc=0" in fail_loose and "fork_err_lines=0" in fail_loose, fail_loose)

    # 4) light 配额下正常命令可执行
    _, s = api("POST", f"{base}/sessions", {"quota_profile": "light"})
    sid = s["session_id"]
    _, e = api("POST", f"{base}/sessions/{sid}/exec", {"cmd": ["echo", "ok"]})
    check("light 配额下正常命令可执行", e["exit_code"] == 0)
    api("POST", f"{base}/sessions/{sid}/stop")

    # 5) CPU 扫描：同一 1GiB 工作，cpus ∈ {0.25, 0.5, 1, 2}
    cpu_scan = {}
    for prof, cpu in (("light", "0.25"), ("half", "0.5"), ("loose", "1.0"), ("cpu_heavy", "2.0")):
        _, s = api("POST", f"{base}/sessions", {"quota_profile": prof})
        sid = s["session_id"]
        times = []
        for _ in range(3):
            _, e = api("POST", f"{base}/sessions/{sid}/exec",
                       {"cmd": CPU_WORK, "timeout_ms": 120000})
            times.append(e["duration_ms"])
        cpu_scan[cpu] = stats.summary(times)
        api("POST", f"{base}/sessions/{sid}/stop")
    t025 = cpu_scan["0.25"]["p50"]
    t2 = cpu_scan["2.0"]["p50"]
    ratio = t025 / t2
    check("CPU 扫描：0.25 配额耗时约为 2.0 的 4~9 倍", 4 <= ratio <= 9,
          f"0.25cpu={t025:.0f}ms 2cpu={t2:.0f}ms ratio={ratio:.1f}")

    # 6) 邻居干扰：受害者单独 vs 配额共享 vs cpuset 硬分区
    # 本机拓扑：6 物理核 ×2 SMT = 12 逻辑核。ALU 密集负载在 SMT 兄弟线程间
    # 存在配额无法消除的执行单元竞争——正解是 cpuset 硬分区（受害者独占核）。
    # 燃烧器必须并发提交（exec 阻塞），否则受害者测量窗口内负载已消失。
    from concurrent.futures import ThreadPoolExecutor

    def victim_time(quota_profile, cpuset=None):
        _, s = api("POST", f"{base}/sessions",
                   {"quota_profile": quota_profile, "cpuset": cpuset})
        vid = s["session_id"]
        ts = []
        for _ in range(3):
            _, e = api("POST", f"{base}/sessions/{vid}/exec",
                       {"cmd": ["sh", "-c", "dd if=/dev/zero bs=1M count=512 2>/dev/null | sha256sum"],
                        "timeout_ms": 120000})
            ts.append(e["duration_ms"])
        api("POST", f"{base}/sessions/{vid}/stop")
        return stats.summary(ts)["p50"]

    def interference(burner_profile, victim_cpuset=None, burner_cpuset=None, n=16):
        burners = []
        pool = ThreadPoolExecutor(max_workers=n + 2)
        futs = []
        for _ in range(n):
            _, s = api("POST", f"{base}/sessions",
                       {"quota_profile": burner_profile, "cpuset": burner_cpuset,
                        "ttl_ms": 120000})
            b = s["session_id"]
            burners.append(b)
            futs.append(pool.submit(
                api, "POST", f"{base}/sessions/{b}/exec",
                {"cmd": ["sha256sum", "/dev/zero"], "timeout_ms": 30000}))
        time.sleep(4.0)  # 等燃烧器全部进入燃烧状态
        t = victim_time("loose", cpuset=victim_cpuset)
        for f in futs:
            f.result(timeout=60)
        pool.shutdown()
        for b in burners:
            api("POST", f"{base}/sessions/{b}/stop")
        return t

    alone = victim_time("loose", cpuset="0,1")  # 受害者独占 0-1 两核
    t_loose_burn = interference("loose")        # 配额共享全部核
    t_half_burn = interference("half")          # 半额配额共享
    # cpuset 硬分区的成立条件：可运行线程总数 ≤ 物理核预算（本机 6 物理核）。
    # 受害者 2 线程钉在两个不同物理核（vCPU 0,2）上；4 燃烧器钉在 4-11 →
    # 共 6 个可运行线程 ≤ 6 物理核，宿主无需 SMT 配对，分区真正成立。
    # 超预算（8/16 燃烧器）时，WSL2 的 vCPU 调度在宿主层竞争，客户机分区失效。
    t_pin_fit = interference("half", victim_cpuset="0,2",
                             burner_cpuset="4-11", n=4)
    t_pin_over = interference("half", victim_cpuset="0,1",
                              burner_cpuset="2-11", n=16)
    check("干扰：共享物理核时配额无法消除 SMT 竞争（>2×单独）",
          t_loose_burn > alone * 2.0,
          f"alone={alone:.0f}ms loose_burners={t_loose_burn:.0f}ms")
    check("配额缓解：半额燃烧器降低干扰但未消除", t_half_burn < t_loose_burn * 0.7,
          f"half_burners={t_half_burn:.0f}ms ({t_half_burn/alone:.2f}×alone)")
    check("正解：cpuset 分区 + 可运行线程≤物理核预算 → 干扰基本消除（<1.2×）",
          t_pin_fit < alone * 1.2,
          f"pin_fit={t_pin_fit:.0f}ms ({t_pin_fit/alone:.2f}×alone)；"
          f"受害者独占物理核（vCPU0,2），4 燃烧器限核4-11，共6线程≤6物理核")
    check("边界：超物理预算时 hypervisor 调度绕过客户机分区（证据记录）",
          t_pin_over > alone * 1.3,
          f"pin_over={t_pin_over:.0f}ms ({t_pin_over/alone:.2f}×alone)；"
          f"16 燃烧器需求≈9核>6物理核，WSL2 vCPU 调度在宿主层竞争")

    left = dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}", include_stopped=False)
    check("实验后容器归零", len(left) == 0, f"left={len(left)}")

    svc.shutdown()
    srv.shutdown()
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "cpu_scan_ms": cpu_scan,
           "interference_ms": {"alone_pinned_cores01": alone, "share_loose": t_loose_burn,
                               "share_half": t_half_burn,
                               "cpuset_partition_within_budget": t_pin_fit,
                               "cpuset_partition_over_budget": t_pin_over}}
    path = results.write_result(
        os.path.dirname(os.path.abspath(__file__)), "quota_scan", doc,
        extra_meta={"images": [images.PINNED["alpine"], images.PINNED["python"]],
                    "cache_state": "warm", "repeats": 3})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
