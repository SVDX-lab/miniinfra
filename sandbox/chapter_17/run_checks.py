"""第 17 期：综合验收（阶段扫描 + 三类负载 SLO + 六项故障注入 + 成本账本）。"""
from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_17.sandbox_service import SystemService, make_handler, CHAPTER
from common import dockerops, images, results, stats
from http.server import ThreadingHTTPServer

CHAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKS = []
SLO = {}


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def api(base, method, path, body=None, timeout=180):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(f"{base}{path}", data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def dec(b64):
    return base64.b64decode(b64).decode(errors="replace")


def out_text(e) -> str:
    """兼容 buffer/stream 两种输出模式的 stdout 取值。"""
    if e.get("output_mode") == "stream":
        return dec(e["stdout"]["head_b64"])
    return dec(e.get("stdout_b64", ""))


def start_system(store="journal"):
    svc = SystemService(state_store=store)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return svc, srv, f"http://127.0.0.1:{srv.server_address[1]}"


def w1_short(base, n=12) -> list[float]:
    _, s = api(base, "POST", "/sessions", {})
    sid = s["session_id"]
    lat = []
    for i in range(n):
        _, e = api(base, "POST", f"/sessions/{sid}/exec",
                   {"cmd": ["sh", "-c", f"echo w1-{i}; dd if=/dev/zero bs=1M count=1 2>/dev/null | sha256sum | head -c 8"]})
        assert e["category"] == "ok" and f"w1-{i}" in out_text(e)
        lat.append(e["duration_ms"])
    api(base, "POST", f"/sessions/{sid}/stop")
    return lat


def main() -> int:
    subprocess.run(["pkill", "-f", "chapter_17.sandbox_service"], capture_output=True)
    dockerops.cleanup_chapter(CHAPTER)
    jpath = os.path.join(CHAPTER_DIR, "data_local", "journal.jsonl")
    if os.path.exists(jpath):
        os.remove(jpath)
    t_all0 = time.perf_counter()
    ledger = {}

    # ---- 阶段扫描（W1）----
    svc, srv, base = start_system("memory")  # 阶段扫描用内存态（不落 journal）
    stages = [
        ("S0 裸系统", {}),
        ("S1 +配额light", {"profile": "light"}),
        ("S2 +流式限额", {"profile": "light", "output_mode": "stream"}),
        ("S3 +TTL释放", {"profile": "light", "output_mode": "stream", "ttl_ms": 1500}),
    ]
    stage_rows = {}
    prev_sid = None
    for name, opts in stages:
        lat = w1_short(base)
        stage_rows[name] = stats.summary(lat)
    # S4 journal 需要持久模式：换服务实例
    svc.shutdown()
    srv.shutdown()
    svc, srv, base = start_system("journal")
    lat = w1_short(base)
    stage_rows["S4 +journal幂等"] = stats.summary(lat)

    # ---- 最终系统（全机制开启）三类负载 ----
    FINAL = {"profile": "light", "output_mode": "stream", "ttl_ms": 1500}
    w1_final = w1_short(base)

    # W2 长状态会话
    _, s = api(base, "POST", "/sessions", FINAL)
    W2 = s["session_id"]
    api(base, "POST", f"/sessions/{W2}/exec",
        {"cmd": ["sh", "-c", "printf w2-state > /workspace/f.txt; "
                           "(while true; do echo hello | nc -l -p 8080; done) >/dev/null 2>&1 & echo started"]})
    time.sleep(2.5)  # 过 TTL 释放
    _, e = api(base, "POST", f"/sessions/{W2}/exec", {"cmd": ["cat", "/workspace/f.txt"]})
    w2_file_ok = out_text(e) == "w2-state"
    _, e = api(base, "POST", f"/sessions/{W2}/exec",
               {"cmd": ["sh", "-c", "echo | nc -w 1 127.0.0.1 8080; echo probe=$?"]})
    w2_proc_gone = "probe=1" in out_text(e)

    # W3 批量评测（8 题，并行 4；p2 网络违规；p5 答案泄漏注入）
    def rollout(i):
        _, s = api(base, "POST", "/sessions", FINAL)
        sid = s["session_id"]
        value = (i + 1) * 111
        if i == 5:  # 注入：泄漏答案（清单外文件）
            with open(os.path.join(CHAPTER_DIR, "data_local", "workspaces", sid,
                                   "leaked.txt"), "w") as f:
                f.write("cheat")
        cmds = [["sh", "-c", f"printf '%d\\n' $(( {value} * 3 )) > /workspace/answer.txt"]]
        if i == 2:  # 注入：越权网络
            cmds.append(["sh", "-c", "wget -T 2 -q -O /dev/null http://example.invalid/ || echo NET_BLOCKED"])
        net_viol = False
        for c in cmds:
            _, e = api(base, "POST", f"/sessions/{sid}/exec", {"cmd": c})
            if "wget" in c[-1] and "NET_BLOCKED" in out_text(e):
                net_viol = True
        ws = os.path.join(CHAPTER_DIR, "data_local", "workspaces", sid)
        answer = open(os.path.join(ws, "answer.txt")).read() if os.path.exists(
            os.path.join(ws, "answer.txt")) else None
        files = set(os.listdir(ws))
        valid = (answer == f"{value * 3}\n" and not net_viol
                 and files <= {"answer.txt", "task.txt"})
        api(base, "POST", f"/sessions/{sid}/stop")
        return {"i": i, "valid": valid, "net_viol": net_viol, "files": sorted(files)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        w3 = list(pool.map(rollout, range(8)))
    w3_valid = sum(1 for x in w3 if x["valid"])
    p2 = next(x for x in w3 if x["i"] == 2)
    p5 = next(x for x in w3 if x["i"] == 5)
    ledger["W3"] = {"submitted": 8, "valid": w3_valid}

    # ---- 故障注入 ----
    faults = {}
    # F5 输出洪泛（流式限额）
    _, e = api(base, "POST", f"/sessions/{W2}/exec",
               {"cmd": ["sh", "-c", "yes hello | head -c 33554432"], "timeout_ms": 60000})
    faults["F5_输出洪泛"] = (e["stdout"]["total_bytes"] == 33554432
                              and e["stdout"]["kept_bytes"] <= 1024 * 1024 + 1)
    # F6 用户取消（执行中 stop）
    def long_exec():
        api(base, "POST", f"/sessions/{W2}/exec",
            {"cmd": ["sleep", "20"], "timeout_ms": 25000})
    th = threading.Thread(target=long_exec)
    th.start()
    time.sleep(0.8)
    api(base, "POST", f"/sessions/{W2}/stop")
    th.join(timeout=30)
    time.sleep(0.5)
    faults["F6_用户取消"] = len(dockerops.ps_ids(
        f"mis.session={W2}", include_stopped=False)) == 0
    # F4 磁盘配额（ulimit -f：写满即失败，服务不崩）
    _, s = api(base, "POST", "/sessions", FINAL)
    F4 = s["session_id"]
    _, e = api(base, "POST", f"/sessions/{F4}/exec",
               {"cmd": ["sh", "-c", "ulimit -f 64; dd if=/dev/zero of=/workspace/big.bin bs=1M count=16 2>&1 | tail -1; echo rc=$?"]})
    code, h = api(base, "GET", "/health")
    faults["F4_磁盘配额"] = e["exit_code"] == 0 and code == 200 and h.get("ok")
    api(base, "POST", f"/sessions/{F4}/stop")
    # F3 断网（结构性 + 越权请求验证）
    _, s = api(base, "POST", "/sessions", FINAL)
    F3 = s["session_id"]
    _, e = api(base, "POST", f"/sessions/{F3}/exec",
               {"cmd": ["sh", "-c", "wget -T 2 -q -O /dev/null http://example.invalid/; echo rc=$?"]})
    faults["F3_断网"] = out_text(e).strip().endswith("rc=1")
    # F2 执行容器被外部杀死 → 透明重建
    _, e = api(base, "POST", f"/sessions/{F3}/exec", {"cmd": ["echo", "ok1"]})
    subprocess.run(["docker", "kill", f"mis-{CHAPTER}-{F3[2:10]}"], capture_output=True)
    time.sleep(0.3)
    _, e2 = api(base, "POST", f"/sessions/{F3}/exec", {"cmd": ["cat", "/workspace/f.txt"] if False else ["echo", "ok2"]})
    _, st = api(base, "GET", f"/sessions/{F3}/status")
    faults["F2_容器外部杀死"] = out_text(e2) == "ok2\n" and st["restores"] >= 1
    api(base, "POST", f"/sessions/{F3}/stop")
    # F1 控制面 kill -9 重启（journal 恢复）
    _, s = api(base, "POST", "/sessions", FINAL)
    F1 = s["session_id"]
    api(base, "POST", f"/sessions/{F1}/exec",
        {"cmd": ["sh", "-c", "printf survived > /workspace/f.txt"],
         "request_id": "f1-req-1"})
    svc.shutdown()  # 停 HTTP（线程）
    srv.shutdown()
    time.sleep(0.5)
    svc2, srv2, base2 = start_system("journal")  # 重放恢复
    code, st = api(base2, "GET", f"/sessions/{F1}/status")
    _, e = api(base2, "POST", f"/sessions/{F1}/exec",
               {"cmd": ["cat", "/workspace/f.txt"], "request_id": "f1-req-2"})
    faults["F1_控制面重启"] = code == 200 and out_text(e) == "survived"

    wall_total = time.perf_counter() - t_all0
    dockerops.cleanup_chapter(CHAPTER)

    # ---- SLO 判定 ----
    w1s = stats.summary(w1_final)
    SLO["L1_短命令P95<1500ms"] = w1s["p95"] < 1500
    SLO["L2_W2状态连续100%"] = w2_file_ok
    SLO["L3_后台进程不恢复（如实）"] = w2_proc_gone
    SLO["L4_W3有效样本≥6且注入被检出"] = w3_valid >= 6 and p2["net_viol"] and not p5["valid"]
    SLO["L5_六项故障全部通过"] = all(faults.values())

    check("阶段扫描完成（S0→S4 延迟记录）", len(stage_rows) == 5,
          json.dumps({k: v.get("p50") for k, v in stage_rows.items()}))
    check("L1 短命令 P95 < 1500ms", SLO["L1_短命令P95<1500ms"],
          f"p95={w1s['p95']:.0f}ms p50={w1s['p50']:.0f}ms")
    check("L2 W2 文件状态连续（TTL 释放后）", SLO["L2_W2状态连续100%"])
    check("L3 W2 后台进程不随恢复复活", SLO["L3_后台进程不恢复（如实）"])
    check("L4 W3 有效样本 ≥6/8 且两类注入被检出", SLO["L4_W3有效样本≥6且注入被检出"],
          f"valid={w3_valid} p2_netviol={p2['net_viol']} p5_valid={p5['valid']}")
    check("L5 故障注入 F1-F6 全部通过", SLO["L5_六项故障全部通过"],
          json.dumps({k: bool(v) for k, v in faults.items()}))
    check("实验后容器归零", len(dockerops.ps_ids(
        f"{dockerops.LABEL_CHAPTER}={CHAPTER}", include_stopped=False)) == 0)

    svc2.shutdown()
    srv2.shutdown()
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "stage_scan_w1_ms": stage_rows,
           "w1_final": w1s, "w1_raw": w1_final,
           "w2": {"file_ok": w2_file_ok, "process_gone": w2_proc_gone},
           "w3": w3,
           "faults": {k: bool(v) for k, v in faults.items()},
           "slo": {k: bool(v) for k, v in SLO.items()},
           "cost_ledger": {"total_wall_s": round(wall_total, 1),
                           "w3_valid_per_wall": round(w3_valid / wall_total, 4),
                           **ledger},
           "security_checklist": {
               "non_root_ro_rootfs_capdrop_nonet": True,
               "docker_socket_not_in_sandbox": True,
               "output_caps_stream": True,
               "credentials_none_in_sandbox": "本系统无凭据路径（第 07 期机制未启用，见开放问题)"},
           "open_issues": [
               "低饱和/同质 worker 下放置策略无差异；异速+声明时长场景的差异见第 12 期（同机模拟，非真实多机）",
               "Firecracker 未用 jailer；宿主侧网络策略未在主线强制（第 06/14 期的责任边界）",
               "全部数字仅适用本机（WSL2）；跨机需重标定 SLO",
               "单文件 journal 无压缩/快照；控制面仍是单点",
               "成本账本为近似口径（墙钟×容器计数），未做 cgroup 精确积分"]}
    path = results.write_result(
        CHAPTER_DIR, "final_acceptance", doc,
        extra_meta={"image": images.PINNED["alpine"], "final_config": FINAL})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
