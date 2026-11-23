"""第 15 期实验：训练 rollout 与评测（单个会话 vs 固定配置批量，含作弊注入检测）。

题目/轨迹/奖励器全部确定性；答案与评测器只在控制面（宿主），
Agent 工作区只拿到题目文件。轨迹哈希 = 各步 (exit, stdout_sha) 的链。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images, results, stats

CHAPTER = "c15"
CHAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

CONSTRAINTS = ["--network", "none", "--read-only", "--user", "65534:65534",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--memory", "128m", "--cpus", "0.5", "--pids-limit", "32",
               "--tmpfs", "/tmp:rw,size=8m,noexec"]

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def sh(argv, timeout=120):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


# ---- 固定题目集：确定性轨迹 + 期望产物（答案在控制面） ----
def problems() -> list[dict]:
    probs = []
    for i in range(6):
        value = (i + 1) * 111
        probs.append({
            "id": f"p{i}",
            "statement": f"compute {value}*3 and write to answer.txt",
            "trajectory": [
                ["sh", "-c", f"printf '%d\\n' $(( {value} * 3 )) > /workspace/answer.txt"],
                ["sh", "-c", "wc -c < /workspace/answer.txt"],
            ],
            "expected_file": "answer.txt",
            "expected_content": f"{value * 3}\n",
            "allowed_workspace_files": {"answer.txt", "task.txt"},
        })
    # 越权网络注入：p2 的轨迹多一步 wget（network none → 失败 → 违规）
    probs[2]["trajectory"].append(
        ["sh", "-c", "wget -T 2 -q -O /dev/null http://example.invalid/leak || echo NET_BLOCKED"])
    probs[2]["network_violation_expected"] = True
    return probs


def new_session(problem: dict, ws_root: str) -> tuple[str, str]:
    sid = f"s{int(time.time()*1000000)%10**9:09d}{problem['id']}"
    ws = os.path.join(ws_root, sid)
    os.makedirs(ws, exist_ok=True)
    os.chmod(ws, 0o777)
    with open(os.path.join(ws, "task.txt"), "w") as f:
        f.write(problem["statement"] + "\n")
    r = sh(["docker", "run", "-d", "--name", f"mis-{CHAPTER}-{sid[:10]}",
            f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
            f"--label={dockerops.LABEL_SESSION}={sid}",
            *CONSTRAINTS, "-v", f"{ws}:/workspace", "-w", "/workspace",
            images.PINNED["alpine"], "sleep", "infinity"])
    assert r.returncode == 0, r.stderr
    return sid, ws


def replay(problem: dict, ws_root: str, plant_answer=False) -> dict:
    """执行一个 rollout：创建→回放轨迹→停止→控制面评分。"""
    t0 = time.perf_counter()
    sid, ws = new_session(problem, ws_root)
    if plant_answer:  # 注入：泄漏的答案副本（允许清单之外的隐藏文件，作弊面）
        with open(os.path.join(ws, "leaked_answers.txt"), "w") as f:
            f.write(problem["expected_content"])
    steps = []
    net_violation = False
    container = f"mis-{CHAPTER}-{sid[:10]}"
    for cmd in problem["trajectory"]:
        r = sh(["docker", "exec", container, *cmd], timeout=60)
        steps.append({"cmd": " ".join(cmd)[:60],
                      "exit": r.returncode,
                      "out_sha": hashlib.sha256((r.stdout or "").encode()).hexdigest()[:12],
                      "out": (r.stdout or "")[:80]})
        # 越权网络命令被沙箱拦截的标记：负载用 `|| echo NET_BLOCKED` 记录被拒事实，
        # 检测依据是输出标记（整体退出码因 || 而为 0，不能作判据）
        if "wget" in cmd[-1] and "NET_BLOCKED" in (r.stdout or ""):
            net_violation = True
    sh(["docker", "rm", "-f", container])
    # ---- 控制面评分（答案/评测器不在工作区）----
    traj_hash = hashlib.sha256(
        json.dumps([{k: s[k] for k in ("cmd", "exit", "out_sha")} for s in steps],
                   sort_keys=True).encode()).hexdigest()
    answer_path = os.path.join(ws, problem["expected_file"])
    got = open(answer_path).read() if os.path.exists(answer_path) else None
    correct = got == problem["expected_content"]
    files = set(os.listdir(ws))
    # 完整性扫描：任何不在允许清单的文件都算残留（含隐藏答案泄漏）
    residue = sorted(files - problem["allowed_workspace_files"])
    valid = correct and not residue and not net_violation
    shutil.rmtree(ws, ignore_errors=True)
    return {"problem": problem["id"], "steps": steps, "traj_hash": traj_hash,
            "reward": 1.0 if valid else 0.0, "correct": correct,
            "residue": residue, "net_violation": net_violation, "valid": valid,
            "total_ms": round((time.perf_counter() - t0) * 1000, 1)}


def run_batch(parallel: int) -> tuple[dict, float, float]:
    """批量模式：固定配置并行回放全部题目，返回逐题结果与创建/总时间。"""
    probs = problems()
    ws_root = os.path.join(CHAPTER_DIR, "data_local", "workspaces")
    os.makedirs(ws_root, exist_ok=True)
    t_create0 = time.perf_counter()
    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            outs = list(pool.map(lambda p: replay(p, ws_root), probs))
    else:
        outs = [replay(p, ws_root) for p in probs]
    wall = time.perf_counter() - t_create0
    return {o["problem"]: o for o in outs}, wall, wall


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)

    batch1, wall1, _ = run_batch(parallel=4)
    batch2, wall2, _ = run_batch(parallel=4)
    single, wall_single, _ = run_batch(parallel=1)

    # 1) 可复现：两次批量运行的轨迹哈希与奖励一致
    same_traj = all(batch1[p]["traj_hash"] == batch2[p]["traj_hash"] for p in batch1)
    same_reward = all(batch1[p]["reward"] == batch2[p]["reward"] for p in batch1)
    check("批量两次运行：轨迹哈希与奖励完全一致（可复现）", same_traj and same_reward,
          f"traj={batch1['p0']['traj_hash'][:12]}…")

    # 2) 单会话与批量：奖励一致
    check("单个会话与批量：逐题奖励一致",
          all(batch1[p]["reward"] == single[p]["reward"] for p in batch1))

    # 3) p2 的越权网络访问被检测
    check("越权网络访问被检测且样本判无效",
          batch1["p2"]["net_violation"] and batch1["p2"]["reward"] == 0.0,
          f"net_violation={batch1['p2']['net_violation']}")

    # 4) 正常题目奖励=1 且工作区干净
    normal_ok = all(batch1[p]["reward"] == 1.0 and not batch1[p]["residue"]
                    for p in batch1 if p != "p2")
    check("正常题全部有效（答案正确、无残留）", normal_ok,
          str({p: batch1[p]["reward"] for p in batch1}))

    # 5) 隐藏答案残留注入（p0 重跑）
    planted = replay(problems()[0],
                     os.path.join(CHAPTER_DIR, "data_local", "workspaces"),
                     plant_answer=True)
    # 注入的是"允许清单外的隐藏副本"
    check("隐藏答案残留被完整性扫描检出",
          len(planted["residue"]) > 0 and planted["reward"] == 0.0,
          f"residue={planted['residue']}")

    # 6) 吞吐与尾部
    t_totals = [batch1[p]["total_ms"] for p in batch1]
    s_totals = [single[p]["total_ms"] for p in single]
    check("批量并行总墙钟低于单个顺序", wall1 < wall_single * 0.8,
          f"batch={wall1:.1f}s single={wall_single:.1f}s")
    dockerops.cleanup_chapter(CHAPTER)
    check("实验后容器归零", len(dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}")) == 0)

    valid_n = sum(1 for p in batch1 if batch1[p]["valid"])
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "batch_run1": {k: {kk: vv for kk, vv in v.items() if kk != "steps"}
                          for k, v in batch1.items()},
           "rewards": {"batch1": {p: batch1[p]["reward"] for p in batch1},
                       "batch2": {p: batch2[p]["reward"] for p in batch2},
                       "single": {p: single[p]["reward"] for p in single}},
           "timing": {"batch_wall_s": round(wall1, 2), "single_wall_s": round(wall_single, 2),
                      "batch_per_problem_ms": stats.summary(t_totals),
                      "single_per_problem_ms": stats.summary(s_totals)},
           "cost": {"valid_samples": valid_n, "total_wall_s": round(wall1, 2),
                    "wall_per_valid_sample_s": round(wall1 / max(valid_n, 1), 3)}}
    path = results.write_result(
        CHAPTER_DIR, "rollout_batch_ab", doc,
        extra_meta={"image": images.PINNED["alpine"], "problems": len(batch1),
                    "parallel": 4, "seed": SEED,
                    "determinism": "recorded trajectories + fixed rewards"})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
