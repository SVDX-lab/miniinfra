"""第 09 期实验：镜像冷启动分段计时（冷拉取 vs 预拉取缓存，不混报）。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images, results, stats

CHAPTER = "c09"
REF = images.PINNED["busybox"]
TAG = "busybox:1.36"

CONSTRAINTS = ["--network", "none", "--read-only", "--user", "65534:65534",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--memory", "64m", "--pids-limit", "32",
               "--tmpfs", "/tmp:rw,size=4m,noexec"]

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def sh(argv, timeout=300):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return (time.perf_counter() - t0) * 1000.0, r


def one_iteration(cold: bool) -> dict:
    name = f"mis-{CHAPTER}-{int(time.time()*100000)%10000000}"
    seg = {"queue_ms": 0.0}  # 单线程实验，排队恒为零，如实记录
    if cold:
        sh(["docker", "rmi", TAG])
        seg["pull_ms"], pr = timed(lambda: sh(["docker", "pull", REF]))
        assert pr.returncode == 0, pr.stderr
    else:
        # 暖缓存：确保在本地（首迭代显式 pull，其后命中）
        sh(["docker", "pull", REF])
        seg["pull_ms"] = 0.0  # 纯缓存语义：跳过准备段
    seg["run_ms"], rr = timed(lambda: sh(["docker", "run", "-d", "--name", name,
                                          f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
                                          *CONSTRAINTS, REF, "sleep", "infinity"]))
    assert rr.returncode == 0, rr.stderr
    seg["ready_ms"], er = timed(lambda: sh(["docker", "exec", name, "echo", "ok"]))
    ok = er.returncode == 0 and er.stdout.strip() == "ok"
    sh(["docker", "rm", "-f", name])
    seg["total_ms"] = seg["queue_ms"] + seg["pull_ms"] + seg["run_ms"] + seg["ready_ms"]
    seg["ready_ok"] = ok
    return seg


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)
    images.ensure(REF)
    r = sh(["docker", "image", "inspect", REF, "--format", "{{.Size}}"])
    image_bytes = int(r.stdout.strip())

    # 冷组（低复用：每次删镜像重拉，5 次）
    cold = [one_iteration(cold=True) for _ in range(5)]
    # 暖组（高复用：缓存命中，10 次）
    warm = [one_iteration(cold=False) for _ in range(10)]

    def seg_summary(rows, key):
        return stats.summary([x[key] for x in rows])

    cold_total = seg_summary(cold, "total_ms")
    warm_total = seg_summary(warm, "total_ms")
    cold_pull = seg_summary(cold, "pull_ms")
    warm_run = seg_summary(warm, "run_ms")
    cold_run = seg_summary(cold, "run_ms")

    check("冷组每次都真实拉取（pull 段远大于 0）", cold_pull["min"] > 100,
          f"pull p50={cold_pull['p50']:.0f}ms min={cold_pull['min']:.0f}ms")
    check("暖组准备段为零、启动就绪正常",
          all(x["pull_ms"] == 0.0 for x in warm) and all(x["ready_ok"] for x in warm))
    check("冷组总延迟显著高于暖组（>3×）",
          cold_total["p50"] > warm_total["p50"] * 3,
          f"cold p50={cold_total['p50']:.0f}ms warm p50={warm_total['p50']:.0f}ms")
    check("冷启动时间主要在镜像准备段（pull 占比 >50%）",
          cold_pull["p50"] > (cold_total["p50"] * 0.5),
          f"pull={cold_pull['p50']:.0f} / total={cold_total['p50']:.0f}ms")

    dockerops.cleanup_chapter(CHAPTER)
    left = dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}")
    check("实验后容器归零", len(left) == 0)

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "image_bytes": image_bytes,
           "cold": {"segments": {k: seg_summary(cold, k)
                                 for k in ("queue_ms", "pull_ms", "run_ms", "ready_ms", "total_ms")},
                    "raw": cold},
           "warm": {"segments": {k: seg_summary(warm, k)
                                 for k in ("queue_ms", "pull_ms", "run_ms", "ready_ms", "total_ms")},
                    "raw": warm},
           "note": "冷组含 Docker Hub 外网延迟；暖组 pull=0 为纯缓存语义（跳过准备段）"}
    path = results.write_result(
        os.path.dirname(os.path.abspath(__file__)), "cold_start_segments", doc,
        extra_meta={"image": REF, "cache_states": {"cold": 5, "warm": 10},
                    "network": "dockerhub-live"})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    print(json.dumps({"cold_total": cold_total, "warm_total": warm_total,
                      "cold_pull": cold_pull, "warm_run": warm_run, "cold_run": cold_run},
                     indent=1))
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
