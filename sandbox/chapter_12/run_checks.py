"""第 12 期实验：单 worker 基线 + 双异速 worker 上 random vs cost-aware 放置对照。

异速（1.0 / 0.5）+ 本地队列 + 长短混合负载是放置策略产生差异的条件：
random 按任务数均分，慢 worker 队列以一半速度消化，尾部等待被拉长；
cost-aware 按声明时长折算的预计排空时间放置，把工作按容量比例分给快 worker。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_12.placement_router import Worker, start_router
from common import dockerops, images, results, stats

CHAPTER = "c12"
CHAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
N_TASKS = 24          # 每轮：8 长(1.8s) + 16 短(0.4s)，交错固定
CONCURRENCY = 8
ROUNDS = 2            # 双 worker 场景轮数（聚合 48 样本）

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def task_set(round_no: int) -> list[dict]:
    tasks = []
    for i in range(N_TASKS):
        dur = 1.8 if i % 3 == 0 else 0.4
        tasks.append({"duration_s": dur,
                      "commands": [["echo", f"r{round_no}-t{i}"], ["sleep", str(dur)]]})
    return tasks


def run_scenario(name: str, workers_spec: list[tuple], policy: str, rounds: int) -> dict:
    workers = [Worker(n, s, sp) for (n, s, sp) in workers_spec]
    srv, router = start_router(workers, policy)
    agg, dist = [], Counter()
    try:
        for r in range(rounds):
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                outs = list(pool.map(router.run_task, task_set(r)))
            for o in outs:
                agg.append(o["queue_wait_ms"])
                dist[o["worker"]] += 1
    finally:
        srv.shutdown()
        for w in workers:
            w.destroy()
    wsnap = router.stats()["workers"]
    total = sum(dist.values())
    return {"scenario": name, "policy": policy, "samples": len(agg),
            "queue_wait_ms": stats.summary(agg),
            "raw": agg,
            "tasks_per_worker": dict(dist),
            "fast_share": round(dist.get("w-fast", dist.get("w1", 0)) / total, 3),
            "workers": wsnap}


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)

    single = run_scenario("single_3slots_fast", [("w1", 3, 1.0)], "aware", rounds=1)
    dual_rand = run_scenario("dual_2+2_random",
                             [("w-fast", 2, 1.0), ("w-slow", 2, 0.5)], "random", ROUNDS)
    dual_aware = run_scenario("dual_2+2_aware",
                              [("w-fast", 2, 1.0), ("w-slow", 2, 0.5)], "aware", ROUNDS)

    for sc in (single, dual_rand, dual_aware):
        print(f"[{sc['scenario']}] {json.dumps({k: v for k, v in sc.items() if k != 'raw'}, ensure_ascii=False)}")

    check("单 worker 基线：全部完成", single["samples"] == N_TASKS)
    check("双异速 random：全部完成", dual_rand["samples"] == N_TASKS * ROUNDS)
    check("双异速 aware：全部完成", dual_aware["samples"] == N_TASKS * ROUNDS)
    check("槽位约束：任意 worker 并发不超过槽位数",
          all(w["max_concurrent"] <= w["slots"] for sc in (single, dual_rand, dual_aware)
              for w in sc["workers"]),
          str([(w["name"], w["max_concurrent"]) for sc in (dual_rand, dual_aware) for w in sc["workers"]]))
    check("random 按任务数近似均分（快 worker 占比 0.35–0.65，二项波动内）",
          0.35 <= dual_rand["fast_share"] <= 0.65, f"fast_share={dual_rand['fast_share']}")
    check("cost-aware 把工作按容量比例给快 worker（占比 >0.55）",
          dual_aware["fast_share"] > 0.55, f"fast_share={dual_aware['fast_share']}")
    # 差异在尾部：random 下慢 worker 队列以半速消化，快 worker 一侧任务先完成
    # （故其中位数可能不差），被随机丢进慢队列的任务等待被拉长数倍。
    check("正解：cost-aware 排队尾部显著优于 random（P95 减半、均值更低）",
          dual_aware["queue_wait_ms"]["p95"] < dual_rand["queue_wait_ms"]["p95"] * 0.5
          and dual_aware["queue_wait_ms"]["mean"] < dual_rand["queue_wait_ms"]["mean"],
          f"random p95={dual_rand['queue_wait_ms']['p95']:.0f}ms mean={dual_rand['queue_wait_ms']['mean']:.0f}ms "
          f"aware p95={dual_aware['queue_wait_ms']['p95']:.0f}ms mean={dual_aware['queue_wait_ms']['mean']:.0f}ms")

    dockerops.cleanup_chapter(CHAPTER)
    check("实验后容器归零", len(dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}")) == 0)

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "scenarios": {"single": single, "dual_random": dual_rand, "dual_aware": dual_aware}}
    path = results.write_result(
        CHAPTER_DIR, "placement_ab", doc,
        extra_meta={"image": images.PINNED["alpine"], "tasks": N_TASKS,
                    "client_concurrency": CONCURRENCY, "rounds": ROUNDS, "seed": 42,
                    "note": "同机异速 worker 模拟节点（sleep 缩放），非真实多机"})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
