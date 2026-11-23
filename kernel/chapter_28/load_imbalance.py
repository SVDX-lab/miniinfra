#!/usr/bin/env python3
"""第 28 期实验 3：负载不均对延迟的影响（构造倾斜分布，CPU 列）。

构造（ep26 planted 注入同思想）：均匀 = top-1 全均匀；倾斜 = 80% token 的
top-1 落在 20% 热专家（热集内 Zipf(α=1)），其余 k-1 均匀。唯一变量 = 分布形状
（同 x / 同专家权重 / 同 k / 同 pairs 数）。

口径：
  - e2e 分组前向（OperatorTimer ch01 口径）+ 分段（注入式 profiled）；
  - 桶结构统计：max/mean/p90 深度、空桶数——"专家桶最大深度 = 串行长度"
    的结构口径（GPU one-CTA-per-bucket 设计的临界路径；CPU 上是 MKL 每桶
    GEMM 内并行，串行深度效应弱化为"桶数×dispatch 开销 + 每桶并行度"）；
  - 线程敏感度探针：threads ∈ {12, 4, 1} × {均匀, 倾斜}（N=2048, E=64, k=6）
    ——观察 CPU 线程池下倾斜的实际代价方向与幅度；
  - 路由段耗时两分布对照（应为常量——路由与分布无关）。
产出 results/load-imbalance-cpu.json。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import moe_config as cfg  # noqa: E402
import moe_reference as mr  # noqa: E402
from experiment_utils import OperatorTimer, write_results_json  # noqa: E402

torch.set_num_threads(min(12, torch.get_num_threads()))

ROUTING_SEGS = ["gate_mm", "topk_norm", "sort_pairs", "gather_x", "scatter_combine"]
EXPERT_SEGS = ["expert_gate_up", "silu_mul", "expert_down"]


def bucket_stats(meta) -> dict:
    counts = meta["counts"]
    sorted_c, _ = counts.sort(descending=True)
    p90 = sorted_c[min(len(sorted_c) - 1, int(round(len(sorted_c) * 0.90)) - 1)]
    return {
        "depth_max": int(counts.max()),
        "depth_mean": round(float(counts.float().mean()), 2),
        "depth_p90": int(p90),
        "n_empty": int((counts == 0).sum()),
        "counts_top8": counts.sort(descending=True).values[:8].tolist(),
        "load_ratio_max_over_mean": round(float(counts.max()) / max(1.0, float(counts.float().mean())), 2),
    }


def profile_forward(x, W, idx, w, repeats):
    seg_acc: dict[str, float] = {}
    for _ in range(repeats):
        _, seg = mr.grouped_moe_profiled_pairs(x, W, idx, w)
        for name, v in seg.items():
            seg_acc[name] = seg_acc.get(name, 0.0) + v
    seg_mean = {name: v / repeats * 1000.0 for name, v in seg_acc.items()}
    routing = sum(seg_mean[s] for s in ROUTING_SEGS)
    expert = sum(seg_mean[s] for s in EXPERT_SEGS)
    return seg_mean, routing, expert


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()

    E, k = cfg.MAIN_E, cfg.MAIN_TOPK
    W = mr.make_expert_weights(E, cfg.SCAN_D_MODEL, cfg.SCAN_D_FF, cfg.SEED_BASE + 64)
    entries = []

    for N in [64, 256, 2048]:
        x = mr.make_inputs(N, cfg.SCAN_D_MODEL, cfg.SEED_BASE + 2000 + N)
        big = (N >= 2048)
        warmup, repeats = (1, 3) if big else (2, 5)

        dists = {}
        idx_u, w_u, st_u = mr.make_zipf_selection(N, E, k, cfg.SEED_BASE + 3000 + N,
                                                  token_frac=0.0)
        idx_z, w_z, st_z = mr.make_zipf_selection(N, E, k, cfg.SEED_BASE + 3000 + N,
                                                  hot_frac=cfg.ZIPF_HOT_FRAC,
                                                  token_frac=cfg.ZIPF_TOKEN_FRAC,
                                                  alpha=cfg.ZIPF_ALPHA)
        dists["uniform"] = (idx_u, w_u, st_u)
        dists["zipf8020"] = (idx_z, w_z, st_z)

        for dist_name, (idx, w, st) in dists.items():
            timer = OperatorTimer(torch.device("cpu"), warmup=warmup, repeats=repeats)
            e2e = timer.measure(lambda: mr.grouped_moe_pairs(x, W, idx, w))
            _, meta = mr.grouped_moe_pairs(x, W, idx, w, return_buckets=True)
            seg_mean, routing_ms, expert_ms = profile_forward(x, W, idx, w, repeats)
            det2 = mr.grouped_moe_pairs(x, W, idx, w)
            det1 = mr.grouped_moe_pairs(x, W, idx, w)
            entry = {
                "tokens": N, "experts": E, "topk": k, "distribution": dist_name,
                "seed_stats": st,
                "warmup": warmup, "repeats": repeats,
                "e2e": e2e,
                "segments_mean_ms": {kk: round(vv, 4) for kk, vv in seg_mean.items()},
                "routing_ms": round(routing_ms, 4),
                "expert_ms": round(expert_ms, 4),
                "routing_share": round(routing_ms / e2e["p50_ms"], 4),
                "buckets": bucket_stats(meta),
                "determinism_bitlevel": torch.equal(det1, det2),
            }
            entries.append(entry)
            b = entry["buckets"]
            print(f"[imbalance] N={N:5d} {dist_name:9s} | e2e p50={e2e['p50_ms']:8.2f}ms | "
                  f"route={routing_ms:7.3f}ms | expert={expert_ms:8.2f}ms | "
                  f"depth_max={b['depth_max']:5d} mean={b['depth_mean']:6.1f} p90={b['depth_p90']:5d} "
                  f"empty={b['n_empty']:2d} | det={entry['determinism_bitlevel']}")

        u = next(e for e in entries if e["tokens"] == N and e["distribution"] == "uniform")
        z = next(e for e in entries if e["tokens"] == N and e["distribution"] == "zipf8020")
        print(f"[imbalance] N={N} zipf/uniform: e2e={z['e2e']['p50_ms'] / u['e2e']['p50_ms']:.3f}x "
              f"expert={z['expert_ms'] / u['expert_ms']:.3f}x route={z['routing_ms'] / u['routing_ms']:.3f}x "
              f"depth_max {u['buckets']['depth_max']}→{z['buckets']['depth_max']} "
              f"({z['buckets']['depth_max'] / u['buckets']['depth_max']:.2f}x)")

    # ---- 线程敏感度探针（N=2048, E=64, k=6：均匀 vs 倾斜 × threads 12/4/1）----
    thread_probe = []
    N, x = 2048, None
    x = mr.make_inputs(N, cfg.SCAN_D_MODEL, cfg.SEED_BASE + 2000 + N)
    idx_u, w_u, _ = mr.make_zipf_selection(N, E, k, cfg.SEED_BASE + 3000 + N, token_frac=0.0)
    idx_z, w_z, _ = mr.make_zipf_selection(N, E, k, cfg.SEED_BASE + 3000 + N,
                                           hot_frac=cfg.ZIPF_HOT_FRAC,
                                           token_frac=cfg.ZIPF_TOKEN_FRAC,
                                           alpha=cfg.ZIPF_ALPHA)
    baseline_threads = torch.get_num_threads()
    for th in [12, 4, 1]:
        torch.set_num_threads(th)
        for dist_name, (idx, w) in [("uniform", (idx_u, w_u)), ("zipf8020", (idx_z, w_z))]:
            timer = OperatorTimer(torch.device("cpu"), warmup=1, repeats=3)
            e2e = timer.measure(lambda: mr.grouped_moe_pairs(x, W, idx, w))
            thread_probe.append({"threads": th, "distribution": dist_name,
                                 "e2e_p50_ms": e2e["p50_ms"], "e2e_mean_ms": e2e["mean_ms"]})
            print(f"[thread-probe] threads={th:2d} {dist_name:9s} e2e p50={e2e['p50_ms']:8.2f}ms")
    torch.set_num_threads(baseline_threads)

    payload = {
        "schema_version": "kernel-chapter29.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "cpu",
        "construction": {
            "uniform": "top-1 全均匀（token_frac=0 的同一构造器）",
            "zipf8020": f"80% token top-1 落 20% 热专家（H={cfg.ZIPF_HOT_FRAC}, α={cfg.ZIPF_ALPHA}）",
            "note": "唯一变量=分布形状；x/权重/k/pairs 数全部相同",
        },
        "serial_depth_note": "专家桶最大深度=串行长度的结构口径：GPU one-CTA-per-bucket "
                             "设计临界路径=最深桶；CPU 参考实现的专家循环为顺序 python loop "
                             "（每桶 GEMM 内部 MKL 并行），倾斜的 CPU 效应=桶数减少（dispatch "
                             "开销降）+单桶变深（桶内并行度饱和），方向由实测给出",
        "entries": entries,
        "thread_probe": thread_probe,
    }
    write_results_json(HERE / "results" / "load-imbalance-cpu.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
