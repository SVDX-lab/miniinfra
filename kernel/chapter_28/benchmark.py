#!/usr/bin/env python3
"""第 28 期实验 2：路由开销占比矩阵（CPU 列主口径）。

矩阵：tokens ∈ {1,8,64,256,2048} × experts ∈ {8,64} × topk ∈ {2,6,8}
  （k=8 且 E=8 = 退化情形 top-k=E，保留并标注）。

口径：
  - e2e = OperatorTimer（ch01：warmup=2 repeats=5；N=2048 大格 warmup=1 repeats=3）；
  - 分段 = grouped_moe_profiled 单遍 8 段 perf_counter，独立跑 repeats 趟取均值
    （gate_mm/topk_norm/sort_pairs/gather_x/expert_gate_up/silu_mul/expert_down/
    scatter_combine；routing = 前四段 + scatter_combine，expert = 三段专家）；
  - 分母对照 = 等价容量单体大 GEMM（dense FFN intermediate = k×d_ff，同专家段
    FLOPs；权重建一次在计时外）；
  - 附：触碰专家数（distinct）/激活权重字节——实验 5 的实测输入。
产出 results/benchmark-results-cpu.json。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import moe_config as cfg  # noqa: E402
import moe_reference as mr  # noqa: E402
from experiment_utils import OperatorTimer, resolve_device, write_results_json  # noqa: E402

torch.set_num_threads(min(12, torch.get_num_threads()))

ROUTING_SEGS = ["gate_mm", "topk_norm", "sort_pairs", "gather_x", "scatter_combine"]
EXPERT_SEGS = ["expert_gate_up", "silu_mul", "expert_down"]


def build_dense_weights(k: int, seed: int, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    d_ff_total = k * cfg.SCAN_D_FF
    d = cfg.SCAN_D_MODEL
    Wg = torch.randn(d_ff_total, d, generator=gen, dtype=torch.float32) / (d ** 0.5)
    Wu = torch.randn(d_ff_total, d, generator=gen, dtype=torch.float32) / (d ** 0.5)
    Wd = torch.randn(d, d_ff_total, generator=gen, dtype=torch.float32) / (d_ff_total ** 0.5)
    return Wg.to(dtype), Wu.to(dtype), Wd.to(dtype)


def dense_forward(x: torch.Tensor, Wg, Wu, Wd) -> torch.Tensor:
    return (mr.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--tokens", type=int, nargs="*", default=cfg.SCAN_TOKENS)
    args = parser.parse_args()
    device = resolve_device(args.backend)

    weights_cache: dict[int, dict] = {}
    entries = []

    for E in cfg.SCAN_EXPERTS:
        if E not in weights_cache:
            t0 = time.perf_counter()
            weights_cache[E] = mr.make_expert_weights(E, cfg.SCAN_D_MODEL, cfg.SCAN_D_FF,
                                                      cfg.SEED_BASE + E)
            print(f"[setup] E={E} 权重生成 {time.perf_counter() - t0:.1f}s "
                  f"({sum(v.numel() for v in weights_cache[E].values()) * 4 / 1e9:.2f} GB fp32)")
        W = weights_cache[E]

        for N in args.tokens:
            x = mr.make_inputs(N, cfg.SCAN_D_MODEL, cfg.SEED_BASE + 1000 + N)
            for k in cfg.SCAN_TOPK:
                degenerate = (k >= E)
                big = (N >= 2048)
                warmup, repeats = (1, 3) if big else (2, 5)

                # e2e
                timer = OperatorTimer(device, warmup=warmup, repeats=repeats)
                e2e = timer.measure(lambda: mr.grouped_moe(x, W, _ROUTER[E], k))

                # 分段（独立 profiled 趟）
                seg_acc: dict[str, float] = {}
                idx_last = None
                prof_repeats = repeats
                for _ in range(prof_repeats):
                    _, seg = mr.grouped_moe_profiled(x, W, _ROUTER[E], k)
                    for name, v in seg.items():
                        seg_acc[name] = seg_acc.get(name, 0.0) + v
                    idx_last = mr.route(x, _ROUTER[E], k)[0]
                seg_mean = {name: v / prof_repeats for name, v in seg_acc.items()}
                # grouped_moe_profiled 返回秒；统一转 ms
                seg_mean = {name: v * 1000.0 for name, v in seg_mean.items()}
                routing_ms = sum(seg_mean[s] for s in ROUTING_SEGS)
                expert_ms = sum(seg_mean[s] for s in EXPERT_SEGS)
                e2e_mean = e2e["mean_ms"]
                touched = int(torch.unique(idx_last).numel())
                flops = mr.moe_flop_counts(N, E, k, cfg.SCAN_D_MODEL, cfg.SCAN_D_FF)
                expert_bytes_fp32 = mr.expert_weight_bytes(1, cfg.SCAN_D_MODEL, cfg.SCAN_D_FF, 4)

                # 等价容量单体 GEMM（同专家段 FLOPs）
                Wdg, Wu_d, Wd_d = build_dense_weights(k, cfg.SEED_BASE + 7000 + E + k)
                dense_t = OperatorTimer(device, warmup=warmup, repeats=repeats)
                dense = dense_t.measure(lambda: dense_forward(x, Wdg, Wu_d, Wd_d))

                entry = {
                    "tokens": N, "experts": E, "topk": k,
                    "degenerate_topk_equals_E": degenerate,
                    "dtype": "float32", "d_model": cfg.SCAN_D_MODEL, "d_ff": cfg.SCAN_D_FF,
                    "warmup": warmup, "repeats": repeats,
                    "e2e": e2e,
                    "segments_mean_ms": {name: round(v, 4) for name, v in seg_mean.items()},
                    "routing_ms": round(routing_ms, 4),
                    "expert_ms": round(expert_ms, 4),
                    "routing_share_vs_e2e_mean": round(routing_ms / e2e_mean, 4) if e2e_mean else None,
                    "expert_share_vs_e2e_mean": round(expert_ms / e2e_mean, 4) if e2e_mean else None,
                    "dense_equiv": dense,
                    "moe_vs_dense_ratio": round(e2e["p50_ms"] / dense["p50_ms"], 3) if dense["p50_ms"] else None,
                    "pairs": flops["pairs"], "flops_total": flops["flops_total"],
                    "experts_touched": touched,
                    "touch_ratio": round(touched / E, 4),
                    "expert_weight_mb_fp32": round(expert_bytes_fp32 / 1e6, 2),
                    "touched_weight_mb_fp32": round(touched * expert_bytes_fp32 / 1e6, 2),
                }
                entries.append(entry)
                print(f"[bench] N={N:5d} E={E:3d} k={k} | e2e p50={e2e['p50_ms']:9.3f}ms | "
                      f"route={routing_ms:8.3f}ms ({entry['routing_share_vs_e2e_mean'] * 100 if entry['routing_share_vs_e2e_mean'] is not None else 0:5.1f}%) | "
                      f"expert={expert_ms:9.3f}ms ({entry['expert_share_vs_e2e_mean'] * 100 if entry['expert_share_vs_e2e_mean'] is not None else 0:5.1f}%) | "
                      f"dense={dense['p50_ms']:9.3f}ms | moe/dense={entry['moe_vs_dense_ratio']:6.2f}x | "
                      f"touch={touched}/{E}")

                del Wdg, Wu_d, Wd_d

    payload = {
        "schema_version": "kernel-chapter29.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "note": "CPU eager 口径：小 batch 段耗时含 PyTorch dispatch/Python 开销——"
                "这正是'小 batch 路由开销吃掉收益'在该平台的机制；GPU 列回填后按"
                "kernel/launch 口径重测",
        "entries": entries,
    }
    write_results_json(HERE / "results" / "benchmark-results-cpu.json", payload)
    return 0


_ROUTER: dict[int, torch.Tensor] = {}


def _setup_routers() -> None:
    for E in cfg.SCAN_EXPERTS:
        _ROUTER[E] = mr.make_router(E, cfg.SCAN_D_MODEL, cfg.SEED_BASE + E)


if __name__ == "__main__":
    _setup_routers()
    raise SystemExit(main())
