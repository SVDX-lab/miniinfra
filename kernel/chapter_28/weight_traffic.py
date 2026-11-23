#!/usr/bin/env python3
"""第 28 期实验 5："省计算不省权重读取"验证（字节数推导 + CPU 访存模拟）。

三层：
  1. 整数字节核算：单专家 / 激活（k×）/ 全池（E×）× dtype ∈ {fp32,fp16,int8,int4}；
  2. 触碰专家曲线：batch B 内 distinct 专家数——均匀解析式 E×(1−(1−k/E)^B) +
     构造路由经验值；batch=1 的省字节 vs batch 增长的快速饱和（coupon collector）；
  3. 跨 pass LRU cache 模拟（**模拟口径显式声明**）：
     - 块 = 整个专家的权重连续区（粒度=专家，无部分读）；
     - 单 pass 内同一专家只读一次（分组 GEMM 已内化 pass 内重用）；
     - cache 建模 pass 间时间重用（自回归 decode 步间的路由相关性）；
     - LRU、容量按"可容纳专家数"计（对照典型 L2/L3 尺寸换算 MB）；
     - 均匀 vs Zipf(80/20) 两流对照 → miss 字节比。
产出 results/weight-traffic.json。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import moe_config as cfg  # noqa: E402
import moe_reference as mr  # noqa: E402

torch.set_num_threads(4)


def byte_accounting() -> dict:
    d, d_ff, E = cfg.SCAN_D_MODEL, cfg.SCAN_D_FF, cfg.MAIN_E
    k = cfg.MAIN_TOPK
    rows = []
    for name, b in cfg.TRAFFIC_DTYPES.items():
        rows.append({
            "dtype": name, "bytes_per_param": b,
            "expert_mb": round(cfg.scan_ffn_bytes(b) / 1e6, 2),
            "activated_per_token_mb": round(cfg.scan_ffn_bytes(b) * k / 1e6, 2),
            "pool_mb": round(cfg.scan_ffn_bytes(b) * E / 1e6, 2),
            "activated_ratio_of_pool": round(k / E, 4),
            "l3_32mb_fit_experts": round(32e6 / cfg.scan_ffn_bytes(b), 2),
        })
    return {
        "dims": {"d_model": d, "d_ff": d_ff, "experts": E, "topk": k},
        "per_expert_bytes_formula": "3 × d_model × d_ff × dtype_bytes",
        "rows": rows,
        "claims": [
            f"每 token 激活字节 = k/E = {k}/{E} = {k / E:.4f} 的池（fp32 池 "
            f"{cfg.scan_ffn_bytes(4) * E / 1e9:.2f} GB）",
            "省 FLOPs 比 = k/E（每 token 只算 k 个专家）；batch≥~10 后触碰专家数饱和"
            "（见 touched curve）→ 权重读取字节 ≈ 全池——省计算不省权重读取",
            "batch=1 确实只读 k 个专家（字节省 k/E），但 (a) 位置随机散布无流式局部性，"
            "(b) 全池仍须驻留显存（容量成本），(c) continuous batching 下 B×k 迅速覆盖"
            "全池",
        ],
    }


def touched_curve() -> dict:
    E, k = cfg.MAIN_E, cfg.MAIN_TOPK
    B = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    gen = torch.Generator().manual_seed(cfg.SEED_BASE + 4321)
    analytic = []
    empirical = []
    expert_bytes_fp32 = cfg.scan_ffn_bytes(4)
    for b in B:
        uniform_idx = torch.stack([torch.randperm(E, generator=gen)[:k] for _ in range(b)])
        distinct = int(torch.unique(uniform_idx).numel())
        analytic.append({"batch": b,
                         "distinct_analytic": round(E * (1 - (1 - k / E) ** b), 1),
                         "distinct_empirical": distinct,
                         "touch_ratio": round(distinct / E, 4),
                         "weight_gb_fp32": round(distinct * expert_bytes_fp32 / 1e9, 3),
                         "flops_vs_dense_capacity_ratio": round(k / E, 4)})
    # batch=1 对照（三个 topk）
    b1 = {}
    for kk in cfg.SCAN_TOPK:
        idx1 = torch.stack([torch.randperm(E, generator=gen)[:kk] for _ in range(64)])
        b1[kk] = round(int(torch.unique(idx1).numel()) / (64 * kk), 3)
    return {
        "uniform_analytic_formula": "E × (1 − (1−k/E)^B)",
        "points": analytic,
        "saturation_note": "k=6, E=64：B=8 已触 ≥50%，B=64 触 >97%——continuous batching "
                           "数个请求后 MoE 权重读取 ≈ dense 等价容量",
        "batch1_selection_uniformity_check_topk": b1,
        "e8_curve": _touched_curve_e8(),
    }


def _touched_curve_e8() -> list:
    E, k = 8, 2
    gen = torch.Generator().manual_seed(cfg.SEED_BASE + 4322)
    rows = []
    for b in [1, 2, 4, 8, 16, 32, 64]:
        idx = torch.stack([torch.randperm(E, generator=gen)[:k] for _ in range(b)])
        distinct = int(torch.unique(idx).numel())
        rows.append({"batch": b, "distinct_analytic": round(E * (1 - (1 - k / E) ** b), 1),
                     "distinct_empirical": distinct, "touch_ratio": round(distinct / E, 4)})
    return rows


class LRUCache:
    def __init__(self, capacity_experts: int):
        self.cap = capacity_experts
        self.od: OrderedDict[int, None] = OrderedDict()
        self.misses = 0
        self.hits = 0

    def access(self, expert: int) -> None:
        if expert in self.od:
            self.hits += 1
            self.od.move_to_end(expert)
        else:
            self.misses += 1
            if len(self.od) >= self.cap:
                self.od.popitem(last=False)
            self.od[expert] = None


def cache_sim(passes: int = 8, n_per_pass: int = 2048) -> dict:
    E, k = cfg.MAIN_E, cfg.MAIN_TOPK
    expert_bytes = {name: cfg.scan_ffn_bytes(b) for name, b in cfg.TRAFFIC_DTYPES.items()}
    streams = {}
    idx_u, _, st_u = mr.make_zipf_selection(n_per_pass, E, k, cfg.SEED_BASE + 5000, token_frac=0.0)
    streams["uniform"] = idx_u
    idx_z, _, st_z = mr.make_zipf_selection(n_per_pass, E, k, cfg.SEED_BASE + 5001,
                                            hot_frac=cfg.ZIPF_HOT_FRAC,
                                            token_frac=cfg.ZIPF_TOKEN_FRAC,
                                            alpha=cfg.ZIPF_ALPHA)
    streams["zipf8020_top1"] = idx_z
    idx_zk, _, st_zk = mr.make_zipf_selection(n_per_pass, E, k, cfg.SEED_BASE + 5002,
                                              hot_frac=cfg.ZIPF_HOT_FRAC,
                                              token_frac=cfg.ZIPF_TOKEN_FRAC,
                                              alpha=cfg.ZIPF_ALPHA, hot_full_k=True)
    streams["zipf8020_fullk"] = idx_zk
    # 多 pass：decode 步间流（同一流重复 = 时间相关路由的保守口径；相邻 pass 流独立采样会更悲观）
    rows = []
    for cap in [4, 8, 16, 32, 64, 128]:
        for dist, idx in streams.items():
            cache = LRUCache(cap)
            for _ in range(passes):
                for t in range(n_per_pass):
                    for e in idx[t].tolist():
                        cache.access(e)
            row = {"capacity_experts": cap, "distribution": dist,
                   "accesses": passes * n_per_pass * k,
                   "misses": cache.misses, "hit_rate": round(1 - cache.misses / (passes * n_per_pass * k), 4),
                   "miss_bytes_fp32_per_token_kb": round(cache.misses * expert_bytes["fp32"] / (passes * n_per_pass) / 1e3, 1),
                   "miss_bytes_fp16_per_token_kb": round(cache.misses * expert_bytes["fp16"] / (passes * n_per_pass) / 1e3, 1)}
            rows.append(row)
    # 汇总：同容量下各倾斜流/uniform miss 比
    ratios = []
    for cap in [4, 8, 16, 32, 64, 128]:
        u = next(r for r in rows if r["capacity_experts"] == cap and r["distribution"] == "uniform")
        for dist in ["zipf8020_top1", "zipf8020_fullk"]:
            z = next(r for r in rows if r["capacity_experts"] == cap and r["distribution"] == dist)
            ratios.append({"capacity_experts": cap, "distribution": dist,
                           "miss_over_uniform": round(z["misses"] / max(1, u["misses"]), 4),
                           "uniform_hit_rate": u["hit_rate"], f"{dist}_hit_rate": z["hit_rate"]})
    return {
        "simulation_scope": "块=整专家权重区（无部分读）；单 pass 内每触碰专家只读一次"
                            "（分组已内化 pass 内重用）；cache 建模 pass 间时间重用"
                            f"（{passes} 个 decode 步 × {n_per_pass} token/步，步间路由流相同"
                            "=时间相关保守口径）；LRU；无预取；容量单位=专家数；"
                            "zipf8020_top1=仅 top-1 受热（其余 k-1 全池均匀）、"
                            "zipf8020_fullk=热 token 全部 k 个专家来自热集（13 个热专家）",
        "expert_mb_by_dtype": {name: round(b / 1e6, 2) for name, b in expert_bytes.items()},
        "hotset_stats": {"top1_variant": st_z, "fullk_variant": st_zk, "uniform": st_u},
        "rows": rows,
        "skew_over_uniform_by_capacity": ratios,
        "finding": "top-1-only 倾斜对 cache 增益小（热访问占比被其余 k-1 均匀摊薄到 ~top1 份额）；"
                   "full-k 倾斜（热 token 的 k 个专家全在热集）才使热集驻留生效——"
                   "真实 MoE 流量的倾斜形态决定 cache/预取策略是否有效",
        "l3_reference_note": "32MB L3 参照：fp32 0.93 专家 / fp16 1.85 / int8 3.7 / int4 7.4"
                             "——倾斜分布 + INT4 时热集可驻留片上，均匀分布下 cache 几乎无增益",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    acc = byte_accounting()
    curve = touched_curve()
    sim = cache_sim()
    for r in acc["rows"]:
        print(f"[bytes] {r['dtype']:5s} expert={r['expert_mb']:7.2f}MB 激活={r['activated_per_token_mb']:7.2f}MB "
              f"池={r['pool_mb']:8.1f}MB 激活比={r['activated_ratio_of_pool']:.4f} 32MB-L3≈{r['l3_32mb_fit_experts']:5.2f}专家")
    print("[curve] k=6,E=64 touched:")
    for p in curve["points"][:9]:
        print(f"    B={p['batch']:5d} distinct={p['distinct_empirical']:3d} ({p['touch_ratio']:.2f}) "
              f"analytic={p['distinct_analytic']:6.1f} weight={p['weight_gb_fp32']:5.2f}GB")
    print("[cache] skew/uniform miss by capacity:")
    for r in sim["skew_over_uniform_by_capacity"]:
        print(f"    cap={r['capacity_experts']:3d} {r['distribution']:16s} "
              f"miss/uniform={r['miss_over_uniform']:.3f} (uniform_hit={r['uniform_hit_rate']:.3f})")
    payload = {
        "schema_version": "kernel-chapter29.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "byte_accounting": acc,
        "touched_curve": curve,
        "cache_simulation": sim,
    }
    out = HERE / "results" / "weight-traffic.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[traffic] 写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
