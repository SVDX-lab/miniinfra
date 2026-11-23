#!/usr/bin/env python3
"""实验 4：GQA 下 KV 字节核算 + S4000 48GB vs 4090 24GB 可服务容量对照。

纯计算 + 记账模拟（不依赖任何实机——S4000 失联期可完整执行）：

  1. 每 Token KV 字节（附录 B 题 2/6 校核，Qwen3-0.6B 口径）：
       per layer = 2 (K/V) × HKV(8) × D(128) × dtype_bytes
       fp16 = 4096 B/层 × 28 层 = 112 KB/Token；fp32 = 8192 B/层 = 224 KB；
       MHA 对照（HKV=16）= 8192 B/层 —— GQA 省 1/2；
  2. 可服务 Token 容量：总显存 − 权重 − 预留 = KV 预算（块粒度取整）；
     权重口径 = ch02 实测 1.19 GB（Qwen3-0.6B fp16 全模型读一遍）；预留扫描
     {1, 2, 4, 8} GB（激活/workspace/框架开销——第 09 期 graph pool 实测
     105/85MB·图、解码工作集约百 MB 量级，预留档位覆盖之）；
  3. batch 均分口径：batch ∈ {32, 64, 128, 256, 512} × 每请求平均上下文；
  4. 预留浪费对照（连续预留方案的容量含义）：[batch, max_ctx] 静态预留
     是否装得下（64×8192 = 58.7 GB > S4000 48GB——连续预留直接装不下）；
  5. 块表开销：max_blocks×4B/请求（GBs 量级以下，可忽略但记账在案）。

产出 results/capacity-accounting.json。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paged_config as cfg  # noqa: E402
import roofline_paged as rp  # noqa: E402
from experiment_utils import write_results_json  # noqa: E402

WEIGHTS_GB_FP16 = 1.19     # ch02 实测（Qwen3-0.6B fp16 权重读一遍口径）
WEIGHTS_GB_FP32 = 2.38
RESERVES_GB = [1.0, 2.0, 4.0, 8.0]
SERVE_BATCHES = [32, 64, 128, 256, 512]
RESERVE_SCENARIOS = [  # (batch, max_ctx, mean_len)
    (16, 2048, 512), (32, 2048, 512), (64, 2048, 512),
    (16, 8192, 1024), (32, 8192, 1024), (64, 8192, 1024),
    (64, 4096, 768), (128, 4096, 768),
]


def main() -> int:
    # ---- 1. 每 Token KV 字节 ----
    per_token = {}
    for name, itemsize in (("fp16", 2), ("fp32", 4)):
        per_token[name] = {
            "per_layer_bytes": rp.kv_bytes_per_token_per_layer(itemsize),
            "per_model_bytes": rp.kv_bytes_per_token_model(itemsize),
            "per_model_kb": rp.kv_bytes_per_token_model(itemsize) / 1024,
            "mha_per_layer_bytes": rp.kv_bytes_per_token_per_layer(itemsize, hkv=16),
            "gqa_saving_ratio": rp.kv_bytes_per_token_per_layer(itemsize, hkv=16)
            / rp.kv_bytes_per_token_per_layer(itemsize),
        }
    print(f"[cap] KV/token/layer fp16 = {per_token['fp16']['per_layer_bytes']} B "
          f"(GQA 省 1/{per_token['fp16']['gqa_saving_ratio']:.2f} vs MHA); "
          f"model = {per_token['fp16']['per_model_kb']:.0f} KB/Token")

    # ---- 2. 可服务容量（两卡 × 两 dtype × 预留扫描）----
    capacity = {}
    for plat, peaks in cfg.PLATFORM_PEAKS.items():
        capacity[plat] = {"gpu_name": peaks["gpu_name"],
                          "total_mem_gb": peaks["total_mem_gb"], "rows": []}
        for wname, wg in (("fp16_weights", WEIGHTS_GB_FP16),
                          ("fp32_weights", WEIGHTS_GB_FP32)):
            for reserve in RESERVES_GB:
                for itemsize, dname in ((2, "fp16_kv"), (4, "fp32_kv")):
                    cap = rp.serveable_tokens(peaks["total_mem_gb"], wg, reserve,
                                              itemsize, cfg.DEFAULT_BLOCK_SIZE)
                    cap["weights_case"] = wname
                    cap["kv_dtype"] = dname
                    capacity[plat]["rows"].append(cap)
    # 主线行（fp16 权重 + fp16 KV + 预留 2GB）打印
    for plat in capacity:
        for row in capacity[plat]["rows"]:
            if row["weights_case"] == "fp16_weights" and row["kv_dtype"] == "fp16_kv" \
                    and row["reserve_gb"] == 2.0:
                print(f"[cap] {plat}: 总 {row['total_mem_gb']}GB - 权重 "
                      f"{row['weights_gb']}GB - 预留 {row['reserve_gb']}GB "
                      f"→ KV 预算 {row['kv_budget_gb']}GB = "
                      f"{row['serveable_tokens']:,} Token "
                      f"({row['serveable_tokens_million']}M, "
                      f"{row['serveable_blocks']:,} blocks @bs={cfg.DEFAULT_BLOCK_SIZE})")

    # ---- 3. batch 均分口径 ----
    batch_table = {}
    for plat, peaks in cfg.PLATFORM_PEAKS.items():
        budget = (peaks["total_mem_gb"] - WEIGHTS_GB_FP16 - 2.0) * 1e9
        tokens = int(budget // rp.kv_bytes_per_token_model(2))
        rows = []
        for batch in SERVE_BATCHES:
            avg_ctx = tokens // batch
            blocks = avg_ctx // cfg.DEFAULT_BLOCK_SIZE
            rows.append({
                "batch": batch, "avg_context_tokens_per_request": avg_ctx,
                "avg_context_kb": round(avg_ctx * rp.kv_bytes_per_token_model(2) / 1024, 1),
                "blocks_per_request": blocks,
                "block_table_bytes_per_request": blocks * 4,
            })
            print(f"[cap] {plat} batch={batch:>3}: 平均上下文 {avg_ctx:,} Token/请求")
        batch_table[plat] = {"kv_budget_tokens": tokens, "rows": rows}

    # ---- 4. 预留浪费对照（连续预留方案的容量含义）----
    reservation = []
    for batch, max_ctx, mean_len in RESERVE_SCENARIOS:
        r = rp.reservation_waste_gb(batch, max_ctx, mean_len, 2)
        for plat, peaks in cfg.PLATFORM_PEAKS.items():
            r[f"fits_{plat}"] = r["reserved_gb"] <= peaks["total_mem_gb"] - WEIGHTS_GB_FP16 - 2.0
        r["paged_reserved_gb_at_mean"] = round(
            rp.kv_cache_bytes(mean_len, 2, batch) / 1e9, 3)
        reservation.append(r)
        print(f"[cap] reserve B={batch:>3} ctx={max_ctx:>5} mean={mean_len:>4}: "
              f"reserved={r['reserved_gb']:>7.3f}GB used={r['used_gb']:>6.3f}GB "
              f"waste={r['waste_gb']:>7.3f}GB util={r['utilization_pct']:>5.1f}% "
              f"fits4090={r['fits_cuda']} fitsS4000={r['fits_musa']}")

    # ---- 5. 块表开销总量级 ----
    table_overhead = {}
    for plat, peaks in cfg.PLATFORM_PEAKS.items():
        budget = (peaks["total_mem_gb"] - WEIGHTS_GB_FP16 - 2.0) * 1e9
        tokens = int(budget // rp.kv_bytes_per_token_model(2))
        blocks = tokens // cfg.DEFAULT_BLOCK_SIZE
        table_overhead[plat] = {
            "blocks": blocks,
            "table_bytes_all_blocks": blocks * 4,
            "table_overhead_pct": round(blocks * 4 / budget * 100.0, 6),
            "granularity_waste_tokens": rp.serveable_tokens(
                peaks["total_mem_gb"], WEIGHTS_GB_FP16, 2.0, 2,
                cfg.DEFAULT_BLOCK_SIZE)["granularity_waste_tokens"],
        }
        print(f"[cap] {plat} 块表总量 {blocks * 4/1e6:.2f} MB "
              f"({table_overhead[plat]['table_overhead_pct']}% 预算)")

    payload = {
        "schema_version": "kernel-chapter22.capacity.v1",
        "method": "纯计算+记账模拟；权重口径 ch02 实测 1.19GB(fp16)；"
                  "预留档 {1,2,4,8}GB；块粒度 bs=128 向下取整",
        "kv_bytes_per_token": per_token,
        "capacity_rows": capacity,
        "batch_context_table": batch_table,
        "reservation_waste": reservation,
        "block_table_overhead": table_overhead,
    }
    write_results_json(HERE / "results" / "capacity-accounting.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
