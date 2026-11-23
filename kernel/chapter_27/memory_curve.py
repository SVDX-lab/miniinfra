#!/usr/bin/env python3
"""第 27 期实验 2：状态访存对比曲线（§10.4 线性注意力状态大小与序列长度解耦）。

推导（附录口径与 ch22 capacity_accounting 的 KV 字节核算一致）：

  线性注意力（Gated DeltaNet，Qwen3-Next 线性层维度）：
    状态 S ∈ R^{16×128×128}（每层）——与序列长度 N 解耦；
    fp16 状态字节 = 16×128×128×2 = 512 KB/层。
    decode 每 token 访存（读为主）：
      读状态 512KB + 写状态 512KB + q/k/v/β/g IO ≈ 16×(128+128+128)×2 + 16×2×2
      = 12.6KB + o 写 4KB ⇒ ~1.04 MB/token/层（常数，不随 N 增长）。
    chunked prefill（块 C）每 token 摊销：
      qkv IO + 2×512KB/C（每块状态读+写一次摊到 C 个 token）。

  全注意力 KV（ch22 口径：Qwen3-0.6B GQA HKV=8, D=128, 28 层）：
    KV 字节/token/层 = 2×8×128×2 = 4 KB；N token 后缓存 = 4KB×N/层。
    decode 每 token 访存：读全部 KV = 4KB×N/层（随 N 线性增长）。

  曲线量化（N ∈ {512, 2048, 8192, 32768}，每层口径）：
    - 存储比  KV_cache(N) : state = 4KB×N : 512KB ⇒ N=128 处交叉；
    - decode 读流量比  4KB×N : 1.03MB ⇒ 同一交叉点（常数比 N/252）；
    - prefill 总读流量  full-attn Σ ≈ 4KB×N²/2 vs chunked ≈ N×(12.6KB+16KB)。

  外推口径（合成外推声明）：48 层×(36 线性 + 12 全注意) 的 Qwen3-Next 型混合
  架构 vs 48 层纯全注意（同 ch22 GQA 头数），模型级 KV/状态字节。

产出 results/memory-curve.json（纸面推导 + 确定性整数核算，无实机计时）。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deltanet_config as cfg  # noqa: E402
from experiment_utils import write_results_json  # noqa: E402

KB = 1024


def linear_layer_bytes(dtype_bytes: int = 2, chunk: int = 64):
    """每层每 token 的线性注意力访存字节（decode 读、decode 总、chunked 摊销）。"""
    heads, k_dim, v_dim = cfg.NUM_VALUE_HEADS, cfg.HEAD_K_DIM, cfg.HEAD_V_DIM
    state = heads * k_dim * v_dim * dtype_bytes
    qkv = heads * (cfg.HEAD_K_DIM * 2 + v_dim) * dtype_bytes      # q,k,v
    gates = heads * dtype_bytes * (3 if False else 2)             # β + g（标量门）
    io = qkv + gates + v_dim * dtype_bytes                        # + o 写
    decode_read = state + io - v_dim * dtype_bytes                # 读状态 + 读输入
    decode_total = state * 2 + io                                 # 状态读+写 + 输入输出
    chunked_per_token = io + 2 * state / chunk                    # 状态读+写摊到 C
    return {"state_bytes": state, "io_bytes": io,
            "decode_read_bytes": decode_read, "decode_total_bytes": decode_total,
            "chunked_per_token_bytes": chunked_per_token}


def kv_layer_bytes(dtype_bytes: int = 2, hkv: int = cfg.CH22_HKV, d: int = cfg.CH22_D):
    """每层 KV：每 token 增量字节与 N token 累计字节（ch22 口径 GQA）。"""
    per_token = 2 * hkv * d * dtype_bytes
    return {"kv_bytes_per_token": per_token}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", help="核算实验不依赖后端")
    args = parser.parse_args()

    lin = linear_layer_bytes(2)
    lin32 = linear_layer_bytes(4)
    kv = kv_layer_bytes(2)
    kv32 = kv_layer_bytes(4)

    curves = []
    for n in cfg.MEMORY_SEQ_LENS:
        kv_total = kv["kv_bytes_per_token"] * n
        decode_attn_read = kv["kv_bytes_per_token"] * n          # 读全历史
        decode_lin_read = lin["decode_read_bytes"]
        prefill_attn_reads = kv["kv_bytes_per_token"] * n * (n + 1) / 2   # Σ causal
        prefill_lin_reads = n * lin["chunked_per_token_bytes"]
        curves.append({
            "seq_len": n,
            "kv_cache_bytes_per_layer": kv_total,
            "linear_state_bytes_per_layer": lin["state_bytes"],
            "storage_ratio_kv_over_state": kv_total / lin["state_bytes"],
            "decode_read_full_attn_per_layer": decode_attn_read,
            "decode_read_linear_per_layer": decode_lin_read,
            "decode_read_ratio": decode_attn_read / decode_lin_read,
            "prefill_reads_full_attn_per_layer": prefill_attn_reads,
            "prefill_reads_linear_chunked_per_layer": prefill_lin_reads,
            "prefill_read_ratio": prefill_attn_reads / prefill_lin_reads,
        })
        print(f"[curve] N={n:>6}: KV缓存/层={kv_total/KB:.0f}KB vs 状态/层={lin['state_bytes']/KB:.0f}KB "
              f"(×{kv_total/lin['state_bytes']:.1f}); decode读 比={decode_attn_read/decode_lin_read:.1f}×; "
              f"prefill读 比={prefill_attn_reads/prefill_lin_reads:.1e}×")

    crossover_n = lin["state_bytes"] / kv["kv_bytes_per_token"]
    per_model_state = lin["state_bytes"] * cfg.CH22_NUM_LAYERS
    print(f"[crossover] 存储交叉点 N* = 状态字节/KV每token字节 = {crossover_n:.0f} token/层")
    print(f"[model] Qwen3-0.6B 28 层全换线性（外推）：状态 = {per_model_state/KB:.0f}KB（常数），"
          f"vs N=32768 KV = {kv['kv_bytes_per_token']*32768*cfg.CH22_NUM_LAYERS/KB/KB:.1f}MB")

    # Qwen3-Next 型混合外推（48 层 = 36 线性 + 12 全注意，声明合成外推口径）
    hybrid_linear_layers, hybrid_full_layers = 36, 12
    hybrid_state = hybrid_linear_layers * lin["state_bytes"]
    hybrid_kv_at = {n: hybrid_full_layers * kv["kv_bytes_per_token"] * n for n in cfg.MEMORY_SEQ_LENS}
    full_kv_at = {n: cfg.NUM_LAYERS * kv["kv_bytes_per_token"] * n for n in cfg.MEMORY_SEQ_LENS}

    result = {
        "schema_version": "kernel-chapter28.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "确定性整数核算（无实机计时）——ch22 capacity_accounting 同口径",
        "linear_layer_fp16": lin,
        "linear_layer_fp32": lin32,
        "kv_layer_fp16_gqa": kv,
        "kv_layer_fp32_gqa": kv32,
        "reference_configs": {
            "linear": "Qwen3-Next 线性层 16 头×(128K×128V)，fp16 状态 512KB/层",
            "kv": "Qwen3-0.6B GQA HKV=8×D=128，fp16 KV 4KB/token/层（ch22 口径）",
        },
        "crossover_tokens_per_layer": crossover_n,
        "memory_horizon_note": (
            "门控衰减下有效记忆窗口 1/(1−ᾱ)：ᾱ=0.9 → 10 token；状态解耦的是"
            "访存量而非信息量——衰减使远期信息指数遗忘，KV 无损保留全历史"
        ),
        "curve": curves,
        "model_level_extrapolation_synthetic": {
            "declaration": "合成外推口径：按 Qwen3-Next 公开层数 48（36 线性 + 12 全注意"
                           "）与 ch22 GQA 每层 KV 字节拼装，非官方总量声明",
            "hybrid_state_kb_fp16": hybrid_state / KB,
            "hybrid_kv_bytes_fp16_by_n": hybrid_kv_at,
            "full_attention_kv_bytes_fp16_by_n": full_kv_at,
            "hybrid_saving_ratio_by_n": {n: full_kv_at[n] / (hybrid_kv_at[n] + hybrid_state)
                                         for n in cfg.MEMORY_SEQ_LENS},
        },
    }
    write_results_json(HERE / "results" / "memory-curve.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
