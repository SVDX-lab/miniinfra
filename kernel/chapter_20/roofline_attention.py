"""第 20 期 roofline 口径单源：FLOPs、理论字节、AI、bound 判定、KV 字节、下限。

两形态口径（改造自 ch02 roofline.py / ch18 roofline_gemm.py 方法）：

prefill（causal，per batch×q 头，S 个 query、S 个 key）：
  FLOPs = 2 × [S(S+1)/2 × D]（QK^T） + 2 × [S(S+1)/2 × D]（PV）
        = 2·S(S+1)·D            （causal 三角：每 query 平均 (q_pos+1) 个 key）
  理论字节（online-softmax kernel 理想单遍）：
    unique 口径 = (2·HQ + 2·HKV) · S · D · itemsize
      （Q 读一遍 + O 写一遍 + K/V 每 (b,kv头) 读一遍——GQA 下 K/V 被
       GQA_GROUP 个 q 头共用，DRAM 唯一流量按 kv 头计）
    issued 口径 = Q+O 每 q 头一遍 + K/V 每 q 头一遍乘以 q-tile 重复数
      （naive kernel 每 q-tile 重扫 K/V 前缀；early-termination 按 causal
       截到 ceil((q0+BR)/BC)·BC；重复读能否被 L2 吸收取决于工作集，
       两个口径都报告，实测带宽对账时分别对照）

decode（单 query per batch×q 头，N 个 KV token）：
  FLOPs = 4·N·D（QK 2ND + PV 2ND）——每头一遍，与 N 线性
  理论字节：
    unique 口径 = (2·HQ + 2·HKV) · D · itemsize × B （与 N 无关的部分）
                 + 2·HKV·N·D·itemsize × B          （K/V 每元素读一遍）
      即 KV 项 = 每 Token 每层 KV 字节 × N × B——与附录 B 题 2 同式
    issued 口径 = K/V 按 q 头各读一遍（同 kv 头的 2 个 q 头各扫一遍 cache，
      L2 可能吸收第二遍）：(2·HQ + 2·HQ)·N·D·itemsize × B

每 Token KV 字节（GQA，附录 B 题 2 口径）：
  per layer = 2 × HKV × D × dtype_bytes = 4096 B（Qwen3-0.6B FP16）
  per model = × 28 层 = 112 KB/Token

bound 判定与 ch18 同式：t_mem = bytes/带宽 vs t_compute = FLOPs/算力，
AI 与脊点比较；脊点 = 峰值算力/峰值带宽（FP16 主线口径）。
"""

from __future__ import annotations

from typing import Any

import attention_config as cfg


def gqa_kv_head(q_head: int) -> int:
    return q_head // cfg.GQA_GROUP


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------

def prefill_flops(batch: int, s: int, d: int = cfg.D, causal: bool = True) -> int:
    """prefill 总 FLOPs（QK^T + PV；causal 三角，非 causal 时满分）。"""
    per_bh_qk = 2 * (s * (s + 1) // 2 if causal else s * s) * d
    return batch * cfg.HQ * (per_bh_qk * 2)


def prefill_bytes_unique(batch: int, s: int, itemsize: int,
                         d: int = cfg.D) -> int:
    """online-softmax 单遍理想字节（K/V 按 kv 头计唯一流量）。"""
    return batch * s * d * itemsize * (2 * cfg.HQ + 2 * cfg.HKV)


def prefill_kv_tiles_issued(s: int, br: int = cfg.BR, bc: int = cfg.BC) -> int:
    """单个 (b,h_q) block 因 causal early-termination 实际扫过的 key 行数。

    q-tile t（q0=t·br）最多需要 key [0, min(S, q0+br))；按 bc 对齐的 tile 数。
    """
    q_tiles = (s + br - 1) // br
    keys = 0
    for t in range(q_tiles):
        need = min(s, t * br + br)
        keys += ((need + bc - 1) // bc) * bc
    return keys


def prefill_bytes_issued(batch: int, s: int, itemsize: int,
                         d: int = cfg.D) -> int:
    """naive kernel 实际发射的 K/V + Q/O 字节（假设无 L2 吸收的上界口径）。"""
    qo = batch * cfg.HQ * 2 * s * d * itemsize            # Q 读 + O 写（每 q 头一遍）
    kv = batch * cfg.HQ * prefill_kv_tiles_issued(s) * d * 2 * itemsize
    return qo + kv


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def decode_flops(batch: int, n: int, d: int = cfg.D) -> int:
    return batch * cfg.HQ * 4 * n * d


def decode_bytes_unique(batch: int, n: int, itemsize: int,
                        d: int = cfg.D) -> int:
    """KV 每 (b,kv头) 读一遍 + q/o 每头一遍（DRAM 唯一流量的理想下界口径）。

    KV 项 = kv_bytes_per_token_per_layer(itemsize) × N × batch = 附录 B 题 2 口径。
    """
    qo = batch * cfg.HQ * 2 * d * itemsize
    kv = batch * cfg.HKV * n * d * 2 * itemsize
    return qo + kv


def decode_bytes_issued(batch: int, n: int, itemsize: int,
                        d: int = cfg.D) -> int:
    """naive decode kernel 每 q 头各扫一遍 K/V 的发射口径（无 L2 去重假设）。"""
    return batch * cfg.HQ * (2 + 2 * n) * d * itemsize


# ---------------------------------------------------------------------------
# KV 字节（附录 B 题 2）
# ---------------------------------------------------------------------------

def kv_bytes_per_token_per_layer(itemsize: int = 2,
                                 hkv: int = cfg.HKV, d: int = cfg.D) -> int:
    return 2 * hkv * d * itemsize


def kv_bytes_per_token_model(itemsize: int = 2) -> int:
    return kv_bytes_per_token_per_layer(itemsize) * cfg.NUM_HIDDEN_LAYERS


def kv_cache_bytes(n_tokens: int, itemsize: int = 2, batch: int = 1) -> int:
    return kv_bytes_per_token_model(itemsize) * n_tokens * batch


# ---------------------------------------------------------------------------
# materialized 变体（对照测量用）：S×S scores 与 attn 各写一遍读一遍
# ---------------------------------------------------------------------------

def prefill_bytes_matmul(batch: int, s: int, itemsize: int,
                         d: int = cfg.D) -> int:
    """物化版三 kernel 的理论字节（scores fp16 同 io dtype）。

    K1 scores：读 Q,K + 写 S×S（causal 三角按满阵保守计？——不，写也只写
    下三角有效……本 kernel 实现写满阵（无效区写 0），所以按满阵 S²×itemsize）；
    K2 softmax：读 scores + 写 attn（满阵）+ 写 l；
    K3 PV：读 attn + 读 V + 写 O。
    """
    s2 = s * s
    return batch * cfg.HQ * (
        (2 * s * d + s2) * itemsize          # K1
        + 2 * s2 * itemsize + 4 * s          # K2（l fp32 每行 4B）
        + (s2 + s * d) * itemsize            # K3：attn 读 + V 读 + O 写（V,O 并入）
    ) - batch * cfg.HKV * 0  # V/O 已按 q 头口径并入 K3（GQA V 读按 q 头重复）


def prefill_bytes_matmul_unique(batch: int, s: int, itemsize: int,
                                d: int = cfg.D) -> int:
    """物化版的 DRAM 唯一口径：中间矩阵（scores/attn）只能按 q 头独占；
    K/V 仍按 kv 头唯一。额外项 = 写+读 scores + 写+读 attn。"""
    base = prefill_bytes_unique(batch, s, itemsize, d)
    extra = batch * cfg.HQ * 4 * s * s * itemsize   # scores/attn 各写一遍读一遍
    return base + extra


# ---------------------------------------------------------------------------
# AI 与 bound 判定
# ---------------------------------------------------------------------------

def arithmetic_intensity(flops: int, bytes_total: int) -> float:
    return flops / bytes_total


def ridge_point(peak_tf: float, peak_gbs: float) -> float:
    """脊点 FLOP/Byte = TFLOPS / GB/s（单位自洽：1e12/1e9 = 1e3）。"""
    return peak_tf * 1000.0 / peak_gbs


def bound_class(ai: float, platform: str, dtype: str = "fp16") -> dict[str, Any]:
    peaks = cfg.PLATFORM_PEAKS[platform]
    if dtype == "fp16":
        tf, gbs = peaks["fp16_gemm_tf_measured"], peaks["copy_gb_s_measured"]
    else:
        tf, gbs = peaks["fp32_gemm_tf_measured"], peaks["copy_gb_s_measured"]
    ridge = ridge_point(tf, gbs)
    ratio = ai / ridge
    if ratio >= 2.0:
        cls = "compute-bound (deep)"
    elif ratio > 1.0:
        cls = "compute-bound"
    elif ratio > 0.5:
        cls = "balanced (偏 memory)"
    elif ratio > 0.05:
        cls = "memory-bound"
    else:
        cls = "memory-bound (deep)"
    return {"ai_flop_per_byte": round(ai, 3), "ridge_flop_per_byte": round(ridge, 2),
            "ai_over_ridge": round(ratio, 4), "bound_class": cls,
            "platform": platform, "dtype": dtype}


def lower_bounds_ms(flops: int, bytes_total: int, platform: str,
                    dtype: str = "fp16") -> dict[str, float]:
    peaks = cfg.PLATFORM_PEAKS[platform]
    if dtype == "fp16":
        tf = peaks["fp16_gemm_tf_measured"]
    else:
        tf = peaks["fp32_gemm_tf_measured"]
    return {
        "t_mem_ms": round(bytes_total / (peaks["copy_gb_s_measured"] * 1e9) * 1000.0, 6),
        "t_compute_ms": round(flops / (tf * 1e12) * 1000.0, 6),
    }


def working_set_region(bytes_total: int, platform: str) -> str:
    l2 = cfg.PLATFORM_PEAKS[platform]["l2_cache_mb"] * 1024 * 1024
    return "L2 区" if bytes_total <= l2 else "DRAM 区"


# ---------------------------------------------------------------------------
# 便捷：单个实验点的完整理论档案
# ---------------------------------------------------------------------------

def prefill_profile(batch: int, s: int, dtype: str = "fp16",
                    platform: str = "cuda") -> dict[str, Any]:
    itemsize = 2 if dtype == "fp16" else 4
    flops = prefill_flops(batch, s)
    bu = prefill_bytes_unique(batch, s, itemsize)
    bi = prefill_bytes_issued(batch, s, itemsize)
    return {
        "form": "prefill", "batch": batch, "seq": s, "dtype": dtype,
        "flops": flops,
        "bytes_unique": bu, "bytes_issued": bi,
        "ai_unique": round(arithmetic_intensity(flops, bu), 2),
        "ai_issued": round(arithmetic_intensity(flops, bi), 2),
        "bounds_unique": bound_class(arithmetic_intensity(flops, bu), platform, dtype),
        "bounds_issued": bound_class(arithmetic_intensity(flops, bi), platform, dtype),
        "lower_bounds": lower_bounds_ms(flops, bu, platform, dtype),
        "kv_tiles_issued_per_bh": prefill_kv_tiles_issued(s),
        "working_set_unique_mb": round(bu / 1e6, 2),
        "working_set_region": working_set_region(bu, platform),
    }


def decode_profile(batch: int, n: int, dtype: str = "fp16",
                   platform: str = "cuda") -> dict[str, Any]:
    itemsize = 2 if dtype == "fp16" else 4
    flops = decode_flops(batch, n)
    bu = decode_bytes_unique(batch, n, itemsize)
    bi = decode_bytes_issued(batch, n, itemsize)
    return {
        "form": "decode", "batch": batch, "kv_len": n, "dtype": dtype,
        "flops": flops,
        "bytes_unique": bu, "bytes_issued": bi,
        "ai_unique": round(arithmetic_intensity(flops, bu), 4),
        "ai_issued": round(arithmetic_intensity(flops, bi), 4),
        "bounds_unique": bound_class(arithmetic_intensity(flops, bu), platform, dtype),
        "bounds_issued": bound_class(arithmetic_intensity(flops, bi), platform, dtype),
        "lower_bounds": lower_bounds_ms(flops, bu, platform, dtype),
        "kv_bytes_unique_mb": round(batch * cfg.HKV * n * cfg.D * 2 * itemsize / 1e6, 2),
        "working_set_region": working_set_region(
            batch * cfg.HKV * n * cfg.D * 2 * itemsize, platform),
    }


if __name__ == "__main__":
    import json

    print(cfg.describe())
    print("KV/token/layer fp16 =", kv_bytes_per_token_per_layer(2), "B;",
          "model =", kv_bytes_per_token_model(2), "B")
    print("\nprefill profiles (cuda fp16):")
    for s in [64, 512, 2048, 8192]:
        for b in [1, 4]:
            p = prefill_profile(b, s, "fp16", "cuda")
            print(f"  S={s} B={b}: AI_uniq={p['ai_unique']:>8} AI_iss={p['ai_issued']:>8} "
                  f"{p['bounds_unique']['bound_class']:<22} "
                  f"t_mem={p['lower_bounds']['t_mem_ms']:.4f}ms "
                  f"t_cmp={p['lower_bounds']['t_compute_ms']:.4f}ms {p['working_set_region']}")
    print("\ndecode profiles (cuda fp16):")
    for n in [64, 512, 2048, 8192, 32768]:
        for b in [1, 4]:
            p = decode_profile(b, n, "fp16", "cuda")
            print(f"  N={n} B={b}: AI_uniq={p['ai_unique']:>7} AI_iss={p['ai_issued']:>7} "
                  f"{p['bounds_unique']['bound_class']:<22} "
                  f"t_mem={p['lower_bounds']['t_mem_ms']:.4f}ms "
                  f"t_cmp={p['lower_bounds']['t_compute_ms']:.5f}ms {p['working_set_region']}")
