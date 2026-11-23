"""第 21 期 roofline/记账口径单源：KV 字节、gather 理论字节、AI、bound、容量。

方法沿用 chapter_20 roofline_attention.py（ch02 方法系），新增分页记账项：

decode（单 query per batch×q 头，N 个 KV token）：
  FLOPs = 4·N·D（QK 2ND + PV 2ND）——与布局无关（分页不改计算量）
  unique 字节 = (2·HQ + 2·HKV)·D·itemsize × B + 2·HKV·N·D·itemsize × B
    —— KV 项即"每 Token KV 字节 × N × B"（附录 B 题 2 口径，ep21 已校核）
  issued 字节 = K/V 按 q 头各扫一遍（GQA 2 q 头共用 1 kv 头，L2 可能吸收第二遍）
  分页新增项：
    block_table 字节 = B × max_blocks × 4（int32，每 kernel 启动读一遍）
    —— 理论上分页**不增加** DRAM 唯一字节（表极小且 L1/L2 常驻；值透明）。
    实测的带宽损失全部是"间接寻址 + 局部性损失"的代价（见性能对账）。
    随机表档的行粒度 = block_size×D×itemsize（一个 (块,kv头) 的连续段），
    bs=128 fp16 = 32 KB —— 局部性损失的物理粒度。

prefill paged（可选对照）：FLOPs/字节口径与 ep21 prefill 相同，表字节另加。

KV 字节（附录 B 题 2/6 校核）：
  per layer = 2 × HKV × D × dtype_bytes = 4096 B（fp16）
  per model = × 28 层 = 112 KB/Token
  可服务容量 = (总显存 − 权重 − 预留) / 每 Token KV 字节，块粒度向上取整。

bound 判定同 ch18/ch21 式：AI 与脊点（峰值算力/峰值带宽）比较。
"""

from __future__ import annotations

from typing import Any

import paged_config as cfg


# ---------------------------------------------------------------------------
# decode（decode 形态两布局共用：值透明 → 计算量与唯一字节与布局无关）
# ---------------------------------------------------------------------------

def decode_flops(batch: int, n: int, d: int = cfg.D) -> int:
    return batch * cfg.HQ * 4 * n * d


def decode_bytes_unique(batch: int, n: int, itemsize: int,
                        d: int = cfg.D) -> int:
    """KV 每 (b,kv头) 读一遍 + q/o 每头一遍（DRAM 唯一流量理想下界）。"""
    qo = batch * cfg.HQ * 2 * d * itemsize
    kv = batch * cfg.HKV * n * d * 2 * itemsize
    return qo + kv


def decode_bytes_issued(batch: int, n: int, itemsize: int,
                        d: int = cfg.D) -> int:
    """naive decode kernel 每 q 头各扫一遍 K/V 的发射口径（无 L2 去重假设）。"""
    return batch * cfg.HQ * (2 + 2 * n) * d * itemsize


def block_table_bytes(batch: int, seq_len: int, block_size: int,
                      itemsize_table: int = 4) -> int:
    """块表字节：[B, max_blocks] int32（max_blocks = ceil(N/bs)，不含 padding 余量）。"""
    max_blocks = (seq_len + block_size - 1) // block_size
    return batch * max_blocks * itemsize_table


def decode_bytes_paged_unique(batch: int, n: int, itemsize: int,
                              block_size: int, d: int = cfg.D) -> int:
    """paged decode 的 DRAM 唯一字节 = 连续口径 + 块表（值透明：KV 项不变）。"""
    return (decode_bytes_unique(batch, n, itemsize, d)
            + block_table_bytes(batch, n, block_size))


def block_granule_kb(block_size: int, itemsize: int = 2,
                     d: int = cfg.D) -> float:
    """随机块表下局部性损失的物理粒度：(块,kv头) 连续段 = bs×D×itemsize。"""
    return block_size * d * itemsize / 1024.0


# ---------------------------------------------------------------------------
# prefill paged（可选对照；口径与 ep21 prefill 相同，另加表字节）
# ---------------------------------------------------------------------------

def prefill_flops(batch: int, s: int, d: int = cfg.D, causal: bool = True) -> int:
    per_bh_qk = 2 * (s * (s + 1) // 2 if causal else s * s) * d
    return batch * cfg.HQ * (per_bh_qk * 2)


def prefill_kv_tiles_issued(s: int, br: int = 16, bc: int = 128) -> int:
    q_tiles = (s + br - 1) // br
    keys = 0
    for t in range(q_tiles):
        need = min(s, t * br + br)
        keys += ((need + bc - 1) // bc) * bc
    return keys


def prefill_bytes_unique(batch: int, s: int, itemsize: int,
                         d: int = cfg.D) -> int:
    return batch * s * d * itemsize * (2 * cfg.HQ + 2 * cfg.HKV)


def prefill_bytes_issued(batch: int, s: int, itemsize: int,
                         d: int = cfg.D) -> int:
    qo = batch * cfg.HQ * 2 * s * d * itemsize
    kv = batch * cfg.HQ * prefill_kv_tiles_issued(s) * d * 2 * itemsize
    return qo + kv


# ---------------------------------------------------------------------------
# KV 字节与容量核算（附录 B 题 2/6；实验 4）
# ---------------------------------------------------------------------------

def kv_bytes_per_token_per_layer(itemsize: int = 2,
                                 hkv: int = cfg.HKV, d: int = cfg.D) -> int:
    return 2 * hkv * d * itemsize


def kv_bytes_per_token_model(itemsize: int = 2) -> int:
    return kv_bytes_per_token_per_layer(itemsize) * cfg.NUM_HIDDEN_LAYERS


def kv_cache_bytes(n_tokens: int, itemsize: int = 2, batch: int = 1) -> int:
    return kv_bytes_per_token_model(itemsize) * n_tokens * batch


def serveable_tokens(total_mem_gb: float, weights_gb: float, reserve_gb: float,
                     itemsize: int = 2, block_size: int | None = None,
                     batch: int = 1) -> dict[str, Any]:
    """可服务容量核算：显存预算 → KV Token 容量（块粒度取整）。

    batch 语义：容量按"并发 batch 个请求平均各占 len"消耗，这里返回的是
    全局 Token 预算（各请求共享池），并给出 batch 均分口径的每请求长度。
    """
    budget_bytes = (total_mem_gb - weights_gb - reserve_gb) * 1e9
    per_token = kv_bytes_per_token_model(itemsize)
    tokens = int(budget_bytes // per_token)
    out: dict[str, Any] = {
        "total_mem_gb": total_mem_gb, "weights_gb": weights_gb,
        "reserve_gb": reserve_gb, "kv_budget_gb": round(budget_bytes / 1e9, 2),
        "kv_bytes_per_token": per_token,
        "serveable_tokens": tokens,
        "serveable_tokens_million": round(tokens / 1e6, 3),
        "batch1_avg_context_tokens": tokens // batch,
    }
    if block_size is not None:
        blocks = tokens // block_size
        out["serveable_blocks"] = blocks
        out["serveable_tokens_block_granular"] = blocks * block_size
        out["granularity_waste_tokens"] = tokens - blocks * block_size
        out["block_table_bytes"] = blocks * 4  # 每块一个 int32 表项的量级
    return out


def reservation_waste_gb(batch: int, max_ctx: int, mean_len: int,
                         itemsize: int = 2) -> dict[str, Any]:
    """连续预留（pre-vLLM 式 worst-case reservation）的浪费记账。

    静态连续方案按 [batch, max_ctx] 预留；实际平均只用到 mean_len。
    """
    reserved = kv_cache_bytes(max_ctx, itemsize, batch)
    used = kv_cache_bytes(mean_len, itemsize, batch)
    return {
        "batch": batch, "max_ctx": max_ctx, "mean_len": mean_len,
        "reserved_gb": round(reserved / 1e9, 3),
        "used_gb": round(used / 1e9, 3),
        "waste_gb": round((reserved - used) / 1e9, 3),
        "utilization_pct": round(used / reserved * 100.0, 1),
    }


# ---------------------------------------------------------------------------
# AI 与 bound 判定（ch21 同式）
# ---------------------------------------------------------------------------

def arithmetic_intensity(flops: int, bytes_total: int) -> float:
    return flops / bytes_total


def ridge_point(peak_tf: float, peak_gbs: float) -> float:
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


def decode_profile(batch: int, n: int, dtype: str = "fp16",
                   platform: str = "cuda", block_size: int = cfg.DEFAULT_BLOCK_SIZE,
                   table_mode: str = "seq") -> dict[str, Any]:
    """单实验点理论档案（decode 形态，布局感知）。"""
    itemsize = 2 if dtype == "fp16" else 4
    flops = decode_flops(batch, n)
    bu = decode_bytes_unique(batch, n, itemsize)
    bi = decode_bytes_issued(batch, n, itemsize)
    bu_paged = decode_bytes_paged_unique(batch, n, itemsize, block_size)
    return {
        "form": "decode", "batch": batch, "kv_len": n, "dtype": dtype,
        "layout": "paged", "table_mode": table_mode, "block_size": block_size,
        "flops": flops,
        "bytes_unique_contig": bu,
        "bytes_unique_paged": bu_paged,
        "block_table_bytes": block_table_bytes(batch, n, block_size),
        "bytes_issued": bi,
        "ai_unique": round(arithmetic_intensity(flops, bu), 4),
        "bounds_unique": bound_class(arithmetic_intensity(flops, bu), platform, dtype),
        "lower_bounds": lower_bounds_ms(flops, bu, platform, dtype),
        "kv_bytes_unique_mb": round(batch * cfg.HKV * n * cfg.D * 2 * itemsize / 1e6, 2),
        "block_granule_kb": round(block_granule_kb(block_size, itemsize), 1),
        "working_set_region": working_set_region(
            batch * cfg.HKV * n * cfg.D * 2 * itemsize, platform),
    }


if __name__ == "__main__":
    import json

    print(cfg.describe())
    print("KV/token/layer fp16 =", kv_bytes_per_token_per_layer(2), "B; "
          "model =", kv_bytes_per_token_model(2), "B")
    print("\ndecode profiles (cuda fp16, bs=128):")
    for n in cfg.BENCH_DECODE_KV_LENS:
        for b in cfg.BENCH_BATCHES:
            p = decode_profile(b, n, "fp16", "cuda")
            print(f"  N={n:>5} B={b:>2}: AI={p['ai_unique']:>7} "
                  f"{p['bounds_unique']['bound_class']:<19} "
                  f"t_mem={p['lower_bounds']['t_mem_ms']:.4f}ms "
                  f"kv={p['kv_bytes_unique_mb']:>7}MB "
                  f"table={p['block_table_bytes']/1024:.0f}KB "
                  f"granule={p['block_granule_kb']}KB {p['working_set_region']}")
    print("\n容量核算（fp16，权重 1.19GB，预留 2.0GB）:")
    for plat, peaks in cfg.PLATFORM_PEAKS.items():
        cap = serveable_tokens(peaks["total_mem_gb"], 1.19, 2.0,
                               block_size=cfg.DEFAULT_BLOCK_SIZE)
        print(f"  {plat}: {json.dumps(cap, ensure_ascii=False)}")
