"""第 22 期维度与 RoPE 口径单源。

RoPE 参数出处（2026-11-21T05:52Z S4000 实机读取）：
chapter_01/models/qwen3-0.6b/config.json（sha256 前缀 660db3b7，与 ch01
models/MANIFEST.json 记录一致）：
    rope_theta = 1000000
    head_dim = 128
    rope_scaling = None          → 无 partial rotary，整头旋转（0..127 全维）
    num_attention_heads = 16     (HQ)
    num_key_value_heads = 8      (HKV)
    hidden_size = 1024, num_hidden_layers = 28
    max_position_embeddings = 40960

⚠️ 口径更正：chapter_20/attention_config.py 曾记录 ROPE_THETA = 10000.0
（注释"本期不用"）；config.json 实际值 = 1000000。ch21 未使用该值，无结果
影响；本期以 config.json 实读值为准（KNOWN_ISSUES 有条目）。

HF 语义（chapter_01/qwen3_model.py RotaryEmbedding + rotate_half 语义，
RoPE-NeXt half-split 约定）：
    inv_freq[j] = 1 / theta ** (2j / head_dim)      j ∈ [0, head_dim/2)
    angle[pos, j] = pos * inv_freq[j]
    cos_dup[pos, :] = cos(angle)（两半重复）          HF 布局 [seq, head_dim]
    out[..., j]        = x[..., j] * cos - x[..., j+H] * sin
    out[..., j+H]      = x[..., j+H] * cos + x[..., j] * sin     （H = head_dim/2）

本期 cache 布局（kernel 与参考实现统一，compact）：cos/sin 各一张
[max_seq, head_dim/2]；HF duplicated 布局可无损展开（expand_dup 视图）。

形态定义：
    prefill  = 批量 Token：q [T, HQ, D]、k [T, HKV, D]，位置 seq_offset+i；
    decode   = 单 Token：q [HQ, D]、k [HKV, D]，位置 = seq_offset（cache_len），
               无批量轴（真实引擎的多序列 decode 批量需逐 Token 位置数组，
               归 out_of_scope，合同声明）；
    decode-cache-free = 单 Token 且不查表、现算 sincos（decode 形态的对照变体）。
"""

from __future__ import annotations

import struct

# ---- Qwen3-0.6B config.json 实读值（S4000 chapter_01/models，2026-11-21）----
ROPE_THETA = 1000000.0
HEAD_DIM = 128               # D
HALF_DIM = HEAD_DIM // 2     # 64：pair 数（compact cache 第二维）
NUM_ATTENTION_HEADS = 16     # HQ
NUM_KEY_VALUE_HEADS = 8      # HKV
HIDDEN_SIZE = 1024
NUM_HIDDEN_LAYERS = 28
MAX_POSITION_EMBEDDINGS = 40960

D = HEAD_DIM
HQ = NUM_ATTENTION_HEADS
HKV = NUM_KEY_VALUE_HEADS

# ---- 正确性矩阵（任务书：seq {1, 64, 512, 2048, 32768} × head_dim=128 ×
#      {fp32, fp16}，对 CPU FP64 锚点）----
CORRECTNESS_SEQ_LENS = [1, 64, 512, 2048, 32768]

# ---- 基准扫描 ----
BENCH_PREFILL_TOKENS = [128, 2048, 8192, 32768]
BENCH_DECODE_POSITIONS = [0, 2048, 32767]
BENCH_CACHE_SEQ = [2048, 32768, 131072]

# ---- 误差放大场景（cache dtype 实验）：theta × seq 二维 ----
CACHE_DTYPE_THETAS = [10000.0, 1000000.0, 100000000.0]
CACHE_DTYPE_SEQS = [512, 2048, 8192, 32768]

FIXTURES_SEED = 20261123

# 平台实测峰值带宽分母（ch02 实测，不重复测量；S4000 拷贝 711.28 GB/s）
PLATFORM_PEAKS = {
    "cuda": {"copy_gb_s_measured": 918.8, "l2_cache_mb": 72.0},
    "musa": {"copy_gb_s_measured": 711.28, "l2_cache_mb": 24.0},
}

# 本地 WSL2 的 CPU 拷贝带宽在 benchmark.py 现场实测（numpy 大数组 copy）作分母。


# ---- 服务器端 MUSA/CUDA standalone 二进制的数据生成 ----
# hash RNG（与 ch21 attention_config.mu_hash 同算法，uint32 溢出语义）
def mu_hash(idx: int, seed: int) -> int:
    x = (idx * 0x9E3779B9) & 0xFFFFFFFF
    x ^= seed
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def mu_value(idx: int, seed: int) -> float:
    return (mu_hash(idx, seed) % 20001) / 10000.0 - 1.0


SEED_Q = 0x23002301
SEED_K = 0x23002302
SEED_CACHE_CHECK = 0x23002303


def checksum_f64(values) -> float:
    """float64 顺序和（与二进制侧 double 顺序和对拍，容差 1e-9）。"""
    total = 0.0
    for v in values:
        total += float(v)
    return total


def f32_le_bytes(numpy_array) -> bytes:
    """fp32 little-endian 原始字节（bin 文件交换格式）。"""
    return numpy_array.astype("<f4").tobytes()


def read_f32_le(path):
    import numpy as np
    return np.fromfile(path, dtype="<f4")


def struct_pack_int(*values) -> bytes:
    return struct.pack(f"<{len(values)}q", *values)
