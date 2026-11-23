"""第 20 期维度与 Kernel 组织参数（口径单源）。

维度出处：Qwen3-0.6B 官方 config.json（与 chapter_01/models、chapter_01/performance/
estimate_qwen3.py 的模型 truth 体系同一口径）：
  hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
  num_key_value_heads=8, head_dim=128, rope_theta=10000。
本期只实现 Attention 本体（Q/K/V 已在核外给定），不做 QKV 投影——
投影归 GEMM（第 17 期），RoPE 归第 22 期。

GQA 分组映射：q 头 h 使用 kv 头 h // (HQ/HKV) = h // 2（HuggingFace 标准
repeat_interleave 语义：kv 头 g 服务 q 头 {2g, 2g+1}）。

softmax_scale = head_dim ** -0.5（官方 SDPA 默认 1/sqrt(128)）。
该 scale 自带归一：score = q·k/sqrt(D) 在 q,k ~ N(0,1) 下 ~ N(0,1)，
softmax 非退化——第 24 期"N(0,1) 权重 → one-hot 退化"教训在 Attention 本体
上不需要 He 缩放（scale 已把分数压回 O(1)），fixtures 按此记录。

合成数据：固定 seed 的 Q/K/V 激活（N(0,1)，fp32 存 fixtures；kernel io fp16
时运行时降半精度）。服务器端 MUSA standalone 二进制用同一 hash-RNG 算法
（attention_config.mu_hash 系列函数）复现同分布数据，manifest 记录预期校验和。
"""

from __future__ import annotations

# ---- Qwen3-0.6B config.json（原始字段名取值）----
HIDDEN_SIZE = 1024
NUM_HIDDEN_LAYERS = 28
NUM_ATTENTION_HEADS = 16     # HQ
NUM_KEY_VALUE_HEADS = 8      # HKV
HEAD_DIM = 128               # D
ROPE_THETA = 10000.0         # 本期不用（RoPE 归第 22 期），记录口径完整性

# ---- 派生 ----
HQ = NUM_ATTENTION_HEADS
HKV = NUM_KEY_VALUE_HEADS
D = HEAD_DIM
GQA_GROUP = HQ // HKV        # 2：每个 kv 头服务的 q 头数
SOFTMAX_SCALE = HEAD_DIM ** (-0.5)

# ---- Kernel 组织参数（与 .cuh 编译期常量严格一致；kernel_sim 按此仿真）----
BR = 16    # prefill 每 block 的 query tile 高（rows per block）
BC = 128   # prefill key tile 宽 = decode chunk 大小 = block 线程数（每线程 1 列/1 维）
THREADS = 128

# 正确性矩阵（任务书：seq {64,512,2048} × batch {1,4}）+ 边界形状归 smoke
CORRECTNESS_SEQ_LENS = [64, 512, 2048]
CORRECTNESS_BATCHES = [1, 4]

# 基准扫描
BENCH_PREFILL_SEQ_LENS = [64, 512, 2048, 8192]
BENCH_DECODE_KV_LENS = [64, 512, 2048, 8192, 32768]
BENCH_BATCHES = [1, 4]
# 物化对照变体基准（S×S scores/attn 显存约束：2048²×64头×4batch fp16 = 537MB×2）
MATERIALIZED_SEQ_LENS = [512, 2048]

# 平台实测峰值分母（不重复测量，引用 ch02 实测；口径单源）
PLATFORM_PEAKS = {
    "cuda": {
        "gpu_name": "NVIDIA GeForce RTX 4090 24GB (SM 8.9 Ada)",
        "copy_gb_s_measured": 918.8,       # ch02 实测拷贝峰
        "fp16_gemm_tf_measured": 171.63,   # ch02 实测 FP16 cuBLAS（大形状）
        "fp32_gemm_tf_measured": 55.6,     # ch02 实测 FP32 cuBLAS（大形状）
        "l2_cache_mb": 72.0,               # ep03 实测
        "spec_copy_gb_s": 1008.0,
        "spec_fp16_tf": 165.2,
        "spec_fp32_tf": 82.6,
    },
    "musa": {
        "gpu_name": "MTT S4000 48GB",
        "copy_gb_s_measured": 711.28,
        "fp16_gemm_tf_measured": 64.55,
        "fp32_gemm_tf_measured": 12.53,
        "l2_cache_mb": 24.0,
        "spec_copy_gb_s": 768.0,
        "spec_fp16_tf": 100.0,
        "spec_fp32_tf": 25.0,
    },
}

# 服务器端 MUSA standalone 二进制的数据生成（hash RNG，numpy 侧同算法复现）
MU_HASH_SEED_Q = 0x51002101
MU_HASH_SEED_K = 0x51002102
MU_HASH_SEED_V = 0x51002103


def mu_hash(idx: int, seed: int) -> int:
    """32 位 splitmix-style hash（.mu 二进制与 numpy 复算共用，uint32 溢出语义）。

    C 侧：x = (uint32)idx * 0x9E3779B9u ^ (uint32)seed;  x ^= x>>16;
          x *= 0x85EBCA6Bu; x ^= x>>13; x *= 0xC2B2AE35u; x ^= x>>16;
    返回值经 (x % 20001 - 10000)/10000 映到 [-1, 1]。
    """
    x = ((idx & 0xFFFFFFFF) * 0x9E3779B9 & 0xFFFFFFFF) ^ (seed & 0xFFFFFFFF)
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def mu_value(idx: int, seed: int) -> float:
    return (mu_hash(idx, seed) % 20001 - 10000.0) / 10000.0


def describe() -> str:
    return (f"Qwen3-0.6B: HQ={HQ} HKV={HKV} D={D} group={GQA_GROUP} "
            f"layers={NUM_HIDDEN_LAYERS} scale={SOFTMAX_SCALE:.6f}")


if __name__ == "__main__":
    print(describe())
    print(f"kv bytes/token/layer fp16 = {2 * HKV * D * 2} B")
    print(f"kv bytes/token/model fp16 = {2 * NUM_HIDDEN_LAYERS * HKV * D * 2} B "
          f"= {2 * NUM_HIDDEN_LAYERS * HKV * D * 2 / 1024:.0f} KB")
