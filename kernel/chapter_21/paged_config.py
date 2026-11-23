"""第 21 期维度与分页 KV 块表语义（口径单源）。

维度出处：与第 20 期同一 Qwen3-0.6B 口径（HQ=16 / HKV=8 / head_dim=128 /
28 层；每 Token KV 字节 fp16 = 4096 B/层 = 112 KB/Token/模型，ep21 已校核）。
本期只关注 KV Cache 的**组织**：连续布局 vs 分页块表布局；Attention 数学
语义（GQA + causal 稳定 softmax，decode 单 query / prefill 整段两形态）沿用
chapter_20 参考实现，不做 QKV 投影与 RoPE。

==========================================================================
块表语义合同（大纲 §8"分页 KV 的块表语义"条目的显式化）——全部实现
（CPU 参考 / CUDA / MUSA / kernel_sim）共用本节定义，任何偏离即合同违例：
==========================================================================
1. 物理块池（per 层 per K 或 V，两个池同构）：
     k_pool / v_pool: [num_blocks, HKV, block_size, D]，dtype fp16 或 fp32。
     物理块 p 的 (kv 头 g, 块内偏移 o) 行 = pool[p, g, o, :]，D 维连续。
2. block_table: [B, max_blocks] int32。行 b 是请求 b 的逻辑块→物理块映射：
     逻辑块 tb ∈ [0, ceil(seq_len_b / block_size)) 有效；
     tb ≥ ceil(seq_len_b / block_size) 的表项为 padding，合同值 = -1，
     kernel / 参考实现不得读取。
   恒等块表（identity）：B=1 时 block_table[0, t] == t——分页布局与连续布局
     严格同构（gather 是恒等拷贝），是"分页 vs 连续"对照的等价点；B>1 时
     退化为"请求独占连续物理段"（段基址 = 前面请求块数和；物理块跨请求
     不得重叠——一块同一时刻只属一个请求）。
   空块（empty）：padding 表项（-1）或 seq_len_b == 0 的请求（整表无效）。
   部分块（partial）：ceil(seq_len_b/bs) 个块中最后一块的有效 token 数
     seq_len_b % bs（非零时）——"非整块 token"。
3. 逻辑→物理映射（token 级）：请求 b 的 token t（0-based）：
     tb = t // block_size;  ot = t % block_size;  p = block_table[b, tb];
     逻辑 K[b, g, t, :] == k_pool[p, g, ot, :]（V 同构）。
4. seq_lens: [B] int32，每请求上下文长度；decode 的 query 只 attends
     tokens [0, seq_len_b)。N=0 为合同违例（显式报错，不静默）。
5. 错误行为（显式报错，不静默回退）：
     表项 < -1 或 ≥ num_blocks      → 越界，报错；
     有效区间内出现 -1 或重复物理块  → 块表损坏（aliasing/空洞），报错；
     max_blocks < ceil(seq_len_b/bs) → 块表不够长，报错；
     block_size ∉ BLOCK_SIZES        → 报错（编译期核按 bs 运行期参数，
                                        D=128 编译期固定）。
6. 值透明性断言（本期正确性的核心）：块表只改变地址、不改变值与求和顺序
   ——同一逻辑 KV 下，paged kernel 与连续布局 kernel 的输出必须位级一致
   （fp32 片上 + 同一归约序；fp16 io 同样位级一致：舍入只发生在写回）。
   与 CPU FP32 参考的对照按容差分层（fp32 ≤ 5e-4、fp16 ≤ 2e-2，ep21 口径）。
==========================================================================

块表三种对照组（大纲 §9 行 483 正确性主对 + 任务书）：
  A 随机块表：把 [0, num_blocks) 随机置换后取前 ceil(N/bs) 个（seed 固定）；
  B 恒等块表：block_table[b,t] = t（等价连续布局）；
  C 边界块表：空块（-1 padding）/ 部分块（N % bs ∈ {1..bs-1}）/ 恰整块
    （N % bs == 0）/ bs=1 token 退化（smoke）。
"""

from __future__ import annotations

# ---- Qwen3-0.6B config.json（与 ch21 同源）----
HIDDEN_SIZE = 1024
NUM_HIDDEN_LAYERS = 28
NUM_ATTENTION_HEADS = 16     # HQ
NUM_KEY_VALUE_HEADS = 8      # HKV
HEAD_DIM = 128               # D

HQ = NUM_ATTENTION_HEADS
HKV = NUM_KEY_VALUE_HEADS
D = HEAD_DIM
GQA_GROUP = HQ // HKV        # 2
SOFTMAX_SCALE = HEAD_DIM ** (-0.5)

# ---- 分页参数（合同扫描域）----
BLOCK_SIZES = [16, 32, 64, 128]
DEFAULT_BLOCK_SIZE = 128     # = decode kernel 的 chunk 大小 BC（单块单 chunk 对齐点）
TABLE_PAD = -1               # padding 表项合同值

# ---- 正确性矩阵 ----
# decode: seq_len（上下文长度）× batch；块表对照组 A/B/C 全覆盖
CORRECTNESS_SEQ_LENS = [512, 2048, 8192]
CORRECTNESS_BATCHES = [1, 4]
CORRECTNESS_BLOCK_SIZES = [16, 128]      # 主矩阵块粒度两端；其余归 smoke
# 边界族（smoke + correctness 边界组）：部分块/恰整块/单块/单 token
BOUNDARY_SEQ_LENS = [1, 15, 16, 17, 127, 128, 129, 255, 256, 511]

# ---- 基准扫描（任务书：seq {512, 2048, 8192}；块表随机 vs 顺序两档）----
BENCH_DECODE_KV_LENS = [512, 2048, 8192]
BENCH_BATCHES = [1, 4, 16]               # B=16 为占用率归因层（256 blocks > 128 SM）
BENCH_TABLE_MODES = ["seq", "rand"]      # 顺序（恒等）vs 随机置换
BENCH_PREFILL_SEQ_LENS = [512, 2048]     # 可选 prefill paged 对照

# ---- 显存记账实验（任务书：32 请求 × 8~2048 随机到达/释放）----
FRAG_NUM_REQUESTS = 32
FRAG_LEN_MIN = 8
FRAG_LEN_MAX = 2048
FRAG_BLOCK_SIZE_SWEEP = BLOCK_SIZES
FRAG_SEED = 20261122

# ---- 平台实测峰值分母（引用 ch02 实测，口径单源；不重复测量）----
PLATFORM_PEAKS = {
    "cuda": {
        "gpu_name": "NVIDIA GeForce RTX 4090 24GB (SM 8.9 Ada)",
        "copy_gb_s_measured": 918.8,
        "fp16_gemm_tf_measured": 171.63,
        "fp32_gemm_tf_measured": 55.6,
        "l2_cache_mb": 72.0,
        "total_mem_gb": 24.0,
    },
    "musa": {
        "gpu_name": "MTT S4000 48GB",
        "copy_gb_s_measured": 711.28,
        "fp16_gemm_tf_measured": 64.55,
        "fp32_gemm_tf_measured": 12.53,
        "l2_cache_mb": 24.0,
        "total_mem_gb": 48.0,
    },
}

# ---- MUSA standalone 二进制的数据生成（hash RNG，与 ep21 同算法）----
MU_HASH_SEED_Q = 0x51002201
MU_HASH_SEED_K = 0x51002202
MU_HASH_SEED_V = 0x51002203
MU_HASH_SEED_T = 0x51002204   # 块表


def mu_hash(idx: int, seed: int) -> int:
    """32 位 splitmix-style hash（ep21 同算法，uint32 溢出语义）。"""
    x = ((idx & 0xFFFFFFFF) * 0x9E3779B9 & 0xFFFFFFFF) ^ (seed & 0xFFFFFFFF)
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def mu_value(idx: int, seed: int) -> float:
    return (mu_hash(idx, seed) % 20001 - 10000.0) / 10000.0


def blocks_for(seq_len: int, block_size: int) -> int:
    """ceil(seq_len / block_size)（块表有效长度；seq_len ≥ 1 合同）。"""
    if seq_len < 1:
        raise ValueError(f"seq_len 必须 ≥ 1（合同 N≥1），实际 {seq_len}")
    if block_size not in BLOCK_SIZES:
        raise ValueError(f"block_size 合同域 {BLOCK_SIZES}，实际 {block_size}")
    return (seq_len + block_size - 1) // block_size


def describe() -> str:
    return (f"Qwen3-0.6B: HQ={HQ} HKV={HKV} D={D} group={GQA_GROUP} "
            f"layers={NUM_HIDDEN_LAYERS} block_sizes={BLOCK_SIZES}")


if __name__ == "__main__":
    print(describe())
    print(f"kv bytes/token/layer fp16 = {2 * HKV * D * 2} B")
    print(f"kv bytes/token/model fp16 = {2 * NUM_HIDDEN_LAYERS * HKV * D * 2} B "
          f"= {2 * NUM_HIDDEN_LAYERS * HKV * D * 2 / 1024:.0f} KB")
