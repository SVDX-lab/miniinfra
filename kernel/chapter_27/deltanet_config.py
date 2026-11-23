"""第 27 期 Gated DeltaNet 线性注意力：合同常量单源。

公开维度出处（2026-11-21 抓取，experiment-contract.yaml 有完整 URL）：
  1. FLA flash-linear-attention main 分支 `fla/layers/gated_deltanet.py`
     GatedDeltaNet 默认参数：hidden_size=2048, expand_v=2, head_dim=256,
     num_heads=6（head_k_dim=256、head_v_dim=512）；衰减门控
     g_kernel = -exp(A_log) * softplus(g_raw + dt_bias)，A=uniform(0,16)，
     dt∈[0.001,0.1]（inverse-softplus 初始化 dt_bias）；beta=sigmoid(b_proj(x))。
  2. FLA `fla/ops/gated_delta_rule/naive.py`：naive_recurrent_gated_delta_rule
     （逐 Token 递推）与 naive_chunk_gated_delta_rule（chunk_size=64、
     scale=1/sqrt(K)）——本期两形式的算法同构来源。
  3. Qwen/Qwen3-Next-80B-A3B-Instruct config.json：hidden_size=2048、
     linear_num_key_heads=16、linear_num_value_heads=32、
     linear_key_head_dim=128、linear_value_head_dim=128、num_hidden_layers=48。
  4. Kimi Linear（arXiv 2510.26692）KDA：per-dim（通道级）衰减门控——
     本期 vector-g 变体的公开存在性出处；具体 per-dim 投影秩等参数未确证，
     按合成设定声明。

合成差异声明（无法确证处，见 experiment-contract.yaml synthetic_declarations）：
  - 主配置 H=HV=16（FLA 的 HV=H 简化）；Qwen3-Next 的 32 V 头 GVA 分组
    作为一行附加等价实验（q/k 按 2 头一组复制到 32 V 头）单独验证；
  - 省略 short conv（conv_size=4）、输出门控（use_gate 的 g_proj/RMSNormGated）、
    q/k L2 norm（use_qk_l2norm_in_kernel）——均不影响两形式等价性结论，
    它们是逐元素/逐头包装层，与递推核正交；
  - per-dim 门控的投影实现为合成低秩结构（KDA 论文未逐行公开实现）。
"""

from __future__ import annotations

# ---- 公开维度（Qwen3-Next 线性注意力层，qwen3next config.json） ----
D_MODEL = 2048              # hidden_size
NUM_KEY_HEADS = 16          # linear_num_key_heads
NUM_VALUE_HEADS = 16        # linear_num_value_heads=32 的 GVA 分组在 HV=H 简化下
                            # 收缩为 16；GVA 行单独实验展开为 32
HEAD_K_DIM = 128            # linear_key_head_dim
HEAD_V_DIM = 128            # linear_value_head_dim
NUM_LAYERS = 48             # num_hidden_layers（模型级外推用）
SCALE = HEAD_K_DIM ** -0.5  # FLA naive 默认 scale = 1/sqrt(K)

# ---- FLA GatedDeltaNet 门控初始化口径（fla/layers/gated_deltanet.py） ----
GATE_A_UNIFORM_HI = 16.0    # A = uniform(0, 16)，A_log = log(A)
GATE_DT_MIN = 0.001
GATE_DT_MAX = 0.1
GATE_DT_FLOOR = 1e-4

# ---- chunked 形式 ----
CHUNK_SIZE_DEFAULT = 64     # FLA naive_chunk_gated_delta_rule 默认
CHUNK_SWEEP = (16, 32, 64)

# ---- 正确性扫描 ----
CORRECTNESS_SEQ_LENS = (1, 3, 33, 127, 128, 2048)   # 含非整块 N=3/33/127
GATE_VARIANTS = ("scalar", "vector")                # per-head 标量 / per-K-dim 向量

# ---- 延迟对照（实验 3） ----
LATENCY_SEQ_LENS = (512, 2048)
LATENCY_EXTRA_CHUNKED_SEQ_LENS = (8192, 32768)      # 递推版 CPU 太慢只做 chunked
LATENCY_BATCH = 1

# ---- 状态访存曲线（实验 2） ----
MEMORY_SEQ_LENS = (512, 2048, 8192, 32768)

# ---- 对照系统（KV 字节口径沿用 ch22 capacity_accounting：Qwen3-0.6B GQA） ----
CH22_HKV = 8
CH22_D = 128
CH22_NUM_LAYERS = 28

# ---- fixtures ----
SEED_BASE = 20260328
FIXTURE_TOL_FP32 = 2e-4     # 跨 GEMM 舍入复核容差（ep26 同源）
FIXTURE_TOL_FP64 = 1e-10

# ---- 门控行为探针（实验 4） ----
GATE_PROBE_SEQ = 4096
CONST_DECAY_LOG_G = (-0.002, -0.01, -0.05, -0.2)  # α ∈ {0.998, 0.990, 0.951, 0.819}
RETRIEVE_DISTANCES = (16, 64, 256, 1024, 4095)

DTYPES_CORRECTNESS = ("float64", "float32", "float16")
