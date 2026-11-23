"""第 25 期 DSA 公开 config 维度与出处（合同口径单源）。

主 Attention 维度（与第 24 期完全一致，便于跨期对照——同一条 MLA absorbed
核在 ep26 里作为"稀疏 Attention 的主计算"复用）：
  - DeepSeek-V2-Lite config.json：hidden_size=2048, num_attention_heads=16,
    kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
    num_hidden_layers=27
  - q_lora_rank=1536：DeepSeek-V2 (full)/V3.2 config.json（V2-Lite 为 null，
    ep25 已声明同一混合口径，本期沿用）

索引器（Lightning Indexer）维度（2026-11-21 原文抓取核对，见
experiment-contract.yaml sources 节）：
  - DeepSeek-V3.2-Exp config.json（HuggingFace raw）：
        index_n_heads = 64
        index_head_dim = 128
        index_topk = 2048
    同一 config：q_lora_rank=1536（索引器 Q 投影的输入维度正是 q_lora_rank，
    见 HF transformers modeling_deepseek_v32.py Indexer.__init__：
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim)
    ——与本期 Q_LORA_RANK=1536 天然一致，无需混合口径）
  - HF transformers modeling_deepseek_v32.py（Indexer 类，逐行核对）：
        self.wk = nn.Linear(self.hidden_size, self.head_dim)   # 单共享 K 头
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads)
        self.softmax_scale = self.index_head_dim ** -0.5       # 1/sqrt(128)
    分数计算（forward，fp32）：
        scores = relu(scale · (q^I_h · k^I))                   # 先 scale 后 relu
        index_scores = Σ_h w_h · scores_h，w = weights_proj(...) × n_heads**-0.5
    top-k：token 级（非块级）torch.topk，k = min(index_topk, seq_len)，
    返回 int32 索引。
  - index K 独立缓存（vLLM 官方博客 2025-09-29："for each token, there is now
    another K cache"；SGLang Day-0 帖："a dedicated key & key_scale cache in the
    memory pool for token scoring"）——即官方实现为每个 Token 缓存索引器 K
    （HF 参考实现缓存 bf16 post-norm+post-rope 的 k；DeepSeek 生产 kernel 用
    fp8 + scale，HF docstring 明言参考实现跳过了 rotate_activation(Hadamard)
    与 fp8_index）。本实验按 Design B（索引 K 缓存 resident）计每步开销。

本期合成设定（无法从公开材料确证的实现细节，显式声明，不冒充官方口径）：
  1. index 路径省略 LayerNorm 与 RoPE（官方 k_norm + 位置相关 k 缓存的精确
     排布未逐行确证；省略只影响索引器得分数值，不影响"索引开销占比 / top-k
     选择 / 给定索引集稀疏 Attention 正确性"三个实验主题）。
  2. index K 的来源：官方 wk 输入是 hidden_size（7168）的当前层 hidden state；
     本实验的 cache 直接合成 latent（ep25 同款口径），故用合成投影
     W_KL: latent(512) → index(128) 生成 k_I，作为"索引 K 投影存在一次写侧
     成本"的代理（Design A 每步重算的纸面对比用它；官方重算需缓存 hidden
     state 14KB/Token，比 latent 更贵——我们的代理是重算成本的乐观下界）。
  3. 检索质量（索引器排名与主 Attention 相关性）来自训练；合成权重是随机的、
     不含这种相关。本实验用 planted relevance 显式注入受控检索质量
     （dsa_reference.planted_relevance / generate_cache，三档 z 分层），
     结论只覆盖"给定检索质量水平下 k—误差—延迟的形态"，不外推真实模型
     的检索质量。manifest/结果 JSON 里同时报告可达成的 recall/mass 统计。
  4. 头数规模：主 Attention 用 V2-Lite 16 头（ep25 同款），非 V3.2 的 128 头。
     索引器维度（64 头×128 维）按 V3.2 公开值。主 Attention 每 Token 每 head
     的 MAC 数与头数无关（latent MQA 打分），头数只线性放大两项 N 相关成本，
     索引/主计算 MAC 比不变——break-even 结论（k/N < 0.5）可外推到 128 头，
     绝对延迟不能。
"""

from __future__ import annotations

# ---- 主 Attention：DeepSeek-V2-Lite config.json（与 ep25 完全一致）----
D_MODEL = 2048            # hidden_size（V2-Lite）
N_HEADS = 16              # num_attention_heads（V2-Lite；V3.2 为 128，见声明 4）
KV_LORA_RANK = 512        # kv_lora_rank
QK_NOPE_HEAD_DIM = 128    # qk_nope_head_dim
QK_ROPE_HEAD_DIM = 64     # qk_rope_head_dim
V_HEAD_DIM = 128          # v_head_dim
NUM_HIDDEN_LAYERS = 27    # V2-Lite 层数（V3.2 为 61，绝对量外推时另注）

# ---- Q 侧（ep25 混合口径沿用；V3.2 config 同为 1536）----
Q_LORA_RANK = 1536        # q_lora_rank

# ---- Lightning Indexer：DeepSeek-V3.2-Exp config.json（公开值）----
INDEX_HEADS = 64          # index_n_heads
INDEX_HEAD_DIM = 128      # index_head_dim
INDEX_TOPK_DEFAULT = 2048  # index_topk（V3.2 部署默认值）

# ---- 派生维度 ----
Q_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM      # 192
Q_B_OUT = N_HEADS * Q_HEAD_DIM                        # 3072
O_IN = N_HEADS * V_HEAD_DIM                           # 2048
Q_I_OUT = INDEX_HEADS * INDEX_HEAD_DIM                # 8192
SOFTMAX_SCALE = Q_HEAD_DIM ** (-0.5)                  # 1/sqrt(192)（主 Attention，ep25 同源）
INDEX_SCALE = INDEX_HEAD_DIM ** (-0.5)                # 1/sqrt(128)（索引器，HF Indexer 同式）
INDEX_WEIGHT_SCALE = INDEX_HEADS ** (-0.5)            # w 乘 1/sqrt(64)（HF Indexer forward 同式）
ROPE_THETA = 10000.0

# ---- 扫描矩阵（任务书实验 2/3）----
TOPK_SWEEP = (512, 1024, 2048, 4096)        # 实验任务指定的 top-k 扫描
K_SWEEP = (256, 512, 1024, 2048, 4096)      # 三元关系曲线 k（实验 3）
N_SWEEP = (4096, 16384, 65536)              # 三元关系曲线 N（实验 3）
CALC_N = 131072                             # 附录 B 题 10 校核：N=128K
CALC_K = 2048                               # 附录 B 题 10 校核：k=2048

# ---- 正确性 N（fixtures；含 N=1 边界 / 非整块 k / k=N 恒等）----
KV_LENS = (1, 3, 33, 127, 128, 2048)

# ---- planted relevance 合成设定（dsa_config 层声明，机制在 dsa_reference）----
# 支撑集随 N 线性缩放：S(n) = max(256, round(n × RELEVANCE_FRACTION))
RELEVANCE_FRACTION = 0.025
# 三档 z 分层目标（强/中/弱，相对各自空间噪声的自校准 z 值；量级推导见
# EXPERIMENT_LOG §4：正确 fan_in 下 softmax 单位噪声 std≈1.0、N=65536 背景
# 最大值 ≈4.4；planted 等化方向使每头 z 精确一致。取 z 使背景 mass ≈ 2%、
# 三档 ≈ 34/31/34%，误差—k 曲线在 k≈S(n) 处出现膝点）。index 侧每档 +0.6
# （索引排名需把 relevance 排到全部背景之前，留并排噪声余量）。
Z_TIERS = (9.4, 8.2, 6.9)          # 主 Attention 分数：strong / medium / weak
Z_TIERS_INDEX = (10.0, 8.8, 7.5)   # 索引器分数：strong / medium / weak

# 合成权重形状表（键 = 权重名，值 = (shape, 角色/出处说明)）
WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    # 主 Attention（ep25 同款 6 张中的 5 张——W_KVA 不需要：ep26 decode 的
    # cache 是直接合成的，query 自身 latent 不进 cache，无新 Token append 段）
    "W_QA":  (D_MODEL, Q_LORA_RANK),                    # q_a_proj
    "W_QB":  (Q_LORA_RANK, Q_B_OUT),                    # q_b_proj（[H, nope|rope] 切分）
    "W_UK":  (N_HEADS, KV_LORA_RANK, QK_NOPE_HEAD_DIM),  # kv_b_proj 的 K 半（per-head）
    "W_UV":  (N_HEADS, KV_LORA_RANK, V_HEAD_DIM),        # kv_b_proj 的 V 半（per-head）
    "W_O":   (O_IN, D_MODEL),                           # o_proj
    # Lightning Indexer（HF modeling_deepseek_v32 Indexer 同构，输入维度按
    # V2-Lite 身体缩放：wk/weights_proj 输入官方是 hidden_size=7168，本期 2048）
    "W_QI":  (Q_LORA_RANK, Q_I_OUT),                    # wq_b：q_lora → 64×128 索引 Q
    "W_IW":  (D_MODEL, INDEX_HEADS),                    # weights_proj：hidden → 64 头权重
    "W_KL":  (KV_LORA_RANK, INDEX_HEAD_DIM),            # 合成设定 2：latent → 索引 K（128）
}

CONFIG_SOURCE = {
    "main_attention": "https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/config.json",
    "indexer_config": "https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json"
                      "（index_n_heads=64, index_head_dim=128, index_topk=2048）",
    "indexer_formula": "https://github.com/huggingface/transformers/blob/main/src/transformers/"
                       "models/deepseek_v32/modeling_deepseek_v32.py（Indexer 类：wq_b/wk/"
                       "weights_proj/softmax_scale=head_dim**-0.5/relu/权重×n_heads**-0.5/"
                       "token 级 topk min(index_topk, seq_len) int32）",
    "index_k_cache": "https://vllm.ai/blog/2025-09-29-deepseek-v3-2（'for each token, there "
                     "is now another K cache'）；SGLang Day-0 帖（dedicated key & key_scale "
                     "cache for token scoring）",
    "fetched_at": "2026-11-21",
}


def relevance_support(n: int) -> int:
    """planted relevance 支撑集大小（合成设定 3）：S(n) = max(256, round(n·0.025))。

    N=4096 → 256；N=16384 → 410；N=65536 → 1638。膝点预期 k* ≈ S(n)
    （EXPERIMENT_LOG 三元曲线校核）。"""
    return max(256, round(n * RELEVANCE_FRACTION))


def tier_sizes(n: int) -> tuple[int, int, int]:
    """三档（strong/medium/weak）token 数：6.25% / 18.75% / 75% of S(n)。"""
    s = relevance_support(n)
    strong = max(4, round(s * 0.0625))
    medium = max(8, round(s * 0.1875))
    weak = s - strong - medium
    return strong, medium, weak


def describe() -> str:
    return (
        f"d_model={D_MODEL} n_heads={N_HEADS} q_lora_rank={Q_LORA_RANK} "
        f"kv_lora_rank={KV_LORA_RANK} qk_nope={QK_NOPE_HEAD_DIM} qk_rope={QK_ROPE_HEAD_DIM} "
        f"v_head={V_HEAD_DIM} | index_heads={INDEX_HEADS} index_head_dim={INDEX_HEAD_DIM} "
        f"index_topk={INDEX_TOPK_DEFAULT} index_scale=1/sqrt({INDEX_HEAD_DIM}) "
        f"index_weight_scale=1/sqrt({INDEX_HEADS})"
    )


if __name__ == "__main__":
    print(describe())
    for name, shape in WEIGHT_SHAPES.items():
        print(f"  {name}: {shape}")
    for n in N_SWEEP:
        print(f"  N={n}: support={relevance_support(n)} tiers={tier_sizes(n)}")
