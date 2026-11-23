"""第 24 期 MLA 公开 config 维度与出处（合同口径单源）。

维度出处（2026-11-21 原文抓取核对，见 experiment-contract.yaml sources 节）：
  - KV 侧全部维度：deepseek-ai/DeepSeek-V2-Lite config.json（HuggingFace raw）：
        hidden_size=2048, num_attention_heads=16, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        num_hidden_layers=27, num_key_value_heads=16
  - q_lora_rank=1536：DeepSeek-V2-Lite config.json 中为 null（Lite 无 Q 低秩，
    q 路径直接 q_proj: 2048→16×192）；本任务书指定维度取 DeepSeek-V2（full）
    config.json 的 q_lora_rank=1536。本期按任务书取 1536 并在此显式声明差异，
    Q 侧形状因此是"V2-Lite 身体 + V2 的 q_lora_rank"的混合口径。
    该差异只影响 Q 侧投影 GEMM 形状，不影响本期主题（KV 压缩与吸收矩阵）：
    两种 MLA 形式的 KV 侧结构、吸收代数、KV 字节数与 q_lora_rank 无关。
  - softmax_scale：官方 modeling_deepseek.py L745：
        self.softmax_scale = self.q_head_dim ** (-0.5)
    其中 q_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192。
  - 规模声明（合同要求）：非完整 V3 128-head 规模。V3/DeepSeek-V2(full) 为
    128 头、hidden 7168/5120；MLA 的每 Token KV 字节与头数无关（见
    roofline_mla.kv_bytes_per_token），故 KV 压缩结论可直接外推到 V3，
    per-head 小 GEMM 的规模扩展系数另行标注（roofline_mla.v3_extrapolation）。

合成权重策略：全部 torch.randn / numpy PCG64 固定 seed（generate_fixtures.py），
标准正态分布。真实 MLA 权重有专门初始化与 LayerNorm，本实验只用随机权重验证
代数等价与访存形状——分布不影响等价性结论（线性代数恒等式），只影响误差的
绝对量级，manifest 里记录 seed 与逐张量 SHA256。
"""

from __future__ import annotations

# ---- DeepSeek-V2-Lite config.json（原始字段名与取值）----
D_MODEL = 2048            # hidden_size
N_HEADS = 16              # num_attention_heads
KV_LORA_RANK = 512        # kv_lora_rank（latent c_KV 维度）
QK_NOPE_HEAD_DIM = 128    # qk_nope_head_dim
QK_ROPE_HEAD_DIM = 64     # qk_rope_head_dim（RoPE 段，MQA 式全头共享）
V_HEAD_DIM = 128          # v_head_dim
NUM_HIDDEN_LAYERS = 27    # num_hidden_layers（KV 字节/层 × 27 用于模型级口径）

# ---- DeepSeek-V2 (full) config.json（任务书指定混合进来的唯一维度）----
Q_LORA_RANK = 1536        # q_lora_rank（V2-Lite 为 null，见模块 docstring）

# ---- 派生维度 ----
Q_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM      # 192，官方 q_head_dim
Q_B_OUT = N_HEADS * Q_HEAD_DIM                        # 3072 = 16×(128+64)
KV_A_OUT = KV_LORA_RANK + QK_ROPE_HEAD_DIM            # 576 = 512+64
O_IN = N_HEADS * V_HEAD_DIM                           # 2048 = 16×128
SOFTMAX_SCALE = Q_HEAD_DIM ** (-0.5)                  # 1/sqrt(192)，官方 L745
ROPE_THETA = 10000.0                                  # config.json rope_theta

# 合成权重形状表（键 = 权重名，值 = (shape, 出处/角色说明)）
WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "W_QA":  (D_MODEL, Q_LORA_RANK),        # q_a_proj
    "W_QB":  (Q_LORA_RANK, Q_B_OUT),        # q_b_proj（输出按 [H, nope|rope] 切）
    "W_KVA": (D_MODEL, KV_A_OUT),           # kv_a_proj_with_mqa（[0:512]=latent, [512:576]=rope）
    "W_UK":  (N_HEADS, KV_LORA_RANK, QK_NOPE_HEAD_DIM),   # kv_b_proj 的 K 半（per-head）
    "W_UV":  (N_HEADS, KV_LORA_RANK, V_HEAD_DIM),         # kv_b_proj 的 V 半（per-head）
    "W_O":   (O_IN, D_MODEL),               # o_proj
}

CONFIG_SOURCE = {
    "kv_side": "https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/config.json",
    "q_lora_rank": "https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json",
    "softmax_scale": "https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/modeling_deepseek.py (L745: q_head_dim**-0.5)",
    "note": "V2-Lite q_lora_rank=null（无 Q 低秩）；1536 取自 DeepSeek-V2 config，任务书指定，混合口径已声明",
    "fetched_at": "2026-11-21",
}


def describe() -> str:
    lines = [
        f"d_model={D_MODEL} n_heads={N_HEADS} q_lora_rank={Q_LORA_RANK} "
        f"kv_lora_rank={KV_LORA_RANK} qk_nope={QK_NOPE_HEAD_DIM} "
        f"qk_rope={QK_ROPE_HEAD_DIM} v_head={V_HEAD_DIM}",
        f"softmax_scale=1/sqrt({Q_HEAD_DIM})={SOFTMAX_SCALE:.6f}  rope_theta={ROPE_THETA}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    for name, shape in WEIGHT_SHAPES.items():
        print(f"  {name}: {shape}")
