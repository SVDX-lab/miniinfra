"""第 24 期 MLA 参考实现（CPU / torch eager，真值锚点）。

两种 decode 形式（同一组合成权重、同一份 KV cache，输出必须数学等价——
大纲 §9 新结构算子正确性定义）：

未吸收形式（参考真值，multi-head 形式 GEMM 序列）：
    q_nope_h · k_nope_h(t) = q_nope_h · (c_kv(t) @ W_UK_h)     ← 每 Token 每 head 小 GEMM，物化 K
    out_h = Σ_t a_h(t) · (c_kv(t) @ W_UV_h)                    ← 每 Token 每 head 小 GEMM，物化 V
吸收形式（代数改写，缓存里只有 latent）：
    q_nope_h · (c_kv(t) @ W_UK_h) = (q_nope_h @ W_UK_h^T) · c_kv(t)
        → W_UK 吸收进 Q 侧：每 Query 一次小 GEMM，与 N 无关
    Σ_t a_h(t) · (c_kv(t) @ W_UV_h) = (Σ_t a_h(t) · c_kv(t)) @ W_UV_h
        → W_UV 吸收进输出侧：V 投影消失，只对 latent 做加权求和

RoPE 段（两形式同构、都不能吸收，实验 4 数值佐证）：
    score_rope(h,t) = (R_{p_q} q_rope_raw_h) · (R_t k_rope_raw(t))
    - q 侧旋转 R_{p_q} 是每步一个固定正交阵 → 可折进 Q 投影（实验验证机器精度一致）；
    - k 侧旋转 R_t 逐 Token 不同（夹在权重与缓存之间）→ 不存在单一矩阵 M 使
      q^T M^T k_raw(t) 对所有 t 成立（数值实验：用 t_ref 处的 M 冒充，t=t_ref 处
      误差恰为 0、随 |t−t_ref| 旋转角差增大而增大）；
    - k_rope_raw 从 x 直接投影（不过 512 维 latent 瓶颈）→ latent 不含 rope 信息，
      64 维必须单独存（最小二乘实验佐证）。

RoPE 约定：Llama/HF 半分约定（x1=x[..,:32], x2=x[..,32:]），freqs_i =
theta^(-2i/64)。q/k 两侧同约定 → 相对位置性质 R_a^T R_b = R_{b-a} 成立（smoke
测试断言）。softmax_scale = 1/sqrt(192)（官方 modeling_deepseek.py L745
q_head_dim**-0.5；两种形式共用同一 scale，等价性结论与 scale 取值无关）。
"""

from __future__ import annotations

from typing import Any

import torch

from mla_config import (
    KV_LORA_RANK,
    N_HEADS,
    O_IN,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    Q_LORA_RANK,
    ROPE_THETA,
    SOFTMAX_SCALE,
)

ROPE_HALF = QK_ROPE_HEAD_DIM // 2  # 32


# ---------------------------------------------------------------------------
# RoPE 工具
# ---------------------------------------------------------------------------

def rope_freqs(theta: float = ROPE_THETA) -> torch.Tensor:
    """[32]：theta^(-2i/64)，i=0..31（dim=64 的标准 RoPE 频率）。"""
    return theta ** (-torch.arange(0, QK_ROPE_HEAD_DIM, 2, dtype=torch.float32) / QK_ROPE_HEAD_DIM)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """对最后一维 64 做 RoPE（Llama/HF 半分约定），返回与 x 同形。

    x: [..., 64]。positions: [P]：
      - P == 1：单位置（query 侧），cos/sin 广播到 x 的前导维；
      - P > 1 且 x.shape[0] == P：逐位置（k_rope_raw [N,64]）。
    等价于左乘块对角正交阵 R_pos（(i, i+32) 对 Givens 旋转）。
    """
    # 角度精度跟随 freqs（fp64 实验 → fp64 角度；fp16 路径调用方传 fp32 freqs
    # → 角度不受 fp16 量化），cos/sin 乘法前落到激活 dtype（io dtype 语义）。
    angles = positions.to(freqs.dtype).unsqueeze(-1) * freqs.unsqueeze(0)  # [P,32]
    cos, sin = angles.cos(), angles.sin()
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    if positions.numel() == 1:
        cos = cos.reshape(*([1] * (x.dim() - 1)), ROPE_HALF)  # 广播
        sin = sin.reshape(*([1] * (x.dim() - 1)), ROPE_HALF)
    x1 = x[..., :ROPE_HALF]
    x2 = x[..., ROPE_HALF:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def rope_matrix(position: float, freqs: torch.Tensor) -> torch.Tensor:
    """显式 [64,64] 块对角正交阵 R_pos（RoPE 吸收实验用，与 apply_rope 同约定）。
    dtype 跟随 freqs（FP64 实验可复用）。"""
    angles = float(position) * freqs
    cos, sin = angles.cos(), angles.sin()
    m = torch.zeros(QK_ROPE_HEAD_DIM, QK_ROPE_HEAD_DIM, dtype=freqs.dtype)
    for i in range(ROPE_HALF):
        m[i, i] = cos[i]
        m[i, i + ROPE_HALF] = -sin[i]
        m[i + ROPE_HALF, i] = sin[i]
        m[i + ROPE_HALF, i + ROPE_HALF] = cos[i]
    return m


# ---------------------------------------------------------------------------
# 权重容器与投影阶段（两形式共用）
# ---------------------------------------------------------------------------

WEIGHT_KEYS = ("W_QA", "W_QB", "W_KVA", "W_UK", "W_UV", "W_O")


def split_q(q_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q_b_proj 输出 [1, 3072] → q_nope [H,128], q_rope_raw [H,64]。

    官方排布：每头先 nope 后 rope（view(1,H,192) 切 [:128] 与 [128:]）。
    """
    q = q_all.view(1, N_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
    return q[0, :, :QK_NOPE_HEAD_DIM], q[0, :, QK_NOPE_HEAD_DIM:]


def split_kv_a(kv_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """kv_a_proj_with_mqa 输出 [1, 576] → (c_kv_raw [1,512], k_rope_raw [1,64])。"""
    return kv_a[..., :KV_LORA_RANK], kv_a[..., KV_LORA_RANK:]


def q_projection_stage(x: torch.Tensor, w: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Q 路径（两形式共用）：x → q_a → q_b → (q_nope [H,128], q_rope_raw [H,64])。"""
    q_a = x @ w["W_QA"]
    return split_q(q_a @ w["W_QB"])


def kv_new_token_stage(x: torch.Tensor, w: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """新 Token 的 KV 路径（两形式共用）：x → kv_a → (c_new [1,512], k_rope_raw [1,64])。

    真实实现里 c 还要过 kv_a_layernorm、q_a 过 q_a_layernorm；本期合成权重
    省略 LayerNorm（两种形式同构省略，不影响等价性），结构差异已在
    experiment-contract.yaml 与 EXPERIMENT_LOG 显式声明。
    """
    return split_kv_a(x @ w["W_KVA"])


def rotate_q_rope(q_rope_raw: torch.Tensor, pos_q: int, freqs: torch.Tensor) -> torch.Tensor:
    """query 侧 RoPE：R_{p_q} q_rope_raw → [H,64]。positions 与 freqs 同设备。"""
    positions = torch.tensor([pos_q], dtype=torch.float32, device=freqs.device)
    return apply_rope(q_rope_raw, positions, freqs)


# ---------------------------------------------------------------------------
# attention 核（N 相关部分）——两种形式
# ---------------------------------------------------------------------------

def mla_core_unabsorbed(
    q_nope: torch.Tensor,   # [H, 128]
    q_rope: torch.Tensor,   # [H, 64]，已按 p_q 旋转
    c_kv: torch.Tensor,     # [N, 512]
    k_rope: torch.Tensor,   # [N, 64]，已按各自位置旋转
    w: dict[str, torch.Tensor],
    scale: float = SOFTMAX_SCALE,
    want_probes: bool = False,
) -> dict[str, torch.Tensor]:
    """未吸收核：per-head 小 GEMM 物化 K/V（每 Token 一遍，N 相关）。

    教学写法：einsum 显式表达"每 head 一个 [N,512]×[512,128] 小 GEMM"。
    """
    k_nope = torch.einsum("nd,hdo->hno", c_kv, w["W_UK"])       # [H,N,128] 物化 K
    scores_nope = torch.einsum("hd,hnd->hn", q_nope, k_nope)     # [H,N]
    scores_rope = torch.einsum("hd,nd->hn", q_rope, k_rope)      # [H,N] rope 段（共享 k_rope）
    scores = (scores_nope + scores_rope) * scale
    attn = torch.softmax(scores, dim=-1)                         # [H,N]
    v = torch.einsum("nd,hdo->hno", c_kv, w["W_UV"])            # [H,N,128] 物化 V
    out_h = torch.einsum("hn,hno->ho", attn, v)                  # [H,128]
    result: dict[str, torch.Tensor] = {"out_h": out_h, "scores": scores, "attn": attn}
    if want_probes:
        result.update({"k_nope": k_nope, "v": v})
    return result


def mla_core_absorbed(
    q_nope: torch.Tensor,   # [H, 128]
    q_rope: torch.Tensor,   # [H, 64]，已按 p_q 旋转
    c_kv: torch.Tensor,     # [N, 512]
    k_rope: torch.Tensor,   # [N, 64]
    w: dict[str, torch.Tensor],
    scale: float = SOFTMAX_SCALE,
    want_probes: bool = False,
) -> dict[str, torch.Tensor]:
    """吸收核：W_UK 吸收进 Q 侧、W_UV 吸收进输出侧（缓存只读 latent，无 K/V 物化）。

    改写 1：q_abs_h = q_nope_h @ W_UK_h^T → [H,512]，score_nope = q_abs·c_kv(t)
    改写 2：s_h = Σ_t a_h(t)·c_kv(t) → [H,512]，out_h = s_h @ W_UV_h
    """
    q_abs = torch.einsum("hd,hcd->hc", q_nope, w["W_UK"])        # [H,512] 吸收 W_UK（每 Query 一遍）
    scores_nope = torch.einsum("ho,no->hn", q_abs, c_kv)         # [H,N] 直接对 latent 打分
    scores_rope = torch.einsum("hd,nd->hn", q_rope, k_rope)      # [H,N] rope 段与未吸收形式逐位相同
    scores = (scores_nope + scores_rope) * scale
    attn = torch.softmax(scores, dim=-1)
    s = torch.einsum("hn,nd->hd", attn, c_kv)                    # [H,512] 加权 latent
    out_h = torch.einsum("hc,hcd->hd", s, w["W_UV"])             # [H,128] 吸收 W_UV（每 Query 一遍）
    result: dict[str, torch.Tensor] = {"out_h": out_h, "scores": scores, "attn": attn}
    if want_probes:
        result.update({"q_abs": q_abs, "s": s})
    return result


# ---------------------------------------------------------------------------
# 完整 decode 步（投影 + 核 + 输出投影）——benchmark 与等价验证的主对象
# ---------------------------------------------------------------------------

def _decode(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    pos_q: int,
    freqs: torch.Tensor,
    scale: float,
    core_fn,
    want_probes: bool,
) -> dict[str, torch.Tensor]:
    q_nope, q_rope_raw = q_projection_stage(x, w)
    q_rope = rotate_q_rope(q_rope_raw, pos_q, freqs)
    core = core_fn(q_nope, q_rope, c_kv, k_rope, w, scale, want_probes)
    out = core["out_h"].reshape(1, O_IN) @ w["W_O"]              # [1, 2048]
    result = {"out": out, **core}
    if want_probes:
        result.update({"q_nope": q_nope, "q_rope": q_rope, "q_rope_raw": q_rope_raw})
    return result


def mla_decode_unabsorbed(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    pos_q: int,
    freqs: torch.Tensor,
    scale: float = SOFTMAX_SCALE,
    want_probes: bool = False,
) -> dict[str, torch.Tensor]:
    """未吸收完整 decode 步 → {"out": [1,2048], scores, attn, [k_nope, v]}。"""
    return _decode(x, w, c_kv, k_rope, pos_q, freqs, scale, mla_core_unabsorbed, want_probes)


def mla_decode_absorbed(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    pos_q: int,
    freqs: torch.Tensor,
    scale: float = SOFTMAX_SCALE,
    want_probes: bool = False,
) -> dict[str, torch.Tensor]:
    """吸收完整 decode 步 → {"out": [1,2048], scores, attn, [q_abs, s]}。"""
    return _decode(x, w, c_kv, k_rope, pos_q, freqs, scale, mla_core_absorbed, want_probes)


# ---------------------------------------------------------------------------
# 实验 4：RoPE 段吸收尝试（数值佐证）
# ---------------------------------------------------------------------------

def rope_segment_scores_correct(
    q_rope_raw: torch.Tensor,   # [H,64] 未旋转
    k_rope_raw: torch.Tensor,   # [N,64] 未旋转（假设缓存存"原始 rope 投影"）
    pos_q: int,
    positions: torch.Tensor,    # [N]
    freqs: torch.Tensor,
) -> torch.Tensor:
    """正确 rope 分数：score(h,t) = (R_{p_q} q_h)·(R_t k(t)) → [H,N]。"""
    q_rot = rotate_q_rope(q_rope_raw, pos_q, freqs)
    k_rot = apply_rope(k_rope_raw, positions, freqs)
    return torch.einsum("hd,nd->hn", q_rot, k_rot)


def rope_fold_q_side_weight(w: dict[str, torch.Tensor], pos_q: int, freqs: torch.Tensor) -> torch.Tensor:
    """把 R_{p_q} 折进 W_QB 的 rope 列（合法的 q 侧吸收）→ 返回修改后的 W_QB。

    代数：q_rot_h = R (x @ W_h) = x @ (W_h @ R^T) → W'_h = W_h @ R^T，
    rope 列块 [1536,H,64] 逐头右乘 R^T。投影输出即已旋转的 q_rope。
    """
    qlr = w["W_QB"].shape[0]
    rope_cols = w["W_QB"].view(qlr, N_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)[
        :, :, QK_NOPE_HEAD_DIM:
    ]  # [1536, H, 64]
    r_t = rope_matrix(pos_q, freqs).T
    folded = rope_cols @ r_t  # [1536, H, 64]
    w_qb = w["W_QB"].clone().view(qlr, N_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
    w_qb[:, :, QK_NOPE_HEAD_DIM:] = folded
    return w_qb.view(qlr, N_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM))


def rope_absorb_attempt(
    q_rope_raw: torch.Tensor,   # [H,64] 未旋转
    k_rope_raw: torch.Tensor,   # [N,64] 未旋转
    pos_q: int,
    t_ref: float,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """变体 c（不合法的"完全吸收"尝试）：假装 k 侧旋转与位置无关，用单一矩阵
    M = R_{p_q}^T R_{t_ref} 折进 Q 侧，缓存只存未旋转 k_rope_raw、decode 时
    不再逐 token 旋转：
        score_attempt(t) = (q_h @ M)·k_raw(t) = q_h^T R_{p_q}^T R_{t_ref} k_raw(t)
    与正确 score(t) = q_h^T R_{p_q}^T R_t k_raw(t) 相比：t = t_ref 处误差恰为 0，
    其余位置随旋转角差增大——"R_t 夹在权重与缓存之间、无法用单个固定矩阵吸收"
    的数值证据。→ [H,N]
    """
    m = rope_matrix(pos_q, freqs).T @ rope_matrix(t_ref, freqs)  # M = R_{p_q}^T R_{t_ref}
    q_folded = q_rope_raw @ m  # [H,64]
    return torch.einsum("hd,nd->hn", q_folded, k_rope_raw)


def rope_attempt_end_to_end(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    c_kv: torch.Tensor,
    k_rope_raw: torch.Tensor,
    pos_q: int,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    t_ref: float,
    scale: float = SOFTMAX_SCALE,
) -> dict[str, torch.Tensor]:
    """把变体 c 的 rope 分数代入完整 attention（nope 分数保持正确），
    输出与未吸收参考对照——量化"错误吸收"的端到端损伤。"""
    q_nope, q_rope_raw = q_projection_stage(x, w)
    scores_rope = rope_absorb_attempt(q_rope_raw, k_rope_raw, pos_q, t_ref, freqs)
    k_nope = torch.einsum("nd,hdo->hno", c_kv, w["W_UK"])
    scores_nope = torch.einsum("hd,hnd->hn", q_nope, k_nope)
    scores = (scores_nope + scores_rope) * scale
    attn = torch.softmax(scores, dim=-1)
    v = torch.einsum("nd,hdo->hno", c_kv, w["W_UV"])
    out_h = torch.einsum("hn,hno->ho", attn, v)
    out = out_h.reshape(1, O_IN) @ w["W_O"]
    return {"out": out, "scores": scores, "attn": attn}


def latent_to_rope_linear_map(c_kv: torch.Tensor, k_rope_raw: torch.Tensor) -> dict[str, float]:
    """最小二乘检验：是否存在 W' 使 k_rope_raw ≈ c_kv @ W'（"rope 能否从 latent
    恢复"）。真实 MLA 中 k_pe 从 x 直接投影、不过 latent 瓶颈——若真存在这样的
    映射，在 c 上拟合出的 W' 应能泛化。

    统计口径：features=512，必须 train/test 分半报告——样本内 R² 在 p/n 比例大时
    有偏（N=2048、p=512 的零假设下样本内 R² ≈ p/n = 0.25），不能作为判据；
    held-out R² 才是"存在共享线性映射"的检验。合成独立随机数据下 held-out R²
    应≈0，佐证 64 维 rope 必须单独存。"""
    n = c_kv.shape[0]
    train_idx = torch.arange(0, n, 2)
    test_idx = torch.arange(1, n, 2)
    solution = torch.linalg.lstsq(c_kv[train_idx], k_rope_raw[train_idx]).solution  # [512,64]
    residual_train = c_kv[train_idx] @ solution - k_rope_raw[train_idx]

    def r2(res: torch.Tensor, target: torch.Tensor) -> float:
        var_total = ((target - target.mean(dim=0)) ** 2).sum()
        return 1.0 - float((res ** 2).sum() / var_total)

    residual_test = c_kv[test_idx] @ solution - k_rope_raw[test_idx]
    return {
        "in_sample_r_squared": r2(residual_train, k_rope_raw[train_idx]),
        "held_out_r_squared": r2(residual_test, k_rope_raw[test_idx]),
        "held_out_rel_residual": float((residual_test ** 2).sum().sqrt() / k_rope_raw[test_idx].norm()),
        "n_train": int(train_idx.numel()),
        "n_test": int(test_idx.numel()),
        "note": "样本内 R² 受 p/n 偏置（零假设下≈p/n），判据看 held_out_r_squared",
    }


# ---------------------------------------------------------------------------
# bmm/matmul 形式（benchmark 与 CUDA 对照链用的算子序列；与 einsum 教学形式
# 数学等价，compare_correctness 里先做一次 CPU FP32 互证）
# ---------------------------------------------------------------------------

def mla_core_unabsorbed_bmm(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    w: dict[str, torch.Tensor],
    scale: float = SOFTMAX_SCALE,
) -> dict[str, torch.Tensor]:
    """未吸收核的 matmul/bmm 序列（K 物化 [H,N,128]、V 物化 [H,N,128]）。"""
    k_nope = torch.matmul(c_kv, w["W_UK"])                          # [H,N,128]（c_kv 广播 batch）
    scores_nope = torch.matmul(q_nope.unsqueeze(1), k_nope.transpose(1, 2))  # [H,1,N]
    scores_rope = q_rope @ k_rope.t()                               # [H,N]
    scores = (scores_nope + scores_rope.unsqueeze(1)) * scale
    attn = torch.softmax(scores, dim=-1)                            # [H,1,N]
    v = torch.matmul(c_kv, w["W_UV"])                               # [H,N,128]
    out_h = torch.matmul(attn, v).squeeze(1)                        # [H,128]
    return {"out_h": out_h, "scores": scores.squeeze(1), "attn": attn.squeeze(1),
            "k_nope": k_nope, "v": v}


def mla_core_absorbed_bmm(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    c_kv: torch.Tensor,
    k_rope: torch.Tensor,
    w: dict[str, torch.Tensor],
    scale: float = SOFTMAX_SCALE,
) -> dict[str, torch.Tensor]:
    """吸收核的 matmul/bmm 序列（无 K/V 物化；CUDA 核对照的 torch 侧基准）。

    q_abs = q_nope @ W_UK^T（吸收 1）；s = attn @ c_kv 后 out_h = s @ W_UV（吸收 2）。
    """
    q_abs = torch.matmul(q_nope.unsqueeze(1), w["W_UK"].transpose(1, 2))  # [H,1,512]
    scores_nope = torch.matmul(q_abs, c_kv.t().unsqueeze(0))              # [H,1,N]（c_kv 广播 batch）
    scores_rope = q_rope @ k_rope.t()                                     # [H,N]
    scores = (scores_nope + scores_rope.unsqueeze(1)) * scale
    attn = torch.softmax(scores, dim=-1)                                  # [H,1,N]
    s = torch.matmul(attn, c_kv)                                          # [H,1,512]
    out_h = torch.matmul(s, w["W_UV"]).squeeze(1)                         # [H,128]
    return {"out_h": out_h, "scores": scores.squeeze(1), "attn": attn.squeeze(1),
            "q_abs": q_abs.squeeze(1), "s": s.squeeze(1)}


def _decode_bmm(x, w, c_kv, k_rope, pos_q, freqs, scale, core_fn):
    q_nope, q_rope_raw = q_projection_stage(x, w)
    q_rope = rotate_q_rope(q_rope_raw, pos_q, freqs)
    core = core_fn(q_nope, q_rope, c_kv, k_rope, w, scale)
    out = core["out_h"].reshape(1, O_IN) @ w["W_O"]
    return {"out": out, **core}


def mla_decode_unabsorbed_bmm(x, w, c_kv, k_rope, pos_q, freqs,
                              scale: float = SOFTMAX_SCALE) -> dict[str, torch.Tensor]:
    """未吸收完整 decode 步（matmul/bmm 算子序列，CPU/CUDA benchmark 共用）。"""
    return _decode_bmm(x, w, c_kv, k_rope, pos_q, freqs, scale, mla_core_unabsorbed_bmm)


def mla_decode_absorbed_bmm(x, w, c_kv, k_rope, pos_q, freqs,
                            scale: float = SOFTMAX_SCALE) -> dict[str, torch.Tensor]:
    """吸收完整 decode 步（matmul/bmm 算子序列，CPU/CUDA benchmark 共用）。"""
    return _decode_bmm(x, w, c_kv, k_rope, pos_q, freqs, scale, mla_core_absorbed_bmm)


# ---------------------------------------------------------------------------
# 投影合并变体（吸收矩阵进一步折进 Q 投影的极限形式）
# ---------------------------------------------------------------------------

def merged_q_projection_weight(w: dict[str, torch.Tensor]) -> torch.Tensor:
    """把 W_UK 直接折进 W_QB 的 nope 列 → [1536, H×512]。

    代数：q_abs_h = q_nope_h @ W_UK_h^T，q_nope_h = x @ W_nope_h
      → q_abs_h = x @ (W_nope_h @ W_UK_h^T)，合并列块 merged_h[l,c] =
        Σ_d W_nope_h[l,d] · W_UK_h[c,d]。
    验证"投影合并"与"两步吸收"等价（机器精度），benchmark 主链路仍用
    两步形式（教学对应"吸收矩阵"），合并形式作为延伸对照。
    """
    qlr = w["W_QB"].shape[0]
    q_nope_cols = w["W_QB"].view(qlr, N_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)[
        :, :, :QK_NOPE_HEAD_DIM
    ]  # [1536, H, 128]
    return torch.einsum("lhd,hcd->lhc", q_nope_cols, w["W_UK"]).reshape(
        qlr, N_HEADS * KV_LORA_RANK
    )
