"""第 22 期 CPU 参考实现：RoPE 公式直算（FP32/FP64 双锚点）。

三路径（与 rope_config 形态定义一致）：
  1. build_cache        —— cos/sin compact cache [max_seq, HALF]（fp64 锚点 /
                           fp32 / fp16 三档 dtype）；
  2. apply_prefill      —— 批量 Token 形态（查表）；
  3. apply_decode       —— 单 Token 形态（查表一行）；
  4. apply_decode_cache_free —— 单 Token 无表现算（decode 对照变体）。

数值口径分层（与正确性实验的误差归因一一对应）：
  fp64 锚点：cache 表 float64 计算 + float64 算术 —— 误差 ≈ 0（机器精度）；
  fp32 路径：cache 表 float32 计算 + float32 算术 —— 角度舍入主导；
  fp16 io  ：cache 表取 dtype 档，内部 float32 算术，输出按 io dtype 舍入
             （与 GPU 核语义一致：half 读入 → float 计算 → half 写出）。

HF 语义对照见 compare_correctness.py 的 vendored HF 路径（chapter_01
qwen3_model.py 语义的复刻，不 import chapter_01）。
"""

from __future__ import annotations

import numpy as np

import rope_config as cfg


# ---------------------------------------------------------------------------
# cache 表构建
# ---------------------------------------------------------------------------

def build_cache(max_seq: int, theta: float, head_dim: int, dtype: str):
    """compact cache：cos/sin 各 [max_seq, head_dim/2]。

    dtype ∈ {"float64", "float32", "float16"}。
    inv_freq 公式镜像 chapter_01 HF 语义：inv_freq[j] = 1 / theta**(2j/head_dim)
    （先幂后取倒数，不合并为 theta**(-x)——浮点不等价，逐路径记录）。
    float16 档语义 = float32 计算整表 + 存储时舍入到 fp16（与 GPU 核
    build_cache_kernel 的 __float2half 存储口径一致；fp16 档不做 fp16 角度
    算术——theta=1e6 超出 fp16 表示范围 [0, 65504]，fp16 角度路径必然 inf，
    本身就是实验结论之一）。
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim 必须为偶数，得到 {head_dim}")
    half = head_dim // 2
    compute_dtype = np.float64 if dtype == "float64" else np.float32
    j = np.arange(half, dtype=compute_dtype)
    exponent = (2.0 * j) / compute_dtype(head_dim)
    inv_freq = 1.0 / np.power(compute_dtype(theta), exponent)    # HF 公式镜像
    positions = np.arange(max_seq, dtype=compute_dtype)
    angles = positions[:, None] * inv_freq[None, :]              # [max_seq, half]
    cos = np.cos(angles)
    sin = np.sin(angles)
    if dtype == "float16":
        return cos.astype(np.float16), sin.astype(np.float16)
    return cos, sin


def expand_dup(cos_compact, sin_compact):
    """compact [S, H] → HF duplicated [S, 2H]（cos[pos, j]==cos[pos, j+H]）。"""
    return np.concatenate([cos_compact, cos_compact], axis=-1), np.concatenate(
        [sin_compact, sin_compact], axis=-1
    )


# ---------------------------------------------------------------------------
# 应用（查表路径）
# ---------------------------------------------------------------------------

def _rotate_pairs(x, cos_row, sin_row, io_dtype=None):
    """一个 [.., head_dim] 张量在行 cos_row/sin_row ([.., half]) 下的旋转。

    内部一律提升到 float32（或输入本身是 float64 时保持 float64）计算，
    io_dtype 给定时按其舍入输出（fp16 io 语义）。
    """
    work_dtype = np.float64 if x.dtype == np.float64 else np.float32
    xw = x.astype(work_dtype)
    cw = cos_row.astype(work_dtype)
    sw = sin_row.astype(work_dtype)
    half = xw.shape[-1] // 2
    x1 = xw[..., :half]
    x2 = xw[..., half:]
    out = np.empty_like(xw)
    out[..., :half] = x1 * cw - x2 * sw
    out[..., half:] = x2 * cw + x1 * sw
    if io_dtype is not None:
        out = out.astype(io_dtype)
    return out


def apply_prefill(q, k, cos_cache, sin_cache, seq_offset: int = 0, io_dtype=None):
    """prefill 批量形态：q [T, H_q, D]、k [T, H_kv, D]，位置 = seq_offset + i。

    cos_cache/sin_cache [max_seq, half]。返回 (q_out, k_out) 同形。
    """
    num_tokens = q.shape[0]
    rows = np.arange(seq_offset, seq_offset + num_tokens)
    cos_rows = cos_cache[rows]          # [T, half]
    sin_rows = sin_cache[rows]
    q_out = _rotate_pairs(q, cos_rows[:, None, :], sin_rows[:, None, :], io_dtype)
    k_out = _rotate_pairs(k, cos_rows[:, None, :], sin_rows[:, None, :], io_dtype)
    return q_out, k_out


def apply_decode(q, k, cos_cache, sin_cache, seq_offset: int, io_dtype=None):
    """decode 单 Token 形态：q [H_q, D]、k [H_kv, D]，位置 = seq_offset。

    与 prefill 的实现差异（形态差异的可见点）：只取 cache 的一行
    （cos_cache[seq_offset]），无 token 维循环。输出与输入同形。
    """
    cos_row = cos_cache[seq_offset]     # [half]，与 x1 [..., half] 直接广播
    sin_row = sin_cache[seq_offset]
    q_out = _rotate_pairs(q, cos_row, sin_row, io_dtype)
    k_out = _rotate_pairs(k, cos_row, sin_row, io_dtype)
    return q_out, k_out


# ---------------------------------------------------------------------------
# decode 无表变体（现算 sincos）
# ---------------------------------------------------------------------------

def apply_decode_cache_free(q, k, theta: float, head_dim: int, seq_offset: int,
                            io_dtype=None):
    """单 Token 不建表：angle = seq_offset * theta^(-j/H) 现算。

    引擎意义：decode 步若只为 1 个新位置，可完全跳过整表预生成；代价是
    每步重复 sincos（逐层×逐头重复，28 层下同一角度被算 24 次）。
    """
    half = head_dim // 2
    work_dtype = np.float64 if q.dtype == np.float64 else np.float32
    j = np.arange(half, dtype=work_dtype)
    inv_freq = 1.0 / np.power(work_dtype(theta), (2.0 * j) / work_dtype(head_dim))  # 与 build_cache 同式（先幂后倒数）
    angle = work_dtype(seq_offset) * inv_freq
    cos_row = np.cos(angle)
    sin_row = np.sin(angle)
    q_out = _rotate_pairs(q, cos_row, sin_row, io_dtype)
    k_out = _rotate_pairs(k, cos_row, sin_row, io_dtype)
    return q_out, k_out


# ---------------------------------------------------------------------------
# RoPE 数学性质（smoke/验收用）
# ---------------------------------------------------------------------------

def rotation_matrix(position: int, theta: float, head_dim: int) -> np.ndarray:
    """位置 p 的旋转矩阵 R(p) ∈ R^{D×D}（pair (i, i+H) 的 2×2 旋转块拼块）。"""
    half = head_dim // 2
    cos_c, sin_c = build_cache(position + 1, theta, head_dim, "float64")
    m = np.zeros((head_dim, head_dim))
    for i in range(half):
        c, s = cos_c[position, i], sin_c[position, i]
        m[i, i] = c
        m[i, i + half] = -s
        m[i + half, i] = s
        m[i + half, i + half] = c
    return m


def relative_position_property(theta: float, head_dim: int, positions=(3, 17, 100)):
    """<R(m)q, R(n)k> 只依赖 m-n（RoPE 的核心数学性质，数值验证）。"""
    rng = np.random.default_rng(cfg.FIXTURES_SEED)
    half = head_dim // 2
    q = rng.standard_normal(head_dim)
    k = rng.standard_normal(head_dim)
    ok = True
    for m in positions:
        for n in positions:
            rm = rotation_matrix(m, theta, head_dim)
            rn = rotation_matrix(n, theta, head_dim)
            d = rm.T @ rn                     # = R(-m)·R(n) = R(n-m)
            # R(a)^T = R(-a)：R(n-m) = m<n 时直接构造；m≥n 时为 R(m-n) 的转置
            dn = (rotation_matrix(n - m, theta, head_dim) if n >= m
                  else rotation_matrix(m - n, theta, head_dim).T)
            if not np.allclose(d, dn, atol=1e-10):
                ok = False
    return ok
