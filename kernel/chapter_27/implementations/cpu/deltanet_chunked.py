"""第 27 期：Gated DeltaNet 分块并行（chunked）形式。

算法与 FLA fla/ops/gated_delta_rule/naive.py naive_chunk_gated_delta_rule
逐行同构（2026-11-21 抓取），并推广到 per-K-通道向量门控（KDA 变体）。

记号（块内 C 行，G = 块内 cumsum(g)，g ≤ 0 ⇒ G 单调不增）：
  块内（可全并行，块间无依赖）：
    L[i,j]   = exp(G_i − G_j)（i≥j，含对角=1；先 tril 后 exp 保证指数 ≤ 0）
    M₁[i,j]  = β_i · Σ_k k_i[k]·L[i,j,k]·k_j[k]   （严格下三角）
    T        = (I + M₁)^{-1}                      # UT transform 前向替换
    w        = T · (βv)                           # 解出的块内写入 u（含 S_0 耦合）
    kd       = T · (kβ ⊙ e^G)                     # S_0 载入投影
  块间（串行 carry，深度 ceil(T/C)，每步一个 [C,K]×[K,V] 级矩阵乘）：
    v_prime  = kd @ S
    v_new    = w − v_prime
    o        = (q ⊙ e^G) @ S + tril(q@kᵀ ⊙ L) @ v_new
    S        = S ⊙ e^{G_last} + (k ⊙ e^{G_last−G})ᵀ @ v_new

数值安全：所有 exp 的参数都先用 tril 掩码（或直接由"G 单调不增"论证
≤ 0），标量与向量门控同一保证——fp16 也不会上溢；深衰减通道下溢为 0
属语义正确。向量门控的 L 是 [H,C,C,K] 逐块瞬态（~34MB @C=64,K=128），
不跨块堆叠；M/w/kd 堆叠为 [H,NC,C,*] 后 UT 与 carry 批量进行。

padding：右侧补零（g=0 无衰减、β=0 无写入、q=0 输出 0）——padding 行的
T 行恰为单位行（M₁ 行为 0），状态与输出不受污染（smoke 覆盖）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # chapter_27/

import deltanet_config as cfg  # noqa: E402


def _pad(n: int, *shape, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    return torch.zeros(n, *shape, dtype=dtype, device=device)


def _ut_solve(neg_lower: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
    """解 (I + M)X = payload 的前向替换（FLA UT transform 同构）。

    neg_lower [...,C,C] = −M₁（严格下三角）；payload [...,C,D]。
    串行深度 C−1 步，步内按 (NC, H) 批量并行；M ≪ N。
    """
    out = payload.clone()
    c = out.shape[-2]
    for i in range(1, c):
        # out[i] = payload[i] + Σ_{j<i} neg_lower[i,j]·out[j]
        # （X + M₁X = P 的前向替换；FLA 的 (A[i,:i,None]·A[:i,:i]).sum(-2)
        # 行间自乘形式与此等价，推导见 EXPERIMENT_LOG §1。行 j<i 在第 i 步
        # 已定稿且后续只读，原地写安全。）
        out[..., i, :] = out[..., i, :] + torch.einsum(
            "...ij,...jd->...id", neg_lower[..., i:i + 1, :i], out[..., :i, :]
        ).squeeze(-2)
    return out


def chunked_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_log: torch.Tensor,
    chunk_size: int = cfg.CHUNK_SIZE_DEFAULT,
    scale: float = cfg.SCALE,
    initial_state: torch.Tensor | None = None,
    return_all_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | list[torch.Tensor]]:
    """分块并行形式。q/k [T,H,K]，v [T,H,V]，beta [T,H]，g_log [T,H]|[T,H,K]。

    返回 (o [T,H,V], S_final [H,K,V])；return_all_states 时返回
    (o, [S_0, S_1, ..., S_NC] 共 NC+1 个块边界状态)。
    """
    t, h, dim_k = k.shape
    dim_v = v.shape[-1]
    dtype = q.dtype
    c = chunk_size

    state = (
        torch.zeros(h, dim_k, dim_v, dtype=dtype, device=q.device) if initial_state is None
        else initial_state.to(device=q.device, dtype=dtype).clone()
    )

    # ---- pad 到整块（补零语义见模块 docstring；pad 张量随输入设备——MUSA
    #      的 torch.cat 要求同设备，CUDA/CPU 同理）----
    pad = (c - (t % c)) % c
    if pad:
        dev = q.device
        q = torch.cat([q, _pad(pad, h, dim_k, dtype=dtype, device=dev)])
        k = torch.cat([k, _pad(pad, h, dim_k, dtype=dtype, device=dev)])
        v = torch.cat([v, _pad(pad, h, dim_v, dtype=dtype, device=dev)])
        beta = torch.cat([beta, _pad(pad, h, dtype=dtype, device=dev)])
        g_log = torch.cat([g_log, _pad(pad, *g_log.shape[1:], dtype=dtype, device=dev)])
    nc = (t + pad) // c

    q = (q * scale).permute(1, 0, 2).reshape(h, nc, c, dim_k)       # [H,NC,C,K]
    k_c = k.permute(1, 0, 2).reshape(h, nc, c, dim_k)
    v_c = (v * beta.unsqueeze(-1)).permute(1, 0, 2).reshape(h, nc, c, dim_v)  # βv
    k_beta = (k * beta.unsqueeze(-1)).permute(1, 0, 2).reshape(h, nc, c, dim_k)
    vector_gate = g_log.dim() == 3
    if not vector_gate:
        g_log = g_log.unsqueeze(-1)                                   # [T,H,1]

    # ---- 块内 cumsum 与安全指数 ----
    if vector_gate:
        g = g_log.permute(1, 0, 2).reshape(h, nc, c, dim_k).cumsum(dim=2)   # [H,NC,C,K]
        g_last = g[..., -1, :].unsqueeze(-2)                          # [H,NC,1,K]
    else:
        g = g_log.permute(1, 0, 2).reshape(h, nc, c, 1).cumsum(dim=2)       # [H,NC,C,1]
        g_last = g[..., -1, :].unsqueeze(-2)                          # [H,NC,1,1]
    exp_g = torch.exp(g)                                              # ≤ 1
    exp_g_last = torch.exp(g_last)                                    # ≤ 1
    # 掩码建在输入设备上（MUSA 的 masked_fill 要求 mask 与 self 同设备——
    # CUDA 接受 CPU mask，MUSA 显式报错；迁移差异，见台账 M-50）
    tril_c = torch.tril(torch.ones(c, c, dtype=torch.bool, device=q.device))
    triu_strict = torch.triu(torch.ones(c, c, dtype=torch.bool, device=q.device), diagonal=1)

    # ---- L 与 M₁（向量门控逐块构造瞬态，避免 [NC,H,C,C,K] 全量堆叠）----
    if vector_gate:
        m_neg_list = []
        for ci in range(nc):
            # diff[h,i,j,k] = G[h,ci,i,k] − G[h,ci,j,k]；tril 掩码（-inf→exp=0）
            # 保证指数 ≤ 0 且上三角严格为 0（torch.tril 作用在最后两维，对
            # [H,C,C,K] 的 (C,K) 不是 (C,C)，必须用掩码不能靠 tril）
            g_blk = g[:, ci]                                          # [H,C,K]
            diff = g_blk[:, :, None, :] - g_blk[:, None, :, :]        # [H,C,C,K]
            diff = diff.masked_fill(~tril_c.unsqueeze(-1), float("-inf"))
            l_blk = torch.exp(diff)                                   # [H,C,C,K]，上三角=0
            # M₁[h,i,j] = β_i Σ_k k_i·L·k_j
            m_blk = torch.einsum("hik,hijk,hjk->hij", k_beta[:, ci], l_blk, k_c[:, ci])
            m_neg_list.append(-m_blk.masked_fill(torch.triu(
                torch.ones(c, c, dtype=torch.bool), diagonal=0), 0.0))
        m_neg = torch.stack(m_neg_list, dim=1)                        # [H,NC,C,C]
    else:
        # 标量门控：批量 FLA 原式（tril→exp→tril，指数 ≤ 0）
        diff = g.squeeze(-1)[:, :, :, None] - g.squeeze(-1)[:, :, None, :]  # [H,NC,C,C]
        l_pair = torch.exp(diff.masked_fill(~tril_c, 0.0)).tril()     # ≤ 1
        m_pos = (k_beta @ k_c.transpose(-1, -2)) * l_pair             # β_i(k_i·k_j)e^{G_i−G_j}
        m_neg = -m_pos.masked_fill(triu_strict, 0.0)                  # 严格下三角取负

    # ---- UT 解两个 payload（批量 [H,NC,C,*]）----
    kd_payload = k_beta * exp_g                                       # kβ ⊙ e^G（≤1 安全）
    w = _ut_solve(m_neg, v_c)                                         # T·(βv)
    kd = _ut_solve(m_neg, kd_payload)                                 # T·(kβ⊙e^G)

    # ---- 跨块 carry（串行深度 NC）----
    outputs = torch.empty(h, nc, c, dim_v, dtype=dtype)
    states: list[torch.Tensor] = [state.clone()]
    q_pos = q * exp_g                                                 # q ⊙ e^G（输出 o_inter 用）
    for ci in range(nc):
        v_prime = torch.einsum("hck,hkv->hcv", kd[:, ci], state)      # [H,C,V]
        v_new = w[:, ci] - v_prime
        o_inter = torch.einsum("hck,hkv->hcv", q_pos[:, ci], state)
        if vector_gate:
            # 逐块 L 输出项（含对角）：attn[i,j] = Σ_k q_i·e^{G_i−G_j}·k_j
            # （注意：此处不能用 q⊙e^{G−G_last} 因式化——G−G_last ≥ 0 会上溢）
            g_blk = g[:, ci]
            diff = g_blk[:, :, None, :] - g_blk[:, None, :, :]
            diff = diff.masked_fill(~tril_c.unsqueeze(-1), float("-inf"))
            l_out = torch.exp(diff)
            attn = torch.einsum("hik,hijk,hjk->hij", q[:, ci], l_out, k_c[:, ci])
        else:
            # 标量门控：成对衰减 e^{G_i−G_j}（tril 后指数 ≤ 0，FLA 原式）
            attn = (q[:, ci] @ k_c[:, ci].transpose(-1, -2)) * l_pair[:, ci]
        outputs[:, ci] = o_inter + attn.masked_fill(triu_strict, 0.0) @ v_new
        # carry：S ⊙ e^{G_last} + (k ⊙ e^{G_last−G})ᵀ @ v_new
        if vector_gate:
            state = state * exp_g_last[:, ci].squeeze(-2).unsqueeze(-1)   # [H,K,1] 按 K 行衰减
            k_back = k_c[:, ci] * torch.exp(g_last[:, ci].squeeze(-2).unsqueeze(-2) - g[:, ci])
        else:
            state = state * exp_g_last[:, ci]                             # [H,1,1]
            k_back = k_c[:, ci] * torch.exp(g_last[:, ci] - g[:, ci])
        state = state + torch.einsum("hck,hcv->hkv", k_back, v_new)
        states.append(state.clone())

    out = outputs.reshape(h, -1, dim_v).permute(1, 0, 2)[:t]          # [T,H,V]
    if pad:
        pass  # padding 行输出已为 0，直接截断
    if return_all_states:
        return out, states
    return out, state


def serial_step_counts(t: int, chunk_size: int = cfg.CHUNK_SIZE_DEFAULT) -> dict[str, int]:
    """两形式的结构性串行深度（延迟实验的结构口径）。

    递推：T 步（每步 4 次小 [K,V] 级运算 + 1 次读出）；
    chunked：C−1 步 UT 批量替换 + NC 步块间 carry + NC 步块内构造/输出循环。
    """
    c = chunk_size
    nc = (t + c - 1) // c
    return {
        "recurrent_serial_steps": t,
        "chunked_serial_steps": (c - 1) + 2 * nc,
        "chunk_count": nc,
        "chunk_size": c,
    }
