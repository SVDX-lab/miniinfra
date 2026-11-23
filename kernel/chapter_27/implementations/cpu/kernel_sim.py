#!/usr/bin/env python3
"""第 27 期 CUDA chunked kernel 算法仿真（本地、无 GPU）。

目的：在 4090 回填前，用 CPU 串行重放验证计划中的 CUDA 核分解的数据划分与
访存序（ch23 EMU / ch26 kernel_sim 同方法论——先仿真后上卡）。

计划中的 CUDA 核分解（见 implementations/cuda/deltanet_chunked_design.md）：
  Phase 1（核 K1，grid=(H, NC)，每块一 CTA）：块内 L/M₁/w/kd 构造，
    每 CTA 负责一个 (head, chunk)，SMEM 装 L [C,C]（标量门）或
    k_blk/L 通道切片（向量门控）；
  Phase 2（核 K2，grid=(H, NC)，串行 C−1 步）：UT 前向替换解 w/kd，
    SMEM 无跨步依赖（就地行更新）；
  Phase 3（核 K3，grid=(H)，串行 NC 步 carry）：单 CTA 按块序推进状态
    [K,V]，每步 einsum(k_back, v_new)——FP32 状态驻留 SMEM/寄存器。
仿真器按 (phase, block) 粒度重放并断言：
  - 每相位输出与 torch 参考实现逐块一致；
  - grid 覆盖：每 (head, chunk) 恰被处理一次；
  - 最终 (o, S) 与 chunked_gdn 位级一致（同 CPU 路径、同算子顺序）。

产出：控制台 PASS/FAIL 计数（0 失败为通过标准）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent                      # implementations/cpu
CHAPTER = HERE.parent.parent                                # chapter_27/
sys.path.insert(0, str(CHAPTER))
sys.path.insert(0, str(HERE))

import deltanet_chunked as chunked  # noqa: E402
import deltanet_config as cfg  # noqa: E402
import deltanet_reference as ref  # noqa: E402

FAIL = 0
PASS = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def sim_chunked(q, k, v, beta, g_log, chunk_size, initial_state=None):
    """按 CUDA 核分解逐相位重放（CPU 张量算子与 deltanet_chunked 相同，
    但按 (H, NC) 块粒度调度并记录覆盖断言）。"""
    t, h, dim_k = k.shape
    dim_v = v.shape[-1]
    dtype = q.dtype
    c = chunk_size
    vector_gate = g_log.dim() == 3
    gl = g_log if vector_gate else g_log.unsqueeze(-1)

    state = torch.zeros(h, dim_k, dim_v, dtype=dtype) if initial_state is None \
        else initial_state.to(dtype).clone()
    pad = (c - (t % c)) % c
    if pad:
        z = lambda *s: torch.zeros(pad, *s, dtype=dtype)
        q = torch.cat([q, z(h, dim_k)])
        k = torch.cat([k, z(h, dim_k)])
        v = torch.cat([v, z(h, dim_v)])
        beta = torch.cat([beta, z(h)])
        gl = torch.cat([gl, z(*gl.shape[1:])])
    nc = (t + pad) // c
    processed_blocks = set()

    # ---- Phase 1/2 模拟：核 K1/K2 每 (head, chunk) 一个 CTA ----
    m_neg_all = torch.zeros(h, nc, c, c, dtype=dtype)
    w_all = torch.zeros(h, nc, c, dim_v, dtype=dtype)
    kd_all = torch.zeros(h, nc, c, dim_k, dtype=dtype)
    tril_c = torch.tril(torch.ones(c, c, dtype=torch.bool))
    triu_strict = torch.triu(torch.ones(c, c, dtype=torch.bool), diagonal=1)
    for hh in range(h):                     # CUDA: blockIdx.y = head
        for ci in range(nc):                # CUDA: blockIdx.x = chunk
            assert (hh, ci) not in processed_blocks, "grid 覆盖重复"
            processed_blocks.add((hh, ci))
            sl = slice(ci * c, (ci + 1) * c)
            q_blk = q[sl, hh] * chunked.cfg.SCALE
            k_blk = k[sl, hh]
            v_blk = (v[sl, hh] * beta[sl, hh].unsqueeze(-1))
            kb_blk = (k[sl, hh] * beta[sl, hh].unsqueeze(-1))
            g_blk = gl[sl, hh].cumsum(dim=0)                     # SMEM 内块内 cumsum
            g_last = g_blk[-1:].expand_as(g_blk) if vector_gate else g_blk[-1:].expand_as(g_blk)
            if vector_gate:
                diff = g_blk[:, None, :] - g_blk[None, :, :]     # [C,C,K]
                diff = diff.masked_fill(~tril_c.unsqueeze(-1), float("-inf"))
                l_blk = torch.exp(diff)
                m_blk = torch.einsum("ik,ijk,jk->ij", kb_blk, l_blk, k_blk)
            else:
                diff = g_blk[:, 0][:, None] - g_blk[:, 0][None, :]
                l_blk = torch.exp(diff.masked_fill(~tril_c, 0.0)).tril()
                m_blk = (kb_blk @ k_blk.T) * l_blk
            m_neg_all[hh, ci] = -m_blk.masked_fill(triu_strict, 0.0)
            payload_kd = kb_blk * torch.exp(g_blk) if not vector_gate else kb_blk * torch.exp(g_blk)
            # Phase 2（K2）：UT 前向替换（每 CTA 内串行 C−1 步）
            w_all[hh, ci] = chunked._ut_solve(m_neg_all[hh, ci].unsqueeze(0), v_blk.unsqueeze(0))[0]
            kd_all[hh, ci] = chunked._ut_solve(m_neg_all[hh, ci].unsqueeze(0), payload_kd.unsqueeze(0))[0]
    check("grid 覆盖：H×NC 块各处理一次", len(processed_blocks) == h * nc,
          f"{len(processed_blocks)} == {h}×{nc}")

    # ---- Phase 3 模拟：核 K3 单 CTA per head，串行 NC 步 carry ----
    outputs = torch.zeros(h, nc, c, dim_v, dtype=dtype)
    q_ch = (q * chunked.cfg.SCALE)
    exp_g = torch.exp(gl.permute(1, 0, 2).reshape(h, nc, c, -1 if vector_gate else 1).cumsum(dim=2))
    for hh in range(h):
        for ci in range(nc):
            sl = slice(ci * c, (ci + 1) * c)
            kd_blk = kd_all[hh, ci]
            v_new = w_all[hh, ci] - kd_blk @ state[hh]
            # scalar: gl[sl,hh] 已是 [C,1]，cumsum 后直接广播到 [C,K]；
            # vector: [C,K] 逐通道
            q_pos = q_ch[sl, hh] * torch.exp(gl[sl, hh].cumsum(dim=0))
            o_inter = q_pos @ state[hh]
            g_blk = gl[sl, hh].cumsum(dim=0)
            if vector_gate:
                diff = g_blk[:, None, :] - g_blk[None, :, :]
                diff = diff.masked_fill(~tril_c.unsqueeze(-1), float("-inf"))
                l_out = torch.exp(diff)
                attn = torch.einsum("ik,ijk,jk->ij", q_ch[sl, hh], l_out, k[sl, hh])
            else:
                diff = g_blk[:, 0][:, None] - g_blk[:, 0][None, :]
                l_out = torch.exp(diff.masked_fill(~tril_c, 0.0)).tril()
                attn = (q_ch[sl, hh] @ k[sl, hh].T) * l_out
            outputs[hh, ci] = o_inter + attn.masked_fill(triu_strict, 0) @ v_new
            if vector_gate:
                state[hh] = state[hh] * torch.exp(g_blk[-1]).unsqueeze(-1) \
                    + torch.einsum("ck,cv->kv", k[sl, hh] * torch.exp(g_blk[-1] - g_blk), v_new)
            else:
                state[hh] = state[hh] * torch.exp(g_blk[-1, 0]) \
                    + torch.einsum("ck,cv->kv",
                                   k[sl, hh] * torch.exp(g_blk[-1, 0] - g_blk[:, 0]).unsqueeze(-1),
                                   v_new)
    out = outputs.reshape(h, -1, dim_v).permute(1, 0, 2)[:t]
    return out, state


def main() -> int:
    weights = ref.make_weights(cfg.SEED_BASE)
    for variant in ("scalar", "vector"):
        for n, b in ((128, 16), (100, 32), (2048, 64)):
            x = ref.make_inputs(n, seed=cfg.SEED_BASE + 7000 + n)
            proj = ref.project_inputs(x, weights, variant)
            o_ref, s_ref = chunked.chunked_gdn(proj["q"], proj["k"], proj["v"],
                                               proj["beta"], proj["g_log"], chunk_size=b)
            o_sim, s_sim = sim_chunked(proj["q"], proj["k"], proj["v"],
                                       proj["beta"], proj["g_log"], chunk_size=b)
            bit = torch.equal(o_ref, o_sim) and torch.equal(s_ref, s_sim)
            check(f"sim vs chunked_gdn 位级一致 {variant} N={n} B={b}", bit,
                  f"max_diff={max((o_ref-o_sim).abs().max().item(), (s_ref-s_sim).abs().max().item()):.2e}")
    print(f"\nkernel_sim 结果：{PASS} PASS / {FAIL} FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
