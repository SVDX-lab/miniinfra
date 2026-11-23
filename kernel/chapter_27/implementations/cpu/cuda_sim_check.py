#!/usr/bin/env python3
"""CUDA 三核逻辑的 Python 镜像验证（nvcc 构建前的桌面检查）。

逐相位镜像 deltanet_chunked.cu 的精确语义：
  K1：越界行读 0 → cumsum → m1[i][j] = −β_i·(k_i·k_j)·e^{G_i−G_j}（j<i）→
      wpay=βv / kpay=kβe^G；
  K2：就地前向替换 X_i = P_i + Σ_{j<i} m1[i][j]·X_j（m1 已取负）；
  K3：scale 只在输出乘 + attn 含对角（j≤i）+ v_new=w−kd@S +
      state=S·e^{G_last}+k_backᵀ@v_new；scheme A/B 状态载体差异。
与 chunked_gdn（fp64）对拍容差 1e-12。通过后才上 nvcc。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parent.parent
sys.path.insert(0, str(CHAPTER))
sys.path.insert(0, str(HERE.parent / "cpu"))

import deltanet_chunked as chunked  # noqa: E402
import deltanet_config as cfg  # noqa: E402
import deltanet_reference as ref  # noqa: E402

KD, VD, HALF = cfg.HEAD_K_DIM, cfg.HEAD_V_DIM, cfg.HEAD_V_DIM // 2


def cuda_mirror(q, k, v, beta, g_log, c, scale, s0=None, scheme=0):
    """deltanet_chunked.cu 三核的逐相位 Python 镜像（fp64）。"""
    n, h, _ = k.shape
    nc = (n + c - 1) // c
    dt = torch.float64

    def ld(t, row, *rest):
        """越界读 0（镜像 CUDA bounds-checked load）。"""
        if 0 <= row < n:
            return t[row, *rest]
        return torch.zeros(*[s for s in rest], dtype=dt) if rest else torch.tensor(0.0, dtype=dt)

    m1 = torch.zeros(nc, h, c, c, dtype=dt)
    wpay = torch.zeros(nc, h, c, VD, dtype=dt)
    kpay = torch.zeros(nc, h, c, KD, dtype=dt)

    # ---- K1 ----
    for ci in range(nc):
        base = ci * c
        for hh in range(h):
            g = torch.zeros(c, dtype=dt)
            for j in range(c):
                g[j] = ld(g_log, base + j, hh).item() if base + j < n else 0.0
            gcum = torch.cumsum(g, dim=0)
            kb = torch.stack([ld(k, base + j, hh) if base + j < n
                              else torch.zeros(KD, dtype=dt) for j in range(c)])
            for i in range(c):
                row_i = base + i
                bi = ld(beta, row_i, hh).item() if row_i < n else 0.0
                for j in range(c):
                    if j < i:
                        m1[ci, hh, i, j] = -bi * (kb[i] @ kb[j]).item() * math.exp(gcum[i] - gcum[j])
                vi = ld(v, row_i, hh) if row_i < n else torch.zeros(VD, dtype=dt)
                wpay[ci, hh, i] = bi * vi
                kpay[ci, hh, i] = bi * kb[i] * math.exp(gcum[i])

    # ---- K2（就地前向替换）----
    for ci in range(nc):
        for hh in range(h):
            for i in range(1, c):
                neg = m1[ci, hh, i, :i]
                wpay[ci, hh, i] = wpay[ci, hh, i] + neg @ wpay[ci, hh, :i]
                kpay[ci, hh, i] = kpay[ci, hh, i] + neg @ kpay[ci, hh, :i]

    # ---- K3 ----
    state = (s0.clone().to(dt) if s0 is not None else torch.zeros(h, KD, VD, dtype=dt))
    o = torch.zeros(n, h, VD, dtype=dt)
    for hh in range(h):
        for half in range(2):
            col0 = half * HALF
            if scheme == 0:
                st_local = state[hh, :, col0:col0 + HALF].clone()
            for ci in range(nc):
                base = ci * c
                g = torch.zeros(c, dtype=dt)
                for j in range(c):
                    g[j] = ld(g_log, base + j, hh).item() if base + j < n else 0.0
                gcum = torch.cumsum(g, dim=0)
                g_last = gcum[-1]
                st_view = st_local if scheme == 0 else state[hh, :, col0:col0 + HALF]

                attn = torch.zeros(c, c, dtype=dt)
                for i in range(c):
                    for j in range(c):
                        if j <= i:
                            qi = ld(q, base + i, hh) if base + i < n else torch.zeros(KD, dtype=dt)
                            kj = ld(k, base + j, hh) if base + j < n else torch.zeros(KD, dtype=dt)
                            attn[i, j] = (qi @ kj).item() * math.exp(gcum[i] - gcum[j])

                vnew = torch.zeros(c, VD, dtype=dt)
                for i in range(c):
                    w_row = wpay[ci, hh, i, col0:col0 + HALF]
                    kd_row = kpay[ci, hh, i]
                    vnew[i, col0:col0 + HALF] = w_row - kd_row @ st_view

                for i in range(c):
                    row = base + i
                    if row < n:
                        qi = ld(q, row, hh)
                        o_inter = (qi * math.exp(gcum[i])) @ st_view
                        acc = o_inter + attn[i, :i + 1] @ vnew[:i + 1, col0:col0 + HALF]
                        o[row, hh, col0:col0 + HALF] = scale * acc

                new_st = torch.zeros(KD, HALF, dtype=dt)
                for kk in range(KD):
                    for i in range(c):
                        kv = ld(k, base + i, hh)[kk].item() if base + i < n else 0.0
                        new_st[kk] += kv * math.exp(g_last - gcum[i]) * vnew[i, col0:col0 + HALF]
                if scheme == 0:
                    st_local = st_view * math.exp(g_last) + new_st
                else:
                    state[hh, :, col0:col0 + HALF] = st_view * math.exp(g_last) + new_st
            if scheme == 0:
                state[hh, :, col0:col0 + HALF] = st_local
    return o, state


def main() -> int:
    weights = ref.make_weights(cfg.SEED_BASE)
    n_fail = 0
    for n, b, scheme in ((33, 16, 0), (100, 32, 0), (128, 64, 1), (33, 16, 1), (70, 64, 0)):
        gen = torch.Generator().manual_seed(cfg.SEED_BASE + 7000 + n)
        x = torch.randn(n, cfg.D_MODEL, generator=gen, dtype=torch.float32)
        proj = ref.project_inputs(x, weights, "scalar")
        o_ref, s_ref = chunked.chunked_gdn(proj["q"].double(), proj["k"].double(),
                                           proj["v"].double(), proj["beta"].double(),
                                           proj["g_log"].double(), chunk_size=b)
        o_sim, s_sim = cuda_mirror(proj["q"].double(), proj["k"].double(), proj["v"].double(),
                                   proj["beta"].double(), proj["g_log"].double(),
                                   b, cfg.SCALE, scheme=scheme)
        ro = ((o_sim - o_ref).abs().max() / o_ref.abs().max()).item()
        rs = ((s_sim - s_ref).abs().max() / s_ref.abs().max()).item()
        ok = ro < 1e-12 and rs < 1e-12
        n_fail += (not ok)
        print(f"[cuda-sim] N={n} B={b} scheme={scheme}: rel_o={ro:.3e} rel_s={rs:.3e} "
              f"{'OK' if ok else 'FAIL'}")
    print(f"cuda_sim_check：{5 - n_fail}/5 PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
