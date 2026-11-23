#!/usr/bin/env python3
"""CUDA 核算法的 CPU 逐行仿真（开发验证工具，正式产物之外）。

目的：本地无 nvcc，在上服务器编译前用 torch 仿真 mla_absorbed.cu 的算法组织
（split-K + per-warp online softmax + per-warp 累加器 + 块内合并 + 跨 split
合并），验证：
  1. 算法与 torch 吸收核（连续 softmax 参考）一致；
  2. splits > N 的空分块路径不破坏结果（对齐 .cu 的 -inf 守卫）；
  3. N=1 / 非 warp 对齐 chunk 边界的正确性。
仿真逐条对应 .cu 的数据流（含 fp32 累加、fp16 输入输出），不是逐指令仿真。

历史战绩：初版 .cu 的跨 split 合并漏掉 per-split max（pm），splits>1 全错——
本仿真在本地抓出（EXPERIMENT_LOG §12.0 bug 1），上服务器前修复。
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from mla_reference import rope_freqs, rotate_q_rope, q_projection_stage  # noqa: E402
import mla_config as cfg  # noqa: E402

WARPS = 8
MAX_SPLITS = 64
CHUNK_TARGET = 256


def emulate_split_merge(q_abs, q_rope, c_kv, k_rope, scale, splits_req=0):
    """mla_absorbed.cu 两 kernel 的算法仿真。输入数值同 .cu 的 fp16 io
    （先降 fp16 再以 fp32 做算术，对齐核的片上精度）。"""
    h = q_abs.shape[0]
    n = c_kv.shape[0]
    if splits_req > 0:
        splits = min(splits_req, MAX_SPLITS)
    else:
        splits = max(1, min((n + CHUNK_TARGET - 1) // CHUNK_TARGET, MAX_SPLITS))
    chunk = max(((n + splits - 1) // splits + WARPS - 1) // WARPS * WARPS, WARPS)

    q_abs16 = q_abs.half().float()
    q_rope16 = q_rope.half().float()
    c16 = c_kv.half().float()
    k16 = k_rope.half().float()

    psum = torch.zeros(h, splits, cfg.KV_LORA_RANK)
    pl = torch.zeros(h, splits)
    pm = torch.full((h, splits), -float("inf"))
    for sp in range(splits):
        t_start = sp * chunk
        t_end = min(t_start + chunk, n)
        for head in range(h):
            warp_m = torch.full((WARPS,), -float("inf"))
            warp_l = torch.zeros(WARPS)
            acc = torch.zeros(WARPS, cfg.KV_LORA_RANK)
            for warp in range(WARPS):
                m = -float("inf")
                l = 0.0
                a = torch.zeros(cfg.KV_LORA_RANK)
                t = t_start + warp
                while t < t_end:
                    dot = float((q_abs16[head] * c16[t]).sum())
                    rope = float((q_rope16[head] * k16[t]).sum())
                    score = (dot + rope) * scale
                    m_new = max(m, score)
                    alpha = 0.0 if m == -float("inf") else float(torch.exp(torch.tensor(m - m_new)))
                    p = float(torch.exp(torch.tensor(score - m_new)))
                    l = l * alpha + p
                    a = a * alpha + p * c16[t]
                    m = m_new
                    t += WARPS
                warp_m[warp] = m
                warp_l[warp] = l
                acc[warp] = a
            # 块内合并（对齐 .cu：重归一因子 w_i 乘 acc 与 l——注意不能乘 l_i，
            # 初版 .cu 曾在这里把 acc 乘上 w_i*l_i 导致多 token per warp 全错，
            # 见 EXPERIMENT_LOG §12.0 bug 2）
            m_global = warp_m.max()
            weights = torch.where(warp_m == -float("inf"),
                                  torch.zeros(WARPS),
                                  torch.exp(warp_m - m_global))
            warp_l_weighted = warp_l * weights
            l_total = float(warp_l_weighted.sum())
            psum[head, sp] = (weights.unsqueeze(1) * acc).sum(dim=0)
            pl[head, sp] = l_total
            pm[head, sp] = m_global
    # 跨 split 合并（flash-decoding：必须用各 split 局部 max 重归一）
    m_split_max = pm.max(dim=1, keepdim=True).values          # [H,1]
    w = torch.where(pm == -float("inf"),
                    torch.zeros_like(pm),
                    torch.exp(pm - m_split_max))              # [H,S]
    l_total_all = (w * pl).sum(dim=1, keepdim=True)           # [H,1]
    inv = torch.where(l_total_all > 0, 1.0 / l_total_all, torch.zeros_like(l_total_all))
    s = ((w.unsqueeze(-1) * psum).sum(dim=1) * inv)           # [H,512]
    return s.half().float()


def main() -> int:
    data = np.load(HERE.parent.parent / "fixtures" / "correctness.npz")
    arrays = {k: data[k] for k in data.files}
    freqs = rope_freqs()
    weights = {k[7:]: torch.from_numpy(arrays[k]) for k in arrays if k.startswith("weight_")}
    x = torch.from_numpy(arrays["x"])
    n = 128
    c_kv = torch.from_numpy(arrays[f"n{n}_c_kv"])
    k_rope = torch.from_numpy(arrays[f"n{n}_k_rope_rot"])
    q_nope, q_rope_raw = q_projection_stage(x, weights)
    q_rope = rotate_q_rope(q_rope_raw, n, freqs)
    q_abs = torch.einsum("hd,hcd->hc", q_nope, weights["W_UK"])
    scale = cfg.SOFTMAX_SCALE

    scores = ((q_abs @ c_kv.t()) + q_rope @ k_rope.t()) * scale
    ref_s = torch.softmax(scores, dim=-1) @ c_kv

    for label, splits in (("heuristic", 0), ("splits=1", 1), ("splits=16(>chunk)", 16),
                          ("splits=64(>N/2)", 64)):
        s = emulate_split_merge(q_abs, q_rope, c_kv, k_rope, scale, splits)
        err = (s - ref_s).abs().max().item()
        rel = err / ref_s.abs().max().item()
        status = "OK" if rel < 2e-3 else "FAIL"   # fp16 io 噪声容差（ref 是 fp32）
        print(f"[sim] {label}: max={err:.3e} rel={rel:.2e} {status}")

    for nb in (1, 3):
        cb = torch.from_numpy(arrays[f"n{nb}_c_kv"])
        kb = torch.from_numpy(arrays[f"n{nb}_k_rope_rot"])
        scores_b = ((q_abs @ cb.t()) + q_rope @ kb.t()) * scale
        ref_b = torch.softmax(scores_b, dim=-1) @ cb
        s = emulate_split_merge(q_abs, q_rope, cb, kb, scale, 0)
        err = (s - ref_b).abs().max().item()
        print(f"[sim] N={nb}: max={err:.3e} rel={err / ref_b.abs().max().item():.2e} "
              f"{'OK' if err / ref_b.abs().max().item() < 2e-3 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
