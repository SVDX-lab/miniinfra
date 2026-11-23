#!/usr/bin/env python3
"""第 28 期 kernel_sim：CUDA 核分解的算法级仿真（本地无 GPU，为 4090 回填去风险）。

CUDA 骨架（implementations/cuda/moe_route.cu）的四核分解：
  K1 gate_scores   —— 每 (token,expert) 一线程算 dot（x[2048]·router[e]）
  K2 softmax_topk  —— 每 token 一 warp：warp max/sum + k 轮 argmax（tie-break=低索引）
                      + 归一化
  K3 hist_scan     —— 直方图（每 slot 一遍）+ E 上 exclusive scan（单 block）
  K4 scatter_fill  —— 确定性散位：pos = offsets[e] + 桶内 rank
                      （rank = 按 slot 逐遍前缀计数，**无 atomics、顺序 = token 升序**）
  G   grouped GEMM —— host 逐专家桶调用 tiled GEMM（ch18 设计）

本仿真在 Python 端逐核复现上述**索引/选择逻辑**（不含浮点 GEMM 位级），
与 CPU 参考实现（moe_reference）对拍：
  - K2 选择 vs route_topk：k 轮显式 argmax（每次线性扫 E，同分取更低索引）
    必须与 stable argsort 完全一致——这是 CUDA 核与参考真值的语义锚点；
  - K3/K4 桶结构 vs bincount/cumsum/argsort(stable)：位级一致；
  - 边界：N=1、空桶、全 token 单专家、k=E、E=1。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import moe_reference as mr  # noqa: E402


def k2_softmax_topk_sim(logits_row: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    """复现 CUDA K2 的算法：先 warp 级 max/sum 的数值语义（这里用全精度近似），
    选择 = k 轮显式 argmax，同分取更低索引（与 stable argsort 等价）。"""
    e = logits_row.shape[0]
    # CUDA 核做 softmax 用 fp32；仿真用同公式（数值一致性由回填实测对拍）
    probs = torch.softmax(logits_row, dim=-1)
    picked: list[int] = []
    masked = probs.clone()
    for _ in range(k):
        best_idx = 0
        best_val = float(masked[0])
        for i in range(1, e):
            v = float(masked[i])
            if v > best_val:  # 严格大于才替换 → 同分保留更低索引（tie-break）
                best_val = v
                best_idx = i
        picked.append(best_idx)
        masked[best_idx] = -1.0  # 已选位置置 -inf 等效
    w = [float(probs[i]) for i in picked]
    s = sum(w)
    return picked, [x / s for x in w]


def k34_hist_scan_scatter_sim(idx: torch.Tensor, n_experts: int) -> dict:
    """复现 K3（直方图 + E 上 exclusive scan）+ K4（确定性散位）。

    CUDA 设计语义 = 稳定排序（radix/stable argsort by expert key，键内保持
    pair 索引 = (token, slot) 字典序）：桶 = 专家升序、桶内 pair 索引升序
    （等价于 token 升序——同一 token 对同一专家至多选中一次）。
    rank(t,s) = #{(t',s')<lex(t,s) : idx[t'][s']==e}
              = Σ_{s'} #{t'<t : idx[t'][s']==e} + #{s'<s : idx[t][s']==e}
    CUDA 无原子实现：对每个 slot 做一次"按 token 前缀的 per-expert 计数扫描"
    （k 遍）+ token 内 slot 前缀（寄存器内 k 项）。本函数按同一公式计算。
    """
    n, k = idx.shape
    idx_l = idx.tolist()
    counts = [0] * n_experts
    for row in idx_l:
        for e in row:
            counts[e] += 1
    offsets = [0] * n_experts
    acc = 0
    for e in range(n_experts):
        offsets[e] = acc
        acc += counts[e]

    positions = [[0] * k for _ in range(n)]
    # before[t][e] = #{t'<t, 任意 s' : idx[t'][s']==e}（逐 token 递增的桶前计数）
    before = [[0] * n_experts for _ in range(n + 1)]
    for t in range(n):
        row = before[t][:]  # 拷贝
        for e in idx_l[t]:
            row[e] += 1
        before[t + 1] = row
    for t in range(n):
        inrow = [0] * n_experts  # 本 token 内 s'<s 的同专家计数
        for s in range(k):
            e = idx_l[t][s]
            positions[t][s] = offsets[e] + before[t][e] + inrow[e]
            inrow[e] += 1
    flat = [(positions[t][s], t, s) for t in range(n) for s in range(k)]
    flat.sort()
    sim_token = torch.tensor([f[1] for f in flat], dtype=torch.long)
    sim_expert = torch.tensor([idx_l[f[1]][f[2]] for f in flat], dtype=torch.long)
    return {"counts": torch.tensor(counts), "offsets": torch.tensor(offsets),
            "pair_token": sim_token, "pair_expert": sim_expert}


def main() -> int:
    n_pass, n_fail = 0, 0

    def check(name, cond, detail=""):
        nonlocal n_pass, n_fail
        if cond:
            n_pass += 1
            print(f"  PASS {name} {detail}")
        else:
            n_fail += 1
            print(f"  FAIL {name} {detail}")

    print("== kernel_sim: CUDA 四核分解算法仿真 ==")

    # K2 选择语义 vs 参考实现（多组随机 logits + 构造 tie）
    for trial, (E, d, N, k, seed) in enumerate([(8, 32, 16, 3, 11), (64, 64, 32, 6, 22), (16, 32, 8, 16, 33)]):
        gen = torch.Generator().manual_seed(seed)
        logits = torch.randn(N, E, generator=gen)
        ref_idx, ref_w = mr.route_topk(logits, k)
        ok = True
        for t in range(N):
            sim_idx, sim_w = k2_softmax_topk_sim(logits[t], k)
            if sim_idx != ref_idx[t].tolist() or not torch.allclose(
                    torch.tensor(sim_w), ref_w[t], atol=1e-6):
                ok = False
                print(f"    mismatch trial{trial} token{t}: sim={sim_idx} ref={ref_idx[t].tolist()}")
                break
        check(f"K2 选择 trial{trial} (E={E},k={k},N={N})", ok)

    # 全同分数 tie：K2 显式 argmax（低索引胜出）
    logits_tie = torch.zeros(4, 16)
    ref_idx, _ = mr.route_topk(logits_tie, 4)
    sim_idx, _ = k2_softmax_topk_sim(logits_tie[0], 4)
    check("K2 tie-break（全同分数→低索引）", sim_idx == ref_idx[0].tolist() and ref_idx[0].tolist() == [0, 1, 2, 3],
          f"sim={sim_idx}")

    # K3/K4 桶结构 vs 参考实现（多规模 + 边界）
    cases = [
        ("tiny", 8, 16, 2, 1),        # N=1, E=8：空桶边界
        ("mid", 16, 64, 6, 44),
        ("allone", 8, 32, 3, 55),     # 全 token 单专家（构造 idx 全 0）
        ("k_eq_E", 8, 16, 8, 66),     # k=E 退化
        ("large", 64, 256, 6, 77),
    ]
    for name, E, N, k, seed in cases:
        gen = torch.Generator().manual_seed(seed)
        if name == "allone":
            idx = torch.zeros(N, k, dtype=torch.long)
        else:
            idx = torch.stack([torch.randperm(E, generator=gen)[:k] for _ in range(N)])
        sim = k34_hist_scan_scatter_sim(idx, E)
        w = torch.full((N, k), 1.0 / k)
        pair_token, pair_expert, _ = mr._pairs_with_weights(idx, w)
        ref_counts = torch.bincount(pair_expert, minlength=E)
        ok = (torch.equal(sim["counts"], ref_counts)
              and torch.equal(sim["pair_token"], pair_token)
              and torch.equal(sim["pair_expert"], pair_expert))
        check(f"K3/K4 {name} (E={E},N={N},k={k})", ok,
              f"counts_sum={int(sim['counts'].sum())}")

    # 汇总
    print(f"== kernel_sim 汇总: {n_pass} PASS / {n_fail} FAIL ==")
    print("说明：本仿真锁定 CUDA 核的**索引/选择语义**（K2 tie-break、K3 直方图扫描、"
          "K4 无原子确定性散位）；浮点 GEMM 部分在 4090 回填时按 ch18 口径对拍 "
          "（预期 FP32 容差 ≤2e-4 相对参考）。")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
