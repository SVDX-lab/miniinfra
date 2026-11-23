#!/usr/bin/env python3
"""第 29 期 kernel_sim：对 AI 起草的三份 .cu 做逐 block/warp 语义的 CPU 仿真。

这是静态预审的行为学部分（ep21/22 kernel_sim 模式在 AI 起草上的应用）：
按 AI 源码的字面组织（块大小、线程到数据的映射、warp 蝶形归约、shared
暂存、同步点）在 numpy 里逐步仿真，与 fixtures 真值对拍。目的：
  - 在无 GPU 的本地暴露"结构正确性"缺陷（归约组织、guard、映射错误）；
  - 浮点舍入噪声与 GPU（rsqrtf/expf 近似、SIMD 求和序）同数量级，容差取
    宽（结构性错误远大于舍入噪声），判断是结构性的而非数值性的。

仿真的已知局限（记入预审结论）：
  - 不仿真 CUDA 语法/launch 配置错误（那部分归 stub 编译检查与服务器构建）；
  - smem 容量只按静态公式检查（attention: N*4B vs 48KB 默认上限）；
  - 不仿真数据竞争时序（同步点按源码顺序完整执行）。

AI 源码的三个已知组织（生成冻结版）：
  - RMSNorm: 每 block 一行，BLOCK=256，warp 蝶形 + shared 跨 warp 归约，
    跨 warp 段只在 warp 0 内完成（返回值未广播——见 process-notes）；
  - GEMM: TILE=16 shared 分块，k 维按 tile 推进，块内 guard；
  - decode Attention: 每 block 一个 (b,h)，threads=128=D，scores 落
    dynamic smem，材料化 softmax（max/exp/sum 三段），线程 d 拥有输出维 d。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

F32 = np.float32

RMSNORM_BLOCK = 256
ATTN_THREADS = 128
ATTN_HQ, ATTN_HKV, ATTN_D = 16, 8, 128
ATTN_SCALE = F32(0.08838834764831845)
ATTN_SMEM_DEFAULT_LIMIT = 48 * 1024  # sm_89 每 block 默认 dynamic smem 上限
GEMM_TILE = 16
EPS = F32(1e-6)


def _butterfly_sum(v: np.ndarray, axis_lane: int = -1) -> np.ndarray:
    """__shfl_xor_sync 蝶形求和的逐步仿真（fp32 逐步加，与硬件加序一致）。"""
    n_lanes = v.shape[-1]
    assert n_lanes & (n_lanes - 1) == 0
    offsets = [n_lanes >> i for i in range(1, int(math.log2(n_lanes)) + 1)]
    for off in offsets:
        idx = np.arange(n_lanes) ^ off
        v = (v + v[..., idx]).astype(F32)
    return v


def _butterfly_max(v: np.ndarray) -> np.ndarray:
    n_lanes = v.shape[-1]
    offsets = [n_lanes >> i for i in range(1, int(math.log2(n_lanes)) + 1)]
    for off in offsets:
        idx = np.arange(n_lanes) ^ off
        v = np.maximum(v, v[..., idx]).astype(F32)
    return v


# ---------------------------------------------------------------------------
# RMSNorm（AI 字面仿真，含归约返回值未广播的组织缺陷）
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    output: np.ndarray | None
    launch_ok: bool = True
    notes: list[str] = field(default_factory=list)
    per_thread_scale_warp0: np.ndarray | None = None  # 调试证据


def simulate_rmsnorm(x: np.ndarray, w: np.ndarray) -> SimResult:
    """x [rows,H] fp32, w [H] fp32 → y [rows,H]（AI 版算法仿真）。"""
    rows, hidden = x.shape
    x = x.astype(F32)
    w = w.astype(F32)
    block = RMSNORM_BLOCK
    warps = block // 32

    # 每线程部分和（线程 t 累加 i = t, t+BLOCK, ... 升序）
    nchunks = (hidden + block - 1) // block
    padded = np.zeros((rows, nchunks * block), dtype=F32)
    padded[:, :hidden] = x
    chunks = padded.reshape(rows, nchunks, block)
    partial = np.zeros((rows, block), dtype=F32)
    for c in range(nchunks):
        v = chunks[:, c, :]
        partial = (partial + v * v).astype(F32)  # 线程内串行累加

    # warp 蝶形
    v = partial.reshape(rows, warps, 32)
    warp_tot = _butterfly_sum(v)[:, :, 0]  # [rows, warps]

    # 跨 warp：AI 源码字面——warp 0 内蝶形得到 block 总和；warp 1..k 的线程
    # 只带着自己的 warp 部分和返回（BlockReduceSum 缺广播）
    lane_vals = np.zeros((rows, 32), dtype=F32)
    lane_vals[:, :warps] = warp_tot
    block_tot = _butterfly_sum(lane_vals)[:, 0]  # [rows]

    # 每线程 scale：warp 0 用 block 总和，其余 warp 用本 warp 部分和（缺陷复现）
    warp_idx = np.arange(block) // 32
    scale = np.empty((rows, block), dtype=F32)
    scale[:, :] = F32(0)
    scale[:, warp_idx == 0] = (1.0 / np.sqrt(
        block_tot[:, None] / F32(hidden) + EPS)).astype(F32)
    for widx in range(1, warps):
        sel = warp_idx == widx
        scale[:, sel] = (1.0 / np.sqrt(
            warp_tot[:, [widx]] / F32(hidden) + EPS)).astype(F32)

    # 写回：线程 t 负责 i = t, t+BLOCK, ...（元素 i 的归属线程 = i % BLOCK）
    y = np.zeros((rows, hidden), dtype=F32)
    for c in range(nchunks):
        idx = np.arange(block) + c * block
        m = idx < hidden
        if not m.any():
            continue
        thr = idx[m]
        owner = idx[m] % block
        y[:, thr] = (w[thr][None, :] * x[:, thr] * scale[:, owner]).astype(F32)
    notes = [
        "组织缺陷仿真：跨 warp 归约结果只在 warp 0 的 32 个线程里是 block 总和；"
        f"warp 1..{warps - 1} 的线程用本 warp 部分和算 scale（hidden>32 时触发）",
    ]
    return SimResult(output=y, notes=notes,
                     per_thread_scale_warp0=block_tot)


# ---------------------------------------------------------------------------
# GEMM（tile 序 k 推进 + guard 的字面仿真）
# ---------------------------------------------------------------------------

def simulate_gemm(a: np.ndarray, b: np.ndarray) -> SimResult:
    """A [M,K], B [K,N] → C [M,N]；k 维按 TILE 推进（AI 版字面）。"""
    m, k = a.shape
    _, n = b.shape
    a = a.astype(F32)
    b = b.astype(F32)
    acc = np.zeros((m, n), dtype=F32)
    for t in range((k + GEMM_TILE - 1) // GEMM_TILE):
        lo, hi = t * GEMM_TILE, min((t + 1) * GEMM_TILE, k)
        acc = (acc + a[:, lo:hi] @ b[lo:hi, :]).astype(F32)
    return SimResult(output=acc, notes=[
        "tile 序推进仿真：guard 由 numpy 切片天然完成；归约无跨线程项，"
        "预期与真值仅差 k 求和顺序的舍入噪声（无结构性缺陷通道）",
    ])


# ---------------------------------------------------------------------------
# decode Attention（材料化 softmax + smem 上限的字面仿真）
# ---------------------------------------------------------------------------

def simulate_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> SimResult:
    """q [B,HQ,D], k/v [B,HKV,N,D] → o [B,HQ,D]（AI 版算法仿真）。"""
    b_dim, hq, d = q.shape
    _, hkv, n, d2 = k.shape
    assert d == d2 == ATTN_D
    q = q.astype(F32)
    k = k.astype(F32)
    v = v.astype(F32)
    thr = ATTN_THREADS
    warps = thr // 32

    smem_bytes = n * 4
    if smem_bytes > ATTN_SMEM_DEFAULT_LIMIT:
        return SimResult(output=None, launch_ok=False, notes=[
            f"launch 失败仿真：dynamic smem {smem_bytes}B > 默认上限 "
            f"{ATTN_SMEM_DEFAULT_LIMIT}B（N={n}；AI 源码未 opt-in 更大 smem）",
        ])

    o = np.zeros((b_dim, hq, d), dtype=F32)
    for b in range(b_dim):
        for h in range(hq):
            kv = h // 2
            qh = q[b, h]                                   # [D]
            kh = k[b, kv]                                  # [N,D]
            vh = v[b, kv]
            # Phase 1: 线程 t%THR 算 s[t]（线程内 d 升序串行——用点积，舍入噪声级）
            s = (kh @ qh * ATTN_SCALE).astype(F32)         # [N]
            # Phase 2: 每线程局部 max（其拥有的 t），warp 蝶形
            nchunks = (n + thr - 1) // thr
            padded = np.full(nchunks * thr, -np.inf, dtype=F32)
            padded[:n] = s
            chunks = padded.reshape(nchunks, thr)
            local_max = np.full(thr, -np.inf, dtype=F32)
            for c in range(nchunks):
                local_max = np.maximum(local_max, chunks[c]).astype(F32)
            warp_max = _butterfly_max(local_max.reshape(warps, 32))[:, 0]
            lane_vals = np.full(32, -np.inf, dtype=F32)
            lane_vals[:warps] = warp_max
            block_max = _butterfly_max(lane_vals)[0]
            # 缺陷复现：线程的 s_max 只有 warp 0 是全局值
            s_max_per_thr = np.empty(thr, dtype=F32)
            s_max_per_thr[:32] = block_max
            for widx in range(1, warps):
                s_max_per_thr[widx * 32:(widx + 1) * 32] = warp_max[widx]
            # Phase 3: exp（用各自线程的 s_max）+ 局部和 + 块和（同型缺陷）
            padded_scores = np.zeros(nchunks * thr, dtype=F32)
            padded_scores[:n] = s
            exp_all = np.exp(padded_scores - np.repeat(s_max_per_thr, nchunks)
                             ).astype(F32)
            exp_all[n:] = 0.0
            exp_chunks = exp_all.reshape(nchunks, thr)
            local_sum = np.zeros(thr, dtype=F32)
            for c in range(nchunks):
                local_sum = (local_sum + exp_chunks[c]).astype(F32)
            warp_sum = _butterfly_sum(local_sum.reshape(warps, 32))[:, 0]
            lane_sums = np.zeros(32, dtype=F32)
            lane_sums[:warps] = warp_sum
            block_sum = _butterfly_sum(lane_sums)[0]
            inv_per_thr = np.empty(thr, dtype=F32)
            inv_per_thr[:32] = F32(1.0) / block_sum
            for widx in range(1, warps):
                inv_per_thr[widx * 32:(widx + 1) * 32] = F32(1.0) / warp_sum[widx]
            p = (exp_all * np.repeat(inv_per_thr, nchunks)).astype(F32)
            p[n:] = 0.0
            # Phase 5: 线程 d 串行累加 o[d] = Σ_t p[t]*v[t,d]（t 升序）
            acc = np.zeros(d, dtype=F32)
            for t in range(n):
                acc = (acc + p[t] * vh[t]).astype(F32)
            o[b, h] = acc
    notes = [
        "组织缺陷仿真：s_max/inv_sum 只在 warp 0 线程中是全局正确值，"
        "warp 1..3 拥有的 t 值权重错（N>32 触发）；"
        f"smem 需求 {n}×4B（默认上限内：{smem_bytes <= ATTN_SMEM_DEFAULT_LIMIT}）",
    ]
    return SimResult(output=o, notes=notes)


# ---------------------------------------------------------------------------
# 对拍入口（fixtures 真值）
# ---------------------------------------------------------------------------

def run_all(max_rows: int = 64, max_n: int = 512) -> dict:
    """预审对拍：控制规模以适配本地 CPU（rows≤max_rows, N≤max_n）。

    全规模（30 组全量 / N=2048）留给 GPU 阶段对照；结构性缺陷在小形状即
    现形（H>32 / N>32 即触发），故限规模不损失检出力。
    """
    import json
    from verify_fixtures import REPO_ROOT, resolve_npz

    root = REPO_ROOT
    results: dict = {"schema_version": "kernel-chapter30.kernel-sim.v1", "cases": []}

    m3 = json.loads((root / "chapter_02/fixtures/manifest.json").read_text())
    d3 = np.load(resolve_npz("chapter_02/fixtures/correctness.npz"))
    for combo in m3["combos"]:
        r, h = combo["rows"], combo["hidden_size"]
        if r > max_rows:
            continue
        x = d3[f"x_{r}x{h}"]
        w = d3[f"w_{h}"]
        truth = d3[f"y_{r}x{h}"]
        sim = simulate_rmsnorm(x, w)
        err = float(np.abs(sim.output - truth).max())
        results["cases"].append({
            "operator": "rmsnorm", "shape": f"{r}x{h}",
            "max_abs_error": round(err, 8),
            "structural_fail": bool(err > 1e-3),
            "notes": sim.notes,
        })

    m18 = json.loads((root / "chapter_17/fixtures/manifest.json").read_text())
    d18 = np.load(resolve_npz("chapter_17/fixtures/correctness.npz"))
    for combo in m18["combos"]:
        m, k, n = combo["shape"]
        if m > max_rows or k > 1024:
            continue
        def arr(pref):
            name = f"{pref}_{m}x{k}x{n}"
            return d18[name] if name in d18.files else d18[f"{pref}"]
        a, bb, truth = arr("A"), arr("B"), arr("C")
        sim = simulate_gemm(a, bb)
        err = float(np.abs(sim.output - truth).max())
        results["cases"].append({
            "operator": "gemm", "shape": f"{m}x{k}x{n}",
            "max_abs_error": round(err, 8),
            "structural_fail": bool(err > 5e-3),
            "notes": sim.notes,
        })

    m21 = json.loads((root / "chapter_20/fixtures/manifest.json").read_text())
    d21 = np.load(resolve_npz("chapter_20/fixtures/correctness.npz"))
    for n_len in (64, 512, 2048):
        for b_dim in (1, 4):
            if n_len > max_n:
                continue
            q = d21[f"d_n{n_len}_b{b_dim}_q"]
            kk = d21[f"d_n{n_len}_b{b_dim}_k"]
            vv = d21[f"d_n{n_len}_b{b_dim}_v"]
            truth = d21[f"d_n{n_len}_b{b_dim}_truth32"]
            sim = simulate_attention(q, kk, vv)
            if not sim.launch_ok:
                results["cases"].append({
                    "operator": "attention", "shape": f"N{n_len}B{b_dim}",
                    "launch_ok": False, "structural_fail": True,
                    "notes": sim.notes,
                })
                continue
            err = float(np.abs(sim.output - truth).max())
            results["cases"].append({
                "operator": "attention", "shape": f"N{n_len}B{b_dim}",
                "max_abs_error": round(err, 8),
                "structural_fail": bool(err > 1e-3),
                "notes": sim.notes,
            })

    results["structural_fail_count"] = sum(
        1 for c in results["cases"] if c.get("structural_fail"))
    results["case_count"] = len(results["cases"])
    return results


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    max_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    rep = run_all(max_rows=max_rows, max_n=max_n)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
