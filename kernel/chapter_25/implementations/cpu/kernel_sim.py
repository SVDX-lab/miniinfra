#!/usr/bin/env python3
"""CUDA 核算法的 CPU 逐行仿真（开发验证工具，正式产物之外）。

目的：本地无 nvcc，上服务器前用 torch/numpy 仿真 dsa_sparse.cu 的算法组织，
抓逻辑 bug（ep25 战绩：两个正确性 bug 都是本地仿真先抓出的）。仿真三条链：

  1. 索引器 score 核：warp/lanes 分工（lane L 持 k^I 维 [4L,4L+4)）、逐头
     4 维部分点积 + xor-shfl 树归约、ReLU + 头加权（fp16 io / fp32 累加）；
  2. 阈值 top-k：fp32→uint32 单调键变换 + 32 步二分 count(≥thr) 求 kth 大 +
     两段确定性 compact（>T 先按块前缀写、==T 按"块前缀 + 全局配额"写）；
  3. gather split attention：ep25 吸收核骨架 + 索引间接寻址（identity 模式 =
     dense 对照）、per-warp online softmax、块内 8 warp 合并（w_i 乘 acc）、
     跨 split 合并（pm 局部 max 重归一）、空 split -inf 守卫、k=0 空集。

仿真逐条对应 .cu 的数据流（含 fp16 io、fp32 片上），不是逐指令仿真。
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import dsa_config as cfg  # noqa: E402
from dsa_reference import (  # noqa: E402
    SEED_BASE,
    absorbed_query,
    apply_rope,
    index_query_stage,
    index_scores,
    index_weights_stage,
    make_weights,
    plant_directions,
    q_projection_stage,
    rope_freqs,
    select_topk,
    sparse_gather_attention_bmm,
    calibrate_plants,
    generate_cache,
)

WARPS = 8
LANES = 32
THREADS = WARPS * LANES
MAX_SPLITS = 64
CHUNK_TARGET = 256
N_INDEX_BLOCKS = 256  # topk count/compact 的 grid 上界（N=65536 / 256 线程）


# ---------------------------------------------------------------------------
# 1. 索引器 score 核仿真（逐 warp 逐头，lane 分工 + shfl 归约顺序）
# ---------------------------------------------------------------------------

def check_indexer_grid_cover(n: int) -> bool:
    """索引器核 grid 循环界覆盖检查：warp 的 t 序列（block_start+warp 起、步长
    WARPS、界 [block_start, block_end)）必须不重不漏覆盖 [0, n)。

    历史战绩：初版 .cu 把循环上界写成全局 n（漏写块尾 block_end），每块越界
    扫到 N——N=65536 时 128× 重复计算（实测 34.7ms），本检查在本地即可抓出。"""
    covered = set()
    blocks = (n + THREADS - 1) // THREADS
    for b in range(blocks):
        block_start = b * THREADS
        block_end = min(block_start + THREADS, n)
        for warp in range(WARPS):
            t = block_start + warp
            while t < block_end:
                covered.add(t)
                t += WARPS
    return covered == set(range(n))


def emulate_indexer_scores(q_i16, k_i16, w_used, n) -> torch.Tensor:
    """按 .cu 最终数据流算 index 分数：lane L 持 dims (2L, 2L+1, 2L+64, 2L+65)
    （两个 uint 的配对布局，q shared 读零冲突），逐头部分和 + xor 树归约，
    ReLU 后按 w_used 加权累加（fp32）。q_i/k_i 已是 fp16→fp32。"""
    h, d = q_i16.shape
    scores = torch.zeros(n)
    for t in range(n):
        # lane l 的 k 部分：dims (2l, 2l+1, 2l+64, 2l+65)
        k_low = k_i16[t][:64].reshape(LANES, 2)           # dims 2l, 2l+1
        k_high = k_i16[t][64:].reshape(LANES, 2)          # dims 2l+64, 2l+65
        k_lane = torch.cat([k_low, k_high], dim=1)        # [32,4]
        acc = 0.0
        for head in range(h):
            q_low = q_i16[head][:64].reshape(LANES, 2)
            q_high = q_i16[head][64:].reshape(LANES, 2)
            q_lane = torch.cat([q_low, q_high], dim=1)    # [32,4]
            partial = (q_lane * k_lane).sum(dim=1)        # [32] 每 lane 4 维部分和
            dot = partial.sum()                           # xor 树归约 = 全和（fp32 求和顺序差异在容差内）
            acc += float(w_used[head]) * max(float(dot) * cfg.INDEX_SCALE, 0.0)
        scores[t] = acc
    return scores


# ---------------------------------------------------------------------------
# 2. 阈值 top-k 仿真（fp32→uint32 单调键 + 二分 count + 确定性 compact）
# ---------------------------------------------------------------------------

def float_to_key(x: np.ndarray) -> np.ndarray:
    """fp32 → 单调 uint32 键（符号翻转技巧：正数翻转符号位、负数全翻）。"""
    bits = x.astype(np.float32).view(np.uint32)
    sign = (bits >> 31).astype(np.uint32)
    return bits ^ (sign * 0xFFFFFFFF + (1 - sign) * 0x80000000)


def key_to_float(key: np.ndarray) -> np.ndarray:
    sign = (key >> 31).astype(np.uint32)
    bits = key ^ (sign * 0x80000000 + (1 - sign) * 0xFFFFFFFF)
    return bits.view(np.float32)


def emulate_topk(scores_fp32: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    """阈值法 top-k：返回 (ascending int32 索引, 统计)。

    二分 32 步：不变量 count(key ≥ lo) ≥ k；结束 lo = 第 k 大的键。
    compact：>T 的按块前缀升序写；==T 的按"块前缀 + 全局配额 (k−c1)"升序写。"""
    n = scores_fp32.shape[0]
    k = min(k, n)
    keys = float_to_key(scores_fp32)
    # 二分（CUDA 侧每步一个 count kernel，此处同序逻辑）
    lo, hi = np.uint32(0), np.uint32(0xFFFFFFFF)
    iterations = 0
    while lo < hi:
        mid = np.uint32((int(lo) + int(hi)) // 2 + 1)  # 上取整避免死循环
        c = int((keys >= mid).sum())
        if c >= k:
            lo = mid
        else:
            hi = np.uint32(int(mid) - 1)
        iterations += 1
    thr_key = lo
    thr = key_to_float(np.array([thr_key]))[0]
    gt_mask = keys > thr_key
    eq_mask = keys == thr_key
    c1 = int(gt_mask.sum())
    # 块级前缀（N_INDEX_BLOCKS 个块，块内线程步长 = 块数×线程——按块内索引升序）
    nblocks = min(N_INDEX_BLOCKS, (n + THREADS - 1) // THREADS)
    gt_counts = np.zeros(nblocks, dtype=np.int64)
    eq_counts = np.zeros(nblocks, dtype=np.int64)
    for b in range(nblocks):
        idx = np.arange(b, n, nblocks)  # 块内跨线程步进后的索引集合（块内升序）
        gt_counts[b] = int(gt_mask[idx].sum())
        eq_counts[b] = int(eq_mask[idx].sum())
    gt_prefix = np.concatenate([[0], np.cumsum(gt_counts)[:-1]])
    eq_prefix = np.concatenate([[0], np.cumsum(eq_counts)[:-1]])
    quota = k - c1
    out = np.zeros(k, dtype=np.int32)
    filled = 0
    cursor_gt = np.zeros(nblocks, dtype=np.int64)
    cursor_eq = np.zeros(nblocks, dtype=np.int64)
    eq_written_total = 0
    # 两段：先全部 >T（按块前缀），再 ==T（块前缀 + 配额截断）
    for b in range(nblocks):
        idx = np.arange(b, n, nblocks)
        sel = idx[gt_mask[idx]]
        pos = gt_prefix[b] + cursor_gt[b]
        take = len(sel)
        out[pos:pos + take] = sel
        cursor_gt[b] += take
        filled += take
    for b in range(nblocks):
        if eq_written_total >= quota:
            break
        idx = np.arange(b, n, nblocks)
        sel = idx[eq_mask[idx]]
        remaining = quota - eq_written_total
        take = min(len(sel), max(0, min(remaining, int(gt_counts[b] + eq_counts[b]) * 0 + len(sel))))
        take = min(len(sel), remaining)
        pos = c1 + eq_prefix[b] + cursor_eq[b]
        out[pos:pos + take] = sel[:take]
        cursor_eq[b] += take
        eq_written_total += take
    stats = {"threshold": float(thr), "count_gt": c1, "count_eq": int(eq_mask.sum()),
             "k_requested": k, "iterations": iterations,
             "filled": filled + eq_written_total}
    return out[:k], stats


# ---------------------------------------------------------------------------
# 3. gather split attention 仿真（ep25 骨架 + 索引间接寻址）
# ---------------------------------------------------------------------------

def emulate_gather_attention(q_abs16, q_rope16, c_kv16, k_rope16, indices, scale, splits_req=0):
    """dsa_sparse.cu 双模式 gather attention 核仿真。

    indices=None → identity 模式（dense 对照，token=i）；
    indices=int64 张量 → gather 模式（token=indices[i]）。
    输入 fp16→fp32（核内 unpack），fp32 online softmax + 累加，输出 fp16→fp32。"""
    h = q_abs16.shape[0]
    n_list = len(indices) if indices is not None else c_kv16.shape[0]
    if n_list == 0:
        return torch.zeros(h, cfg.KV_LORA_RANK)
    if splits_req > 0:
        splits = min(splits_req, MAX_SPLITS)
    else:
        splits = max(1, min((n_list + CHUNK_TARGET - 1) // CHUNK_TARGET, MAX_SPLITS))
    chunk = max(((n_list + splits - 1) // splits + WARPS - 1) // WARPS * WARPS, WARPS)

    psum = torch.zeros(h, splits, cfg.KV_LORA_RANK)
    pl = torch.zeros(h, splits)
    pm = torch.full((h, splits), -float("inf"))
    for sp in range(splits):
        i_start = sp * chunk
        i_end = min(i_start + chunk, n_list)
        for head in range(h):
            warp_m = torch.full((WARPS,), -float("inf"))
            warp_l = torch.zeros(WARPS)
            acc = torch.zeros(WARPS, cfg.KV_LORA_RANK)
            for warp in range(WARPS):
                m = -float("inf")
                l = 0.0
                a = torch.zeros(cfg.KV_LORA_RANK)
                i = i_start + warp
                while i < i_end:
                    tok = int(indices[i]) if indices is not None else i
                    dot = float((q_abs16[head] * c_kv16[tok]).sum())
                    rope = float((q_rope16[head] * k_rope16[tok]).sum())
                    score = (dot + rope) * scale
                    m_new = max(m, score)
                    alpha = 0.0 if m == -float("inf") else float(torch.exp(torch.tensor(m - m_new)))
                    p = float(torch.exp(torch.tensor(score - m_new)))
                    l = l * alpha + p
                    a = a * alpha + p * c_kv16[tok]
                    m = m_new
                    i += WARPS
                warp_m[warp] = m
                warp_l[warp] = l
                acc[warp] = a
            m_global = warp_m.max()
            weights = torch.where(warp_m == -float("inf"), torch.zeros(WARPS),
                                  torch.exp(warp_m - m_global))
            warp_l_weighted = warp_l * weights
            l_total = float(warp_l_weighted.sum())
            psum[head, sp] = (weights.unsqueeze(1) * acc).sum(dim=0)
            pl[head, sp] = l_total
            pm[head, sp] = m_global
    m_split_max = pm.max(dim=1, keepdim=True).values
    w = torch.where(pm == -float("inf"), torch.zeros_like(pm), torch.exp(pm - m_split_max))
    l_total_all = (w * pl).sum(dim=1, keepdim=True)
    inv = torch.where(l_total_all > 0, 1.0 / l_total_all, torch.zeros_like(l_total_all))
    s = ((w.unsqueeze(-1) * psum).sum(dim=1) * inv)
    return s.half().float()


# ---------------------------------------------------------------------------
# 验证主流程
# ---------------------------------------------------------------------------

def main() -> int:
    data = np.load(HERE.parent.parent / "fixtures" / "correctness.npz")
    arrays = {k: data[k] for k in data.files}
    weights = make_weights()
    # 与 fixtures 同一 x（SEED_BASE+1）
    gen = torch.Generator().manual_seed(SEED_BASE + 1)
    x = torch.randn(1, cfg.D_MODEL, generator=gen)
    freqs = rope_freqs()

    failures = []
    # grid 覆盖检查（初版 .cu 循环界 bug 的回归测试）
    for n in (33, 255, 256, 257, 2048, 65536):
        ok = check_indexer_grid_cover(n)
        status = "OK" if ok else "FAIL"
        if not ok:
            failures.append(f"grid-cover n{n}")
        print(f"[sim] indexer grid 覆盖 N={n}: {status}")
    for n in (33, 128, 2048):
        prefix = f"n{n}_"
        c_kv16 = torch.from_numpy(arrays[prefix + "c_kv"]).half().float()
        k_rope16 = torch.from_numpy(arrays[prefix + "k_rope_rot"]).half().float()
        k_i16 = torch.from_numpy(arrays[prefix + "k_i"]).half().float()
        scores_i_ref = torch.from_numpy(arrays[prefix + "index_scores_fp32"])
        q_nope, q_rope_raw = q_projection_stage(x, weights)
        q_rope = apply_rope(q_rope_raw, torch.tensor([n], dtype=torch.float32), freqs)
        q_abs = absorbed_query(q_nope, weights)
        q_i = index_query_stage(x @ weights["W_QA"], weights)
        w_used = index_weights_stage(x, weights)

        # --- 1. 索引器核仿真 vs 参考链（fp16 io / fp32 算术）---
        sim_scores = emulate_indexer_scores(q_i.half().float(), k_i16, w_used, n)
        ref_scores_fp16io = index_scores(q_i.half().float(), w_used, k_i16)
        err = float((sim_scores - ref_scores_fp16io).abs().max())
        rel = err / max(float(ref_scores_fp16io.abs().max()), 1e-9)
        status = "OK" if rel < 2e-3 else "FAIL"
        if status == "FAIL":
            failures.append(f"indexer n{n}")
        print(f"[sim] indexer N={n}: max={err:.3e} rel={rel:.2e} {status}")

        # --- 2. 阈值 topk 仿真 vs 参考选择（集合相等，整数级）---
        for k in (0, 1, 7, 33, 127, 256, n):
            sim_idx, stats = emulate_topk(sim_scores.numpy(), k)
            ref_idx = select_topk(ref_scores_fp16io, k)  # 参考链选 topk（同输入）
            # 集合比较：注意 sim 用 sim_scores（核算术）而 ref 用 torch 链分数——
            # 两者 fp32 舍入差可能改变边界并列；合同以"同分数输入"为准，此处用
            # ref 分数喂 sim 做集合相等校验，另用 sim 分数做自洽校验
            sim_idx_ref_input, _ = emulate_topk(ref_scores_fp16io.numpy(), k)
            set_ok = set(sim_idx_ref_input.tolist()) == set(ref_idx.tolist())
            status = "OK" if (set_ok and stats["filled"] == min(k, n)) else "FAIL"
            if status == "FAIL":
                failures.append(f"topk n{n} k{k}")
            print(f"[sim] topk N={n} k={k}: set_equal={set_ok} filled={stats['filled']}/{min(k, n)} "
                  f"iters={stats['iterations']} thr={stats['threshold']:.4e} {status}")

        # --- 3. gather attention 仿真 vs torch 子集 dense（fp16 io 容差）---
        for k, label in ((33, "k=33 非整块"), (127, "k=127 非整块"), (min(512, n), "k=512"), (n, "k=N 恒等")):
            idx = select_topk(ref_scores_fp16io, k)
            ref = sparse_gather_attention_bmm(q_abs.half().float(), q_rope.half().float(),
                                              c_kv16, k_rope16, idx, weights)
            for splits, slabel in ((0, "启发式"), (1, "splits=1"), (64, "splits=64")):
                s_sim = emulate_gather_attention(q_abs.half().float(), q_rope.half().float(),
                                                 c_kv16, k_rope16, idx, cfg.SOFTMAX_SCALE, splits)
                err = float((s_sim - ref["s"]).abs().max())
                rel = err / max(float(ref["s"].abs().max()), 1e-9)
                status = "OK" if rel < 2e-3 else "FAIL"
                if status == "FAIL":
                    failures.append(f"attention n{n} k{k} splits{splits}")
                print(f"[sim] attention N={n} {label} splits={slabel}: "
                      f"max={err:.3e} rel={rel:.2e} {status}")

        # identity 模式 vs dense 参考
        ref_dense = sparse_gather_attention_bmm(q_abs.half().float(), q_rope.half().float(),
                                                c_kv16, k_rope16, torch.arange(n), weights)
        s_ident = emulate_gather_attention(q_abs.half().float(), q_rope.half().float(),
                                           c_kv16, k_rope16, None, cfg.SOFTMAX_SCALE)
        err = float((s_ident - ref_dense["s"]).abs().max())
        rel = err / max(float(ref_dense["s"].abs().max()), 1e-9)
        status = "OK" if rel < 2e-3 else "FAIL"
        if status == "FAIL":
            failures.append(f"identity n{n}")
        print(f"[sim] identity(k=N) N={n}: max={err:.3e} rel={rel:.2e} {status}")

    # k=0 空集
    s0 = emulate_gather_attention(torch.zeros(16, 512), torch.zeros(16, 64),
                                  torch.zeros(4, 512), torch.zeros(4, 64),
                                  torch.zeros(0, dtype=torch.long), cfg.SOFTMAX_SCALE)
    ok0 = bool((s0 == 0).all())
    status = "OK" if ok0 else "FAIL"
    if not ok0:
        failures.append("empty set")
    print(f"[sim] 空集 k=0: 全零={ok0} {status}")

    print(f"\n[sim] {'全部通过' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
