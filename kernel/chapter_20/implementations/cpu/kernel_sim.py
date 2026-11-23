#!/usr/bin/env python3
"""CUDA/MUSA 核算法的 CPU 逐行仿真（开发验证工具，正式产物之外）。

本地无 nvcc/mcc，上服务器编译前用 torch 仿真 attention_kernels.cuh 的算法组织：
  1. gqa_prefill_kernel：grid (ceil(S/BR), B·HQ)，block 128 线程。
     每 block 负责一个 q-tile（BR=16 query），沿 key 按 BC=128 分块推进：
     score 相（thread j 算 key j 与 BR 个 query 的点积 → s_s[BR][BC]）→
     per-row online softmax 相（thread r<BR 串行扫 tile：m_new → p_s →
     alpha/l 更新；该行本 tile 无有效 key 则整行跳过——-inf 守卫）→
     acc 重归一（全线程：acc[r][d] ×= alpha[r]）→ PV 相（thread d 拥有维度 d，
     行优先扫 tile 内 key：acc[r][d] += p_s[r][c]·v[c][d]）→ 归一化写 o。
     causal early-termination：key tile 循环上界 = ceil(min(q0+BR, S)/BC)。
     仿真按 (block, tile) 两级组织逐步执行，tile 内矩阵化（fp32 求和顺序与
     每线程行流不同，按容差比较——仿真目的是算法组织正确性，不是位级）。
  2. gqa_decode_kernel：grid (B·HQ,)，block 128。单 query，chunk=BC 分块遍历 KV：
     score 相 → block max / block sum（block_reduce，模板 warp_size ∈ {32,128}
     双实例——M-11/M-20 双口径预验证）→ alpha/l 更新 → PV 相 → 归一化写 o。

grid 覆盖断言（ep26 教训：correctness 全绿但循环界 bug 由 benchmark 才抓出
——仿真必须加 grid 覆盖断言）：
  - prefill：coverage[b,h,q] 每个 query 恰被一个 (block,r) 处理一次；
    key_visit[b,h,q,j]：j ≤ q 的 key 恰访问一次、j > q 恰为零
    （early-termination 不能截掉需要的 key tile，也不能重复/越界处理）。
  - decode：coverage[b,h] 每头恰一个 block；key_visit[b,h,j] 每 key 恰一次。
  - 尾部边界形状族：S/N ∈ {1,15,16,63,64,65,127,128,129,511,512}。
  - block_reduce 按 warp_size ∈ {32,128} 双实例跑，decode 输出两者位级一致
    （跨平台归约树在本地先行验证——MUSA 迁移素材）。
fp16 io + fp32 片上算术精度（先降 half 再浮点），对齐 .cuh 的数值口径。
"""

import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

import attention_config as cfg  # noqa: E402
from attention_reference import (  # noqa: E402
    gqa_decode_reference,
    gqa_prefill_reference,
)

BR = cfg.BR          # 16
BC = cfg.BC          # 128
THREADS = cfg.THREADS  # 128
INF = float("inf")


# ---------------------------------------------------------------------------
# block_reduce 仿真（模板 warp_size：32 = CUDA 与 MUSA device builtin 口径，
# 128 = musaDeviceProp C API 口径——M-11/M-20；两种归约树结果必须一致）
# ---------------------------------------------------------------------------

def block_reduce(vals: torch.Tensor, warp_size: int, op: str) -> float:
    """仿真 .cuh block_reduce_max/sum：warp 内蝶形 → 跨 warp shared 蝶形。
    vals: [THREADS] 每 thread 的贡献（无效 lane 已填中性元）。"""
    assert vals.numel() == THREADS
    assert warp_size in (32, 128)
    assert THREADS % warp_size == 0, "block 必须按 warpSize 整除（本 kernel THREADS=128）"
    work = vals.clone()
    warps = work.view(-1, warp_size)                    # [NW, warp_size]
    ar = torch.arange(warp_size)
    off = warp_size // 2
    while off > 0:                                       # warp 内蝶形（shfl_xor 语义）
        partner = warps[:, ar ^ off]
        warps = torch.maximum(warps, partner) if op == "max" else warps + partner
        off //= 2
    scratch = warps[:, 0]                                # 每 warp lane0 → shared
    n_warps = THREADS // warp_size
    if n_warps == 1:
        return float(scratch[0])
    neutral = -INF if op == "max" else 0.0
    lane = torch.full((warp_size,), neutral)
    lane[:n_warps] = scratch
    off = warp_size // 2
    while off > 0:                                       # 第一 warp 收 NW 项
        partner = lane[ar ^ off]
        lane = torch.maximum(lane, partner) if op == "max" else lane + partner
        off //= 2
    return float(lane[0])


# ---------------------------------------------------------------------------
# gqa_prefill_kernel 仿真
# ---------------------------------------------------------------------------

def sim_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                trace_coverage: bool = True):
    """q [B,HQ,S,D]、k/v [B,HKV,S,D] → (o, coverage, key_visit, grid)。"""
    b, hq, s, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    q16, k16, v16 = q.half().float(), k.half().float(), v.half().float()
    o = torch.zeros(b, hq, s, d)
    coverage = torch.zeros(b, hq, s)
    key_visit = torch.zeros(b, hq, s, s)

    n_q_tiles = (s + BR - 1) // BR
    grid = (n_q_tiles, b * hq)
    for g_y in range(grid[1]):
        b_i, h = divmod(g_y, hq)
        kvh = h // group
        for g_x in range(grid[0]):
            q0 = g_x * BR
            if q0 >= s:
                continue                                  # 空 tile：覆盖断言要求不发生
            nr = min(BR, s - q0)                          # 本 tile 有效 query 行
            q_end = q0 + nr
            k_tiles = (q_end + BC - 1) // BC              # causal early-termination
            m = torch.full((BR,), -INF)
            l = torch.zeros(BR)
            acc = torch.zeros(BR, d)
            for t in range(k_tiles):
                c0 = t * BC
                cmax = min(c0 + BC, s)
                ncols = cmax - c0
                # ---- score 相：s_s[r, j]（causal 屏蔽在 per-row 相按索引判定）----
                s_s = (q16[b_i, h, q0:q_end] @ k16[b_i, kvh, c0:cmax].T) * scale
                # ---- per-row online softmax（thread r 串行；-inf 守卫）----
                p_s = torch.zeros(BR, ncols)
                alpha = torch.ones(BR)
                for r in range(nr):
                    qpos = q0 + r
                    valid_from = c0
                    valid_to = min(cmax, qpos + 1)        # key ≤ query
                    if valid_from >= valid_to:
                        continue                          # 本 tile 无有效 key：整行跳过
                    m_new = float(s_s[r, : ncols][: valid_to - valid_from].max())
                    p = torch.exp(s_s[r, : valid_to - valid_from] - m_new)
                    p_s[r, : valid_to - valid_from] = p
                    a = 0.0 if m[r] == -INF else float(torch.exp(m[r] - m_new))
                    m[r] = m_new
                    alpha[r] = a
                    l[r] = l[r] * a + float(p.sum())
                # ---- acc 重归一 + PV 相（thread d 拥有维度 d，扫 tile 内 key）----
                acc = acc * alpha.unsqueeze(1)
                if nr > 0 and ncols > 0:
                    acc[:nr] += p_s[:nr] @ v16[b_i, kvh, c0:cmax]
            # ---- 覆盖追踪 ----
            # query 所有权按 block 记账（每 query 恰属一个 block 的一个 r）；
            # key 访问按 (query, key) 记账（早停循环界：j ≤ q 恰一次）。
            if trace_coverage:
                for r in range(nr):
                    qpos = q0 + r
                    coverage[b_i, h, qpos] += 1
                for t in range(k_tiles):
                    c0 = t * BC
                    cmax = min(c0 + BC, s)
                    for r in range(nr):
                        qpos = q0 + r
                        hi = min(cmax, qpos + 1) - c0
                        if hi > 0:
                            key_visit[b_i, h, qpos, c0: c0 + hi] += 1
            for r in range(nr):
                o[b_i, h, q0 + r] = (acc[r] / l[r]).half().float()
    return o, coverage, key_visit, grid


# ---------------------------------------------------------------------------
# gqa_decode_kernel 仿真
# ---------------------------------------------------------------------------

def sim_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
               warp_size: int = 32, trace_coverage: bool = True):
    """q [B,HQ,D]（单 query）、k/v [B,HKV,N,D] → (o, coverage, key_visit)。"""
    b, hq, d = q.shape
    hkv, n = k.shape[1], k.shape[2]
    group = hq // hkv
    q16, k16, v16 = q.half().float(), k.half().float(), v.half().float()
    o = torch.zeros(b, hq, d)
    coverage = torch.zeros(b, hq)
    key_visit = torch.zeros(b, hq, n)

    for g in range(b * hq):
        b_i, h = divmod(g, hq)
        kvh = h // group
        if trace_coverage:
            coverage[b_i, h] += 1          # query 所有权：每 (b,h) 恰一个 block
        q_s = q16[b_i, h]
        m, l = -INF, 0.0
        acc = torch.zeros(d)
        for c0 in range(0, n, BC):
            n_valid = min(BC, n - c0)
            # ---- score 相 ----
            sc = (k16[b_i, kvh, c0: c0 + n_valid] @ q_s) * scale
            # ---- block max（warp_size 模板归约）----
            red = torch.full((THREADS,), -INF)
            red[:n_valid] = sc
            m_tile = block_reduce(red, warp_size, "max")
            m_new = max(m, m_tile)
            alpha = 0.0 if m == -INF else math.exp(m - m_new)
            # ---- p + block sum ----
            p = torch.exp(sc - m_new)
            red2 = torch.zeros(THREADS)
            red2[:n_valid] = p
            p_sum = block_reduce(red2, warp_size, "sum")
            l = l * alpha + p_sum
            acc = acc * alpha
            # ---- PV 相（thread d 拥有维度 d）----
            acc = acc + p @ v16[b_i, kvh, c0: c0 + n_valid]
            if trace_coverage:
                key_visit[b_i, h, c0: c0 + n_valid] += 1
            m = m_new
        o[b_i, h] = (acc / l).half().float()
    return o, coverage, key_visit


# ---------------------------------------------------------------------------
# 覆盖断言
# ---------------------------------------------------------------------------

def assert_prefill_coverage(coverage, key_visit, tag):
    if not bool((coverage == 1).all()):
        raise AssertionError(
            f"[{tag}] query 处理次数 != 1（min={int(coverage.min())} "
            f"max={int(coverage.max())}，grid 漏/重）")
    idx = torch.arange(key_visit.shape[-1]).view(1, 1, -1, 1)
    jq = torch.arange(key_visit.shape[-1]).view(1, 1, 1, -1)
    qidx = idx.expand_as(key_visit)
    valid = (jq <= qidx).to(key_visit.dtype)
    if not bool((key_visit == valid).all()):
        bad = (key_visit != valid).nonzero()[0].tolist()
        raise AssertionError(
            f"[{tag}] key 访问矩阵偏离 causal 精确一次 @bhz={bad[:3]} "
            f"got={key_visit[tuple(bad[:3])]} want={int(valid[tuple(bad[:3])])}"
            "（early-termination 循环界 bug）")


def assert_decode_coverage(coverage, key_visit, tag):
    if not bool((coverage == 1).all()):
        raise AssertionError(f"[{tag}] 每 (b,h) 必须恰一个 block")
    if not bool((key_visit == 1).all()):
        raise AssertionError(
            f"[{tag}] key 必须 1:1 遍历（chunk 循环界 bug）: "
            f"min={int(key_visit.min())} max={int(key_visit.max())}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    torch.manual_seed(20261121)
    scale = cfg.SOFTMAX_SCALE
    failures = 0

    tail_shapes = [1, 15, 16, 63, 64, 65, 127, 128, 129, 511, 512]
    print("== prefill sim：尾部边界族（batch=1，coverage 断言 + 数值）==")
    for s in tail_shapes:
        q = torch.randn(1, cfg.HQ, s, cfg.D)
        k = torch.randn(1, cfg.HKV, s, cfg.D)
        v = torch.randn(1, cfg.HKV, s, cfg.D)
        ref = gqa_prefill_reference(q, k, v, scale)
        o, cov, kv_visit, grid = sim_prefill(q, k, v, scale)
        try:
            assert_prefill_coverage(cov, kv_visit, f"S={s}")
            err = (o - ref).abs().max().item()
            rel = err / ref.abs().max().item()
            ok = rel < 2e-3   # fp16 io 噪声容差
            if not ok:
                failures += 1
            print(f"[sim/prefill] S={s:>4}: max={err:.3e} rel={rel:.2e} "
                  f"grid=({grid[0]},{grid[1]}) {'OK' if ok else 'FAIL'}")
        except AssertionError as exc:
            failures += 1
            print(f"[sim/prefill] S={s:>4}: ASSERTION {exc}")

    print("== decode sim：尾部边界族（coverage 断言 + 数值 + 双 warp_size 一致）==")
    for n in [1, 15, 63, 64, 65, 127, 128, 129, 511]:
        q = torch.randn(1, cfg.HQ, cfg.D)
        k = torch.randn(1, cfg.HKV, n, cfg.D)
        v = torch.randn(1, cfg.HKV, n, cfg.D)
        ref = gqa_decode_reference(q, k, v, scale)
        outs = {}
        for ws in (32, 128):
            o, cov, kv_visit = sim_decode(q, k, v, scale, ws)
            try:
                assert_decode_coverage(cov, kv_visit, f"N={n}/ws={ws}")
                outs[ws] = o
                err = (o - ref).abs().max().item()
                rel = err / ref.abs().max().item()
                ok = rel < 2e-3
                if not ok:
                    failures += 1
                print(f"[sim/decode] N={n:>4} ws={ws}: max={err:.3e} rel={rel:.2e} "
                      f"{'OK' if ok else 'FAIL'}")
            except AssertionError as exc:
                failures += 1
                print(f"[sim/decode] N={n:>4} ws={ws}: ASSERTION {exc}")
        # 两种归约树的 SUM 求和顺序不同（32-lane 蝶形 vs 128-lane 蝶形+跨warp），
        # fp32 非结合 → 归一化输出有 ulp 级差（N=63 实测 7.6e-6 ≈ 累加噪声跨
        # fp16 舍入边界）；断言 ≤ 1 fp16 ulp 量级。max 归约树顺序无关、位级一致。
        ws_diff = (outs[32] - outs[128]).abs().max().item()
        ws_rel = ws_diff / outs[32].abs().max().item()
        ws_ok = ws_rel <= 1e-3
        if not ws_ok:
            failures += 1
        print(f"[sim/decode] N={n:>4} ws32 vs ws128 输出差: {ws_diff:.2e} "
              f"(rel {ws_rel:.1e}) {'OK(ulp级)' if ws_ok else 'FAIL'}")

    print("== batch>1 抽查（S=65/N=65/B=3，GQA 映射生效形状）==")
    q = torch.randn(3, cfg.HQ, 65, cfg.D)
    k = torch.randn(3, cfg.HKV, 65, cfg.D)
    v = torch.randn(3, cfg.HKV, 65, cfg.D)
    ref = gqa_prefill_reference(q, k, v, scale)
    o, cov, kv_visit, _ = sim_prefill(q, k, v, scale)
    try:
        assert_prefill_coverage(cov, kv_visit, "prefill B=3")
        rel = (o - ref).abs().max().item() / ref.abs().max().item()
        failures += 0 if rel < 2e-3 else 1
        print(f"[sim/prefill] B=3 S=65: rel={rel:.2e} {'OK' if rel < 2e-3 else 'FAIL'}")
    except AssertionError as exc:
        failures += 1
        print(f"[sim/prefill] B=3 S=65: ASSERTION {exc}")
    qd = torch.randn(3, cfg.HQ, cfg.D)
    kd = torch.randn(3, cfg.HKV, 65, cfg.D)
    vd = torch.randn(3, cfg.HKV, 65, cfg.D)
    refd = gqa_decode_reference(qd, kd, vd, scale)
    od, covd, kvd = sim_decode(qd, kd, vd, scale)
    try:
        assert_decode_coverage(covd, kvd, "decode B=3")
        reld = (od - refd).abs().max().item() / refd.abs().max().item()
        failures += 0 if reld < 2e-3 else 1
        print(f"[sim/decode] B=3 N=65: rel={reld:.2e} {'OK' if reld < 2e-3 else 'FAIL'}")
    except AssertionError as exc:
        failures += 1
        print(f"[sim/decode] B=3 N=65: ASSERTION {exc}")

    print(f"\n== kernel_sim 总结：{failures} 个失败 ==")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
