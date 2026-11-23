#!/usr/bin/env python3
"""CUDA/MUSA paged kernel 算法的 CPU 逐行仿真（开发验证工具，正式产物之外）。

本地无 nvcc/mcc，上服务器编译前用 torch 仿真 paged_kernels.cuh 的算法组织：
  1. paged_decode_kernel：grid (B·HQ,)，block THREADS=128，单 query，
     chunk=BC=128 分块遍历逻辑 KV；**块表寻址**：token j → tb=j//bs,
     ot=j%bs, p=block_table[b, tb]，K/V 行 = pool[p, kvh, ot, :]（D 连续）。
     score 相（thread j 一行 + 表查找）→ block max / block sum（block_reduce，
     warp_size ∈ {32,128} 双实例，M-11/M-20 双口径预验证）→ alpha/l 更新 →
     PV 相（thread d 拥有维度 d，逐 token 表查找）→ 归一化写 o。
     每请求独立 seq_len（从 seq_lens[b] 读，引擎口径）。
  2. contig_decode_kernel：同核去块表化（控制组）——除地址计算外逐行相同。
  3. paged_prefill_kernel（可选对照）：ep21 prefill 结构（tile BR=16 +
     online softmax + causal early-termination）+ K/V 行块表寻址。

grid 覆盖断言（ep26 教训制度化，ep21 模式沿用 + 分页扩展）：
  - decode：coverage[b,h] 每 (b,h) 恰一个 block；token_visit[b,h,t] ∈ {0,1}
    且 t < seq_len_b 恰一次、t ≥ seq_len_b 恰零次（变长边界）；
    table_touch[b, tb]：tb < ceil(seq_len_b/bs) 恰被读、padding（-1 区）零读。
  - prefill：query 1:1、key causal 精确一次、padding 表项零读。
  - 位级断言：paged sim 与 contig sim 在同一逻辑 KV（恒等表==连续）输出
    位级一致（fp16 io + fp32 片上；值透明性——随机表亦然，因为求和序不变）。
尾部边界族：seq_len ∈ {1,15,16,17,63,127,128,129,255,256,511} × bs ∈
  {16,64,128}（部分块/恰整块/跨多块）。
"""

import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

import paged_config as cfg  # noqa: E402
from paged_kv_reference import (  # noqa: E402
    build_block_table,
    contiguous_decode_reference,
    contiguous_prefill_reference,
    gather_from_pool,
    scatter_to_pool,
)

BC = 128       # decode chunk = key tile 宽 = block 线程数（ch21 同参数）
BR = 16        # prefill query tile 高
THREADS = 128
INF = float("inf")


# ---------------------------------------------------------------------------
# block_reduce 仿真（ch21 同式：warp 蝶形 → 跨 warp shared 蝶形）
# ---------------------------------------------------------------------------

def block_reduce(vals: torch.Tensor, warp_size: int, op: str) -> float:
    assert vals.numel() == THREADS
    assert warp_size in (32, 128)
    assert THREADS % warp_size == 0
    work = vals.clone()
    warps = work.view(-1, warp_size)
    ar = torch.arange(warp_size)
    off = warp_size // 2
    while off > 0:
        partner = warps[:, ar ^ off]
        warps = torch.maximum(warps, partner) if op == "max" else warps + partner
        off //= 2
    scratch = warps[:, 0]
    n_warps = THREADS // warp_size
    if n_warps == 1:
        return float(scratch[0])
    neutral = -INF if op == "max" else 0.0
    lane = torch.full((warp_size,), neutral)
    lane[:n_warps] = scratch
    off = warp_size // 2
    while off > 0:
        partner = lane[ar ^ off]
        lane = torch.maximum(lane, partner) if op == "max" else lane + partner
        off //= 2
    return float(lane[0])


def pool_row(k_pool: torch.Tensor, table_row: torch.Tensor, b: int, kvh: int,
             j: int, bs: int) -> torch.Tensor:
    """kernel 的行地址解析（合同映射）：token j → pool[p, kvh, ot, :]。"""
    p = int(table_row[j // bs])
    ot = j % bs
    return k_pool[p, kvh, ot]


# ---------------------------------------------------------------------------
# paged decode kernel 仿真
# ---------------------------------------------------------------------------

def sim_decode(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
               block_table: torch.Tensor, seq_lens: list[int], bs: int,
               scale: float, warp_size: int = 32, use_table: bool = True,
               trace: bool = True):
    """paged/contig decode 仿真。use_table=False 时为 contig 控制组（K/V 按
    逻辑连续布局寻址：调用方传 gather 后的 cache + 单位表口径）。

    返回 (o [B,HQ,D], coverage, token_visit, table_touch)。
    """
    bsz, hq, d = q.shape
    hkv = k_pool.shape[1]
    group = hq // hkv
    max_blocks = block_table.shape[1]
    q16 = q.half().float()
    k16, v16 = k_pool.half().float(), v_pool.half().float()
    o = torch.zeros(bsz, hq, d)
    coverage = torch.zeros(bsz, hq)
    token_visit = torch.zeros(bsz, hq, max(seq_lens))
    table_touch = torch.zeros(bsz, max_blocks)

    for g in range(bsz * hq):
        b_i, h = divmod(g, hq)
        kvh = h // group
        n = seq_lens[b_i]
        coverage[b_i, h] += 1
        q_s = q16[b_i, h]
        table_row = block_table[b_i]
        m, ll = -INF, 0.0
        acc = torch.zeros(d)
        for c0 in range(0, n, BC):
            nv = min(BC, n - c0)
            sc = torch.zeros(THREADS)
            for j in range(nv):                       # thread j (j < nv)
                tok = c0 + j
                if use_table:
                    krow = pool_row(k16, table_row, b_i, kvh, tok, bs)
                    if trace:
                        table_touch[b_i, tok // bs] += 1
                else:
                    krow = k16[b_i, kvh, tok]
                sc[j] = float((krow @ q_s) * scale)
            m_tile = block_reduce(sc, warp_size, "max")
            m_new = max(m, m_tile)
            alpha = 0.0 if m == -INF else math.exp(m - m_new)
            p = torch.exp((sc[:nv] - m_new))
            red = torch.zeros(THREADS)
            red[:nv] = p
            p_sum = block_reduce(red, warp_size, "sum")
            ll = ll * alpha + p_sum
            acc = acc * alpha
            for c in range(nv):                       # PV：thread d 逐 token
                tok = c0 + c
                if use_table:
                    vrow = pool_row(v16, table_row, b_i, kvh, tok, bs)
                else:
                    vrow = v16[b_i, kvh, tok]
                acc = acc + p[c] * vrow
            if trace:
                token_visit[b_i, h, c0: c0 + nv] += 1
            m = m_new
        o[b_i, h] = (acc / ll).half().float()
    return o, coverage, token_visit, table_touch


# ---------------------------------------------------------------------------
# paged prefill kernel 仿真（可选对照；ep21 prefill 结构 + 表寻址）
# ---------------------------------------------------------------------------

def sim_prefill(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
                block_table: torch.Tensor, s_len: int, bs: int, scale: float,
                trace: bool = True):
    """q [B,HQ,S,D]（连续布局——Q 不是 cache），KV 在池。返回 (o, coverage,
    key_visit, table_touch)。"""
    bsz, hq, s, d = q.shape
    hkv = k_pool.shape[1]
    group = hq // hkv
    max_blocks = block_table.shape[1]
    q16 = q.half().float()
    k16, v16 = k_pool.half().float(), v_pool.half().float()
    o = torch.zeros(bsz, hq, s, d)
    coverage = torch.zeros(bsz, hq, s)
    key_visit = torch.zeros(bsz, hq, s, s)
    table_touch = torch.zeros(bsz, max_blocks)

    n_q_tiles = (s + BR - 1) // BR
    for g_y in range(bsz * hq):
        b_i, h = divmod(g_y, hq)
        kvh = h // group
        table_row = block_table[b_i]
        for g_x in range(n_q_tiles):
            q0 = g_x * BR
            if q0 >= s:
                continue
            q_end = min(q0 + BR, s)
            nr = q_end - q0
            k_tiles = (q_end + BC - 1) // BC
            m = torch.full((BR,), -INF)
            ll = torch.zeros(BR)
            acc = torch.zeros(BR, d)
            for t in range(k_tiles):
                c0 = t * BC
                cmax = min(c0 + BC, s)
                ncols = cmax - c0
                s_s = torch.zeros(BR, BC)
                for j in range(ncols):                # thread j：key 列 + 表查找
                    tok = c0 + j
                    krow = pool_row(k16, table_row, b_i, kvh, tok, bs)
                    if trace:
                        table_touch[b_i, tok // bs] += 1
                    s_s[:nr, j] = (q16[b_i, h, q0:q_end] @ krow) * scale
                p_s = torch.zeros(BR, ncols)
                alpha = torch.ones(BR)
                for r in range(nr):                   # per-row online softmax
                    qpos = q0 + r
                    valid_to = min(cmax, qpos + 1)
                    if c0 >= valid_to:
                        continue
                    m_new = float(s_s[r, : valid_to - c0].max())
                    p = torch.exp(s_s[r, : valid_to - c0] - m_new)
                    p_s[r, : valid_to - c0] = p
                    a = 0.0 if m[r] == -INF else float(torch.exp(m[r] - m_new))
                    m[r] = m_new
                    alpha[r] = a
                    ll[r] = ll[r] * a + float(p.sum())
                acc = acc * alpha.unsqueeze(1)
                for c in range(ncols):                # PV：thread d 维度
                    tok = c0 + c
                    vrow = pool_row(v16, table_row, b_i, kvh, tok, bs)
                    for r in range(nr):
                        if tok <= q0 + r:
                            acc[r] += p_s[r, c] * vrow
                if trace:
                    for r in range(nr):
                        qpos = q0 + r
                        hi = min(cmax, qpos + 1) - c0
                        if hi > 0:
                            key_visit[b_i, h, qpos, c0: c0 + hi] += 1
            for r in range(nr):
                o[b_i, h, q0 + r] = (acc[r] / ll[r]).half().float()
            if trace:
                for r in range(nr):
                    coverage[b_i, h, q0 + r] += 1
    return o, coverage, key_visit, table_touch


# ---------------------------------------------------------------------------
# 覆盖断言
# ---------------------------------------------------------------------------

def assert_decode_coverage(coverage, token_visit, seq_lens, table_touch,
                           seq_lens_full, bs, tag):
    if not bool((coverage == 1).all()):
        raise AssertionError(f"[{tag}] 每 (b,h) 必须恰一个 block")
    for b_i, n in enumerate(seq_lens):
        vis = token_visit[b_i, :, :n]
        if not bool((vis == 1).all()):
            raise AssertionError(
                f"[{tag}] b={b_i} token 必须精确一次遍历（chunk 循环界 bug）: "
                f"min={int(vis.min())} max={int(vis.max())}")
        if n < token_visit.shape[-1] and not bool((token_visit[b_i, :, n:] == 0).all()):
            raise AssertionError(f"[{tag}] b={b_i} padding 区被访问（越界读）")
        nb = (n + bs - 1) // bs
        tt = table_touch[b_i, :nb]
        if not bool((tt >= 1).all()):
            raise AssertionError(f"[{tag}] b={b_i} 有效块未被读取")
        if not bool((table_touch[b_i, nb:] == 0).all()):
            raise AssertionError(
                f"[{tag}] b={b_i} padding 表项被读取（空块越界）")


def assert_prefill_coverage(coverage, key_visit, table_touch, s, bs, tag):
    if not bool((coverage == 1).all()):
        raise AssertionError(f"[{tag}] query 处理次数 != 1")
    idx = torch.arange(key_visit.shape[-1]).view(1, 1, -1, 1)
    jq = torch.arange(key_visit.shape[-1]).view(1, 1, 1, -1)
    valid = (jq <= idx.expand_as(key_visit)).to(key_visit.dtype)
    if not bool((key_visit == valid).all()):
        raise AssertionError(f"[{tag}] key 访问矩阵偏离 causal 精确一次")
    nb = (s + bs - 1) // bs
    if not bool((table_touch[:, nb:] == 0).all()):
        raise AssertionError(f"[{tag}] padding 表项被读取（空块越界）")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    torch.manual_seed(20261122)
    scale = cfg.SOFTMAX_SCALE
    failures = 0

    print("== decode sim：尾部边界族 × bs（coverage 断言 + 数值 + 位级透明性）==")
    tail = [1, 15, 16, 17, 63, 127, 128, 129, 255, 256, 511]
    for bs in (16, 64, 128):
        for n in tail:
            nb_total = (n + bs - 1) // bs
            q = torch.randn(1, cfg.HQ, cfg.D)
            k = torch.randn(1, cfg.HKV, n, cfg.D)
            v = torch.randn_like(k)
            ref = contiguous_decode_reference(q, k, v, [n], scale)
            # 恒等表（等价连续布局）
            table = build_block_table([n], bs, nb_total, "identity", seed=1)
            kp, vp = scatter_to_pool(k, v, table, [n], bs, nb_total)
            o_p, cov, tv, tt = sim_decode(q, kp, vp, table, [n], bs, scale, 32)
            # contig 控制组（同核去块表化）：直接寻址连续 cache 本体
            o_c, _, _, _ = sim_decode(q, k, v, table, [n], bs, scale, 32,
                                      use_table=False)
            # 随机表
            num_blocks = nb_total + 3
            table_r = build_block_table([n], bs, num_blocks, "rand", seed=42)
            kp2, vp2 = scatter_to_pool(k, v, table_r, [n], bs, num_blocks)
            o_r, cov_r, tv_r, tt_r = sim_decode(q, kp2, vp2, table_r, [n], bs,
                                                scale, 32)
            try:
                assert_decode_coverage(cov, tv, [n], tt, [n], bs, f"id N={n}/bs={bs}")
                assert_decode_coverage(cov_r, tv_r, [n], tt_r, [n], bs,
                                       f"rand N={n}/bs={bs}")
                err = (o_p - ref).abs().max().item()
                rel = err / ref.abs().max().item()
                bit_pc = bool(torch.equal(o_p, o_c))    # paged vs contig 位级
                bit_pr = bool(torch.equal(o_p, o_r))    # identity vs random 位级
                ok = rel < 2e-3 and bit_pc and bit_pr
                if not ok:
                    failures += 1
                print(f"[sim/decode] bs={bs:>3} N={n:>4}: rel={rel:.2e} "
                      f"paged==contig(bit)={bit_pc} id==rand(bit)={bit_pr} "
                      f"{'OK' if ok else 'FAIL'}")
            except AssertionError as exc:
                failures += 1
                print(f"[sim/decode] bs={bs:>3} N={n:>4}: ASSERTION {exc}")

    print("== decode sim：双 warp_size 归约树一致 + 变长 batch（引擎口径）==")
    for n in (129, 511):
        bs = 16
        nb_total = (n + bs - 1) // bs
        q = torch.randn(1, cfg.HQ, cfg.D)
        k = torch.randn(1, cfg.HKV, n, cfg.D)
        v = torch.randn_like(k)
        table = build_block_table([n], bs, nb_total, "identity", seed=3)
        kp, vp = scatter_to_pool(k, v, table, [n], bs, nb_total)
        ref = contiguous_decode_reference(q, k, v, [n], scale)
        o32, _, _, _ = sim_decode(q, kp, vp, table, [n], bs, scale, 32)
        o128, _, _, _ = sim_decode(q, kp, vp, table, [n], bs, scale, 128)
        rel32 = (o32 - ref).abs().max().item() / ref.abs().max().item()
        diff = (o32 - o128).abs().max().item() / o32.abs().max().item()
        ok = rel32 < 2e-3 and diff <= 1e-3
        if not ok:
            failures += 1
        print(f"[sim/decode] N={n} ws32-vs-ws128 rel_diff={diff:.2e} {'OK' if ok else 'FAIL'}")
    # 变长 batch（每请求独立 seq_len；max_blocks 按 max 对齐 + padding -1）
    seq_lens = [1, 17, 255, 511]
    bs = 32
    maxb = max((n + bs - 1) // bs for n in seq_lens)
    nb_pool = sum((n + bs - 1) // bs for n in seq_lens)
    q = torch.randn(len(seq_lens), cfg.HQ, cfg.D)
    k = torch.randn(len(seq_lens), cfg.HKV, max(seq_lens), cfg.D)
    v = torch.randn_like(k)
    table = build_block_table(seq_lens, bs, nb_pool, "per_request_identity", seed=5)
    kp, vp = scatter_to_pool(k, v, table, seq_lens, bs, nb_pool)
    table = torch.cat([table,
                       torch.full((len(seq_lens), maxb - table.shape[1]),
                                  cfg.TABLE_PAD, dtype=torch.int32)], dim=1)
    ref = contiguous_decode_reference(q, k, v, seq_lens, scale)
    o_v, cov, tv, tt = sim_decode(q, kp, vp, table, seq_lens, bs, scale, 32)
    try:
        assert_decode_coverage(cov, tv, seq_lens, tt, seq_lens, bs, "varlen")
        rel = (o_v - ref).abs().max().item() / ref.abs().max().item()
        ok = rel < 2e-3
        if not ok:
            failures += 1
        print(f"[sim/decode] varlen {seq_lens} bs={bs}: rel={rel:.2e} "
              f"{'OK' if ok else 'FAIL'}")
    except AssertionError as exc:
        failures += 1
        print(f"[sim/decode] varlen: ASSERTION {exc}")

    print("== prefill paged sim：causal 覆盖 + padding 零读 ==")
    for s_len, bs in ((65, 16), (128, 128), (200, 64)):
        nb = (s_len + bs - 1) // bs
        q = torch.randn(1, cfg.HQ, s_len, cfg.D)
        k = torch.randn(1, cfg.HKV, s_len, cfg.D)
        v = torch.randn_like(k)
        ref = contiguous_prefill_reference(q, k, v, scale)
        table = build_block_table([s_len], bs, nb + 2, "rand", seed=7)
        kp, vp = scatter_to_pool(k, v, table, [s_len], bs, nb + 2)
        o, cov, kv_visit, tt = sim_prefill(q, kp, vp, table, s_len, bs, scale)
        try:
            assert_prefill_coverage(cov, kv_visit, tt, s_len, bs, f"prefill S={s_len}")
            rel = (o - ref).abs().max().item() / ref.abs().max().item()
            ok = rel < 2e-3
            if not ok:
                failures += 1
            print(f"[sim/prefill] S={s_len} bs={bs}: rel={rel:.2e} "
                  f"{'OK' if ok else 'FAIL'}")
        except AssertionError as exc:
            failures += 1
            print(f"[sim/prefill] S={s_len} bs={bs}: ASSERTION {exc}")

    print(f"\n== kernel_sim 总结：{failures} 个失败 ==")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
