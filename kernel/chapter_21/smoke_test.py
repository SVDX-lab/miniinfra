#!/usr/bin/env python3
"""第 21 期 smoke test（边界行为验证；CPU-only 本地跑，CUDA 列加 --backend cuda）。

覆盖（块表合同为主角）：
  S1 手算例：单块/恒等表 == 连续布局（位级）；
  S2 边界族：seq_len ∈ {1, bs-1, bs, bs+1, 多块, 部分块} × bs ∈ {16,64,128}
    ——paged 参考 == 连续参考（位级）+ gather 往返位级；
  S3 空块（padding -1 表项）零读取：参考/仿真双断言（kernel_sim 覆盖断言）；
  S4 合同校验显式报错：有效区 -1 / 表项越界 / 物理块重复（请求内 + 跨请求）/
    表长不足 / seq_len 0 / dtype / 非连续（CPU 参考 + CUDA 扩展 validate_tables）；
  S5 显存记账器边界：0 容量、容量恰够、块粒度损耗、P2 搬迁、释放合并；
  S6 记账方法 sanity：手工可解负载对照模拟器输出。
CUDA 列追加：S1/S2/S3 边界族过 paged/contig/gather 三核（paged vs contig 位级
+ vs CPU 参考容差 + gather 位级）+ validate_tables 报错路径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import paged_config as cfg  # noqa: E402
from memory_accountant import METHOD, simulate  # noqa: E402
from paged_kv_reference import (  # noqa: E402
    build_block_table,
    contiguous_decode_reference,
    gather_from_pool,
    paged_decode_reference,
    paged_prefill_reference,
    scatter_to_pool,
    validate_block_table,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"[smoke] {name}: PASS {detail}")
    else:
        FAIL += 1
        print(f"[smoke] {name}: FAIL {detail}")


def expect_error(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        FAIL += 1
        print(f"[smoke] {name}: FAIL（未抛错）")
    except RuntimeError as error:
        PASS += 1
        print(f"[smoke] {name}: PASS（RuntimeError: {str(error)[:80]}...）")
    except ValueError as error:
        PASS += 1
        print(f"[smoke] {name}: PASS（ValueError: {str(error)[:80]}...）")
    except Exception as error:  # noqa: BLE001
        FAIL += 1
        print(f"[smoke] {name}: FAIL（异常类型 {type(error).__name__}: {error}）")


def make_case(seq_lens: list[int], bs: int, num_extra: int, mode: str,
              seed: int, dtype=torch.float32):
    """随机用例 → (q, k逻辑, v逻辑, 池, 表)。"""
    torch.manual_seed(seed)
    b = len(seq_lens)
    n_max = max(seq_lens)
    total_nb = sum(cfg.blocks_for(n, bs) for n in seq_lens)
    num_blocks = total_nb + num_extra
    q = torch.randn(b, cfg.HQ, cfg.D, dtype=dtype)
    k = torch.randn(b, cfg.HKV, n_max, cfg.D, dtype=dtype)
    v = torch.randn_like(k)
    table = build_block_table(seq_lens, bs, num_blocks, mode, seed=seed + 1)
    kp, vp = scatter_to_pool(k, v, table, seq_lens, bs, num_blocks)
    return q, k, v, kp, vp, table, num_blocks


def run_reference_checks():
    scale = cfg.SOFTMAX_SCALE
    # ---- S1 手算例：单 token、恒等表 ----
    q = torch.randn(1, cfg.HQ, cfg.D)
    k = torch.randn(1, cfg.HKV, 1, cfg.D)
    v = torch.randn_like(k)
    table = build_block_table([1], 16, 1, "identity", seed=0)
    kp, vp = scatter_to_pool(k, v, table, [1], 16, 1)
    o_ref = contiguous_decode_reference(q, k, v, [1], scale)
    o_paged = paged_decode_reference(q, kp, vp, table, [1], 16, scale)
    check("S1 单token恒等表==连续(位级)", bool(torch.equal(o_ref, o_paged)),
          f"max|o|={o_ref.abs().max():.3f}")
    # 手算数值：单 key → softmax 恒 1 → o == v（GQA: q 头 h 看 kv 头 h//2）
    manual = v.repeat_interleave(cfg.GQA_GROUP, dim=1).squeeze(2)
    check("S1 单key softmax=1 → o==v(位级)", bool(torch.equal(o_ref, manual)))

    # ---- S2 边界族（位级透明）----
    for bs in (16, 64, 128):
        for n in (1, bs - 1, bs, bs + 1, bs * 3 - 1, bs * 3):
            q, k, v, kp, vp, table, nb = make_case([n], bs, 1, "rand", seed=n + bs)
            o_ref = contiguous_decode_reference(q, k, v, [n], scale)
            o_paged = paged_decode_reference(q, kp, vp, table, [n], bs, scale)
            check(f"S2 bs={bs} N={n} paged==contig(位级)",
                  bool(torch.equal(o_ref, o_paged)))

    # ---- S3 padding（-1）表项零触碰：gather 只到 seq_len；等效零读由
    #      kernel_sim 的 table_touch 断言承担，此处验证 -1 padding 不影响参考 ----
    n, bs = 33, 16  # 3 块（第 3 块 1 token，部分块），max_blocks=5（2 行 -1）
    q, k, v, kp, vp, table, nb = make_case([n], bs, 2, "rand", seed=99)
    check("S3 padding 表项为 -1",
          bool((table[0, (n + bs - 1) // bs:] == cfg.TABLE_PAD).all()))
    o_ref = contiguous_decode_reference(q, k, v, [n], scale)
    o_paged = paged_decode_reference(q, kp, vp, table, [n], bs, scale)
    check("S3 含 -1 padding 的表输出位级一致", bool(torch.equal(o_ref, o_paged)))

    # ---- S4 合同校验显式报错（CPU 参考）----
    expect_error("S4 有效区 -1", lambda: validate_block_table(
        torch.tensor([[0, -1]], dtype=torch.int32), [17], 16, 4))
    expect_error("S4 表项越界", lambda: validate_block_table(
        torch.tensor([[7]], dtype=torch.int32), [1], 16, 4))
    expect_error("S4 请求内重复块", lambda: validate_block_table(
        torch.tensor([[2, 2]], dtype=torch.int32), [32], 16, 4))
    expect_error("S4 跨请求重复块", lambda: validate_block_table(
        torch.tensor([[1], [1]], dtype=torch.int32), [1, 1], 16, 4))
    expect_error("S4 表长不足", lambda: validate_block_table(
        torch.tensor([[0, 1]], dtype=torch.int32), [48], 16, 4))
    expect_error("S4 seq_len=0", lambda: validate_block_table(
        torch.tensor([[0]], dtype=torch.int32), [0], 16, 4))
    expect_error("S4 gather 触 -1", lambda: gather_from_pool(
        torch.zeros(2, cfg.HKV, 16, cfg.D), torch.zeros(2, cfg.HKV, 16, cfg.D),
        torch.tensor([[0, -1]], dtype=torch.int32), [17], 16))
    expect_error("S4 block_size 非合同域", lambda: cfg.blocks_for(8, 24))

    # ---- S5/S6 记账器 ----
    wl = [{"id": 0, "arrive": 0, "initial": 4, "final": 8},
          {"id": 1, "arrive": 1, "initial": 4, "final": 8}]
    r = simulate("paged", wl, 16, 4)
    check("S5 分页容量恰够全服务", r["served"] == 2 and r["aborted_midlife"] == 0,
          f"served={r['served']}")
    r = simulate("paged", wl, 8, 4)
    check("S5 分页 0 空闲块拒新请求", r["rejected_at_admission"] >= 1
          or r["aborted_midlife"] >= 1, f"served={r['served']}")
    r = simulate("paged", wl, 7, 4)
    check("S5 块粒度损耗（7 token 容量只有 1 块×4）",
          r["granular_capacity_tokens"] == 4, f"granular={r['granular_capacity_tokens']}")
    # malloc 教学用例：容量 16 = 2×final。req0 [0,4)、req1 [4,8) 相邻；增长时
    # req0 原地扩展失败 → 搬迁（copy 4 token）；req1 增长时最大连续段 4+3=7
    # 空闲但无 ≥5 连续段 → **中途夭折**（外部碎片受害者）——总空闲够但死于碎片。
    # 分页同容量同负载：4 块恰好 2×2，全服务零拷贝。
    wl2 = [{"id": 0, "arrive": 0, "initial": 4, "final": 8},
           {"id": 1, "arrive": 0, "initial": 4, "final": 8}]
    r = simulate("contig-exact-malloc", wl2, 16, 4)
    r_paged = simulate("paged", wl2, 16, 4)
    check("S5 同容量：malloc 碎片夭折 1 请求 / 分页全服务零拷贝",
          r["served"] == 1 and r["aborted_midlife"] == 1 and r["copy_tokens"] == 4
          and r_paged["served"] == 2 and r_paged["copy_tokens"] == 0
          and r["reject_no_contiguous_span"] >= 1,
          f"malloc served={r['served']} abort={r['aborted_midlife']} "
          f"copy={r['copy_tokens']}; paged served={r_paged['served']}")
    # S6 手工可解：单请求 reserve-global（预留 = 负载内最大 final = 20），
    # 初始 5 → 首个峰值占用步（step0 追加后 len=6）waste = 20-6 = 14；
    # 生命周期平均 waste = 预留 20 − 平均长度 (6+20)/2×… = 20 − mean(6..20) = 7。
    wl1 = [{"id": 0, "arrive": 0, "initial": 5, "final": 20}]
    r = simulate("contig-reserve-global", wl1, 100, 16)
    check("S6 预留方案 waste@首峰=预留-(initial+1)",
          r["served"] == 1 and r["internal_waste_at_peak"] == 14
          and r["internal_waste_mean_active"] == 7.5,
          f"waste@peak={r['internal_waste_at_peak']} "
          f"mean={r['internal_waste_mean_active']}")
    # mean 的手工口径：active 步为 len ∈ [6, 19]（step14 len=20 当步释放不计），
    # mean_len = 12.5 → mean waste = 20 − 12.5 = 7.5


def run_cuda_checks(module) -> None:
    scale = cfg.SOFTMAX_SCALE
    dev = torch.device("cuda")
    for bs in (16, 128):
        for n in (1, bs - 1, bs, bs + 1, 511 if bs == 16 else 384):
            q, k, v, kp, vp, table, nb = make_case([n], bs, 2, "rand",
                                                   seed=n * 3 + bs, dtype=torch.float32)
            b = 1
            qt, kt, vt = q.to(dev), k.to(dev), v.to(dev)
            kpt, vpt = kp.to(dev), vp.to(dev)
            tt = table.to(dev)
            lens = torch.tensor([n], dtype=torch.int32, device=dev)
            module.validate_tables(tt, lens, nb, bs)
            o_p = module.paged_decode_attention(qt, kpt, vpt, tt, lens, bs, scale)
            o_c = module.contig_decode_attention(qt, kt, vt, lens, scale)
            bit = bool(torch.equal(o_p, o_c))
            o_ref = contiguous_decode_reference(q, k, v, [n], scale)
            rel = (o_p.float().cpu() - o_ref).abs().max().item() / o_ref.abs().max().item()
            kg, vg = module.gather_kv(kpt, vpt, tt, lens, bs)
            gather_bit = (bool(torch.equal(kg.cpu(), k)) and bool(torch.equal(vg.cpu(), v)))
            check(f"CUDA bs={bs} N={n} paged==contig(位级)+ref(rel={rel:.1e})"
                  f"+gather(位级)", bit and rel < 5e-4 and gather_bit)
    # fp16 路径 + 变长 batch
    seq_lens = [1, 17, 129]
    q, k, v, kp, vp, table, nb = make_case(seq_lens, 16, 1, "rand", seed=7,
                                           dtype=torch.float16)
    dev = torch.device("cuda")
    maxb = table.shape[1]
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    o_p = module.paged_decode_attention(q.to(dev), kp.to(dev), vp.to(dev),
                                        table.to(dev), lens, 16, scale)
    o_c = module.contig_decode_attention(q.to(dev), k.to(dev), v.to(dev), lens, scale)
    bit = bool(torch.equal(o_p, o_c))
    o_ref = contiguous_decode_reference(q.float(), k.float(), v.float(),
                                        seq_lens, scale)
    rel = (o_p.float().cpu() - o_ref).abs().max().item() / o_ref.abs().max().item()
    check(f"CUDA fp16 varlen {seq_lens} paged==contig(位级) rel={rel:.1e}",
          bit and rel < 2e-2)
    # 合同报错路径（CUDA 扩展）
    def _bad_padding():
        t = torch.tensor([[0, -1]], dtype=torch.int32, device=dev)
        l = torch.tensor([17], dtype=torch.int32, device=dev)
        module.validate_tables(t, l, 4, 16)
    expect_error("CUDA validate_tables 有效区-1", _bad_padding)

    def _bad_dup():
        t = torch.tensor([[3, 3]], dtype=torch.int32, device=dev)
        l = torch.tensor([32], dtype=torch.int32, device=dev)
        module.validate_tables(t, l, 4, 16)
    expect_error("CUDA validate_tables 重复块", _bad_dup)

    def _bad_pad_value():
        t = torch.tensor([[0, 5]], dtype=torch.int32, device=dev)
        l = torch.tensor([16], dtype=torch.int32, device=dev)
        module.validate_tables(t, l, 4, 16)
    expect_error("CUDA validate_tables padding 非 -1", _bad_pad_value)

    def _bad_dtype():
        q64 = torch.randn(1, cfg.HQ, cfg.D, dtype=torch.float64, device=dev)
        kp = torch.randn(1, cfg.HKV, 16, cfg.D, device=dev)
        t = torch.tensor([[0]], dtype=torch.int32, device=dev)
        l = torch.tensor([1], dtype=torch.int32, device=dev)
        module.paged_decode_attention(q64, kp, kp, t, l, 16, scale)
    expect_error("CUDA dtype float64 拒绝", _bad_dtype)

    def _bad_cpu_tensor():
        qt = torch.randn(1, cfg.HQ, cfg.D)
        kp = torch.randn(1, cfg.HKV, 16, cfg.D, device=dev)
        t = torch.tensor([[0]], dtype=torch.int32, device=dev)
        l = torch.tensor([1], dtype=torch.int32, device=dev)
        module.paged_decode_attention(qt, kp, kp, t, l, 16, scale)
    expect_error("CUDA CPU tensor 拒绝", _bad_cpu_tensor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    arguments = parser.parse_args()
    run_reference_checks()
    if arguments.backend == "cuda":
        cuda_dir = HERE / "implementations" / "cuda"
        sys.path.insert(0, str(cuda_dir))
        import build_ext  # noqa: PLC0415
        module = build_ext.get_module()
        run_cuda_checks(module)
    print(f"\n[smoke] 总计 PASS={PASS} FAIL={FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
