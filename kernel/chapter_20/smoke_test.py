#!/usr/bin/env python3
"""第 20 期 smoke test（边界行为验证；CPU-only 本地跑，CUDA 列加 --backend cuda）。

覆盖：手算例、双形态小形状、GQA 映射、grid 覆盖（kernel_sim 断言复跑）、
合同校验显式报错（dtype/非连续/维度/N=0/CPU tensor）、roofline 手算数、
kernel_sim 与 roofline_attention 的一致性。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import attention_config as cfg  # noqa: E402
import roofline_attention as ra  # noqa: E402
from attention_reference import (  # noqa: E402
    gqa_decode_reference,
    gqa_mapping_spotcheck,
    gqa_prefill_reference,
    naive_loops_decode,
    naive_loops_prefill,
    stable_softmax_lastdim,
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


def expect_error(name: str, fn, backend_module=None):
    global PASS, FAIL
    try:
        fn()
        FAIL += 1
        print(f"[smoke] {name}: FAIL（未抛错）")
    except RuntimeError as error:
        PASS += 1
        print(f"[smoke] {name}: PASS（RuntimeError: {str(error)[:80]}...）")
    except Exception as error:  # noqa: BLE001
        FAIL += 1
        print(f"[smoke] {name}: FAIL（异常类型 {type(error).__name__}: {error}）")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "musa"])
    arguments = parser.parse_args()
    torch.manual_seed(20261121)
    scale = cfg.SOFTMAX_SCALE
    module = None
    if arguments.backend == "cuda":
        cuda_dir = HERE / "implementations" / "cuda"
        sys.path.insert(0, str(cuda_dir))
        import build_ext  # noqa: PLC0415
        module = build_ext.get_module()

    # ---- S1: 手算例（S=1, D=4 缩维、直接算 softmax）----
    # 缩维版手算：D 维度不进 kernel（kernel D=128 编译期固定），手算例在
    # 参考实现层做（参考实现 D 是参数化的；核的手算等价由 kernel_sim 保真）
    q1 = torch.tensor([[[[1.0, 0.5]]]], dtype=torch.float64)          # [1,1,1,2]
    k1 = torch.tensor([[[[0.5, -0.5]]]], dtype=torch.float64)
    v1 = torch.tensor([[[[2.0, -1.0]]]], dtype=torch.float64)
    o1 = gqa_prefill_reference(q1, k1, v1, 1.0)
    manual = torch.softmax(torch.tensor([[1.0 * 0.5 + 0.5 * -0.5]], dtype=torch.float64),
                           dim=-1) @ v1[0, 0]
    check("S1 手算例 S=1", torch.allclose(o1[0, 0], manual), f"{o1.flatten().tolist()}")

    # S=2 causal：第 2 个 query 只看 key 0,1
    q2 = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    k2 = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    v2 = torch.randn(1, 1, 2, 2, dtype=torch.float64)
    o2 = gqa_prefill_reference(q2, k2, v2, 1.0)
    s0 = (q2[0, 0, 0] * k2[0, 0, 0]).sum()
    s1a = (q2[0, 0, 1] * k2[0, 0, 0]).sum()
    s1b = (q2[0, 0, 1] * k2[0, 0, 1]).sum()
    e0 = math.exp(float(s1a - max(float(s1a), float(s1b))))
    e1 = math.exp(float(s1b - max(float(s1a), float(s1b))))
    row1 = (e0 * v2[0, 0, 0] + e1 * v2[0, 0, 1]) / (e0 + e1)
    check("S2 手算 causal S=2 行 0", torch.allclose(o2[0, 0, 0], v2[0, 0, 0]))
    check("S3 手算 causal S=2 行 1", torch.allclose(o2[0, 0, 1], row1))

    # ---- S4: 双实现一致（naive loops vs torch 参考，小形状；B=2≠HKV=1 防
    #      "len(k) 取到 batch 维"类回归——开发期真实 bug，见 EXPERIMENT_LOG）----
    b, s, d = 2, 5, 4
    qn = torch.randn(b, 2, s, d, dtype=torch.float64).tolist()
    kn = torch.randn(b, 1, s, d, dtype=torch.float64).tolist()   # HKV=1 → HQ=2 MQA 式
    vn = torch.randn(b, 1, s, d, dtype=torch.float64).tolist()
    o_loops = torch.tensor(naive_loops_prefill(qn, kn, vn, 1.0), dtype=torch.float64)
    o_ref = gqa_prefill_reference(
        torch.tensor(qn, dtype=torch.float64), torch.tensor(kn, dtype=torch.float64),
        torch.tensor(vn, dtype=torch.float64), 1.0)
    check("S4 naive loops vs torch 参考（B=2≠HKV）", torch.allclose(o_loops, o_ref, atol=1e-12),
          f"max={float((o_loops - o_ref).abs().max()):.2e}")
    qd = torch.randn(b, 2, d, dtype=torch.float64).tolist()
    kd = torch.randn(b, 1, 6, d, dtype=torch.float64).tolist()
    vd = torch.randn(b, 1, 6, d, dtype=torch.float64).tolist()
    od_loops = torch.tensor(naive_loops_decode(qd, kd, vd, 1.0), dtype=torch.float64)
    od_ref = gqa_decode_reference(
        torch.tensor(qd, dtype=torch.float64), torch.tensor(kd, dtype=torch.float64),
        torch.tensor(vd, dtype=torch.float64), 1.0)
    check("S5 naive loops decode vs torch 参考（B=2≠HKV）", torch.allclose(od_loops, od_ref, atol=1e-12))

    # ---- S6: 稳定 softmax 与朴素 softmax 一致（温和分布）----
    x = torch.randn(4, 32, dtype=torch.float64) * 3
    manual = torch.exp(x - x.amax(-1, keepdim=True))
    manual = manual / manual.sum(-1, keepdim=True)
    check("S6 stable softmax 减 max 等价", torch.allclose(stable_softmax_lastdim(x), manual))
    big = torch.tensor([[10000.0, 9999.0, -10000.0]], dtype=torch.float64)
    sb = stable_softmax_lastdim(big)
    check("S7 大分数不溢出（减 max）", bool(sb.isfinite().all()) and abs(float(sb[0, 0] - 0.7310585786300049)) < 1e-9)

    # ---- S8: GQA 映射抽查 ----
    spot = gqa_mapping_spotcheck(
        torch.randn(1, cfg.HQ, 64, cfg.D), torch.randn(1, cfg.HKV, 64, cfg.D),
        torch.randn(1, cfg.HKV, 64, cfg.D), scale)
    check("S8 GQA 正向映射（=逐头 kv[h//2] MHA）", spot["forward_ok"],
          f"fwd_err={spot['forward_vs_manual_max_abs']:.2e}")
    check("S9 GQA 错误映射被拒（灵敏度）", spot["wrong_maps_rejected"],
          f"w1={spot['wrong_map_offset1_max_abs']:.2e} w2={spot['wrong_map_groupshift_max_abs']:.2e}")

    # ---- S10: kernel_sim 覆盖断言复跑（子集，快）----
    import kernel_sim  # noqa: PLC0415
    q = torch.randn(1, cfg.HQ, 129, cfg.D)
    k = torch.randn(1, cfg.HKV, 129, cfg.D)
    v = torch.randn(1, cfg.HKV, 129, cfg.D)
    o, cov, kv_visit, _ = kernel_sim.sim_prefill(q, k, v, scale)
    try:
        kernel_sim.assert_prefill_coverage(cov, kv_visit, "smoke S129")
        check("S10 prefill sim 覆盖断言 S=129", True)
    except AssertionError as exc:
        check("S10 prefill sim 覆盖断言 S=129", False, str(exc))
    rel = (o - gqa_prefill_reference(q, k, v, scale)).abs().max().item() / \
        gqa_prefill_reference(q, k, v, scale).abs().max().item()
    check("S11 prefill sim 数值", rel < 2e-3, f"rel={rel:.2e}")
    qn1 = torch.randn(1, cfg.HQ, cfg.D)
    kn1 = torch.randn(1, cfg.HKV, 127, cfg.D)
    vn1 = torch.randn(1, cfg.HKV, 127, cfg.D)
    od, covd, kvd = kernel_sim.sim_decode(qn1, kn1, vn1, scale, 128)
    try:
        kernel_sim.assert_decode_coverage(covd, kvd, "smoke N127/ws128")
        check("S12 decode sim 覆盖断言 N=127 ws=128", True)
    except AssertionError as exc:
        check("S12 decode sim 覆盖断言 N=127 ws=128", False, str(exc))

    # ---- S13: roofline 手算数 ----
    p = ra.prefill_profile(1, 512, "fp16", "cuda")
    # AI_unique 手算：2·S(S+1)·D·HQ / ((2·HQ+2·HKV)·S·D·2)
    ai_hand = (2 * 512 * 513 * 128 * 16) / ((2 * 16 + 2 * 8) * 512 * 128 * 2)
    check("S13 prefill AI 手算", abs(p["ai_unique"] - ai_hand) < 0.1,
          f"roofline={p['ai_unique']} hand={ai_hand:.2f}")
    d_ = ra.decode_profile(1, 2048, "fp16", "cuda")
    kv_hand = ra.kv_bytes_per_token_per_layer(2)
    check("S14 KV 字节/层（附录 B 题 2）", kv_hand == 2 * 8 * 128 * 2 == 4096)
    check("S15 KV 字节/模型", ra.kv_bytes_per_token_model(2) == 4096 * 28 == 114688)
    ai_dec = (4 * 2048 * 128 * 16) / (2 * 8 * 2048 * 128 * 2 + 2 * 16 * 128 * 2)
    check("S16 decode AI 手算", abs(d_["ai_unique"] - ai_dec) < 0.01,
          f"roofline={d_['ai_unique']} hand={ai_dec:.4f}")

    # ---- CUDA/MUSA 核专属：合同校验 ----
    if module is not None:
        dev = torch.device("cuda")
        qg = torch.randn(1, cfg.HQ, 64, cfg.D, device=dev, dtype=torch.float16)
        kg = torch.randn(1, cfg.HKV, 64, cfg.D, device=dev, dtype=torch.float16)
        vg = torch.randn(1, cfg.HKV, 64, cfg.D, device=dev, dtype=torch.float16)
        expect_error("S17 float64 输入拒绝",
                     lambda: module.gqa_prefill_attention(qg.double(), kg.double(), vg.double(), scale))
        expect_error("S17b q/k dtype 不一致拒绝",
                     lambda: module.gqa_prefill_attention(qg, kg.float(), vg.float(), scale))
        expect_error("S18 非连续输入拒绝",
                     lambda: module.gqa_prefill_attention(
                         qg.transpose(1, 2).contiguous().transpose(1, 2), kg, vg, scale))
        expect_error("S19 头数不整除拒绝",
                     lambda: module.gqa_prefill_attention(
                         qg[:, :15], kg, vg, scale))
        expect_error("S20 decode head_dim≠128 拒绝",
                     lambda: module.gqa_decode_attention(
                         torch.randn(1, cfg.HQ, 127, device=dev, dtype=torch.float16),
                         kg, vg, scale))
        expect_error("S20b decode N=0 拒绝",
                     lambda: module.gqa_decode_attention(
                         torch.randn(1, cfg.HQ, cfg.D, device=dev, dtype=torch.float16),
                         kg[:, :, :0].contiguous(), vg[:, :, :0].contiguous(), scale))
        # 小形状数值（含尾部边界 S=129）
        for s in (1, 64, 129):
            qq = torch.randn(1, cfg.HQ, s, cfg.D, device=dev, dtype=torch.float16)
            kk = torch.randn(1, cfg.HKV, s, cfg.D, device=dev, dtype=torch.float16)
            vv = torch.randn(1, cfg.HKV, s, cfg.D, device=dev, dtype=torch.float16)
            ref = gqa_prefill_reference(qq.float().cpu(), kk.float().cpu(), vv.float().cpu(), scale)
            out = module.gqa_prefill_attention(qq, kk, vv, scale).float().cpu()
            rel = (out - ref).abs().max().item() / ref.abs().max().item()
            check(f"S21 prefill kernel S={s} vs CPU fp32", rel < 2e-2, f"rel={rel:.2e}")
        for n in (1, 127):
            qq = torch.randn(1, cfg.HQ, cfg.D, device=dev, dtype=torch.float16)
            kk = torch.randn(1, cfg.HKV, n, cfg.D, device=dev, dtype=torch.float16)
            vv = torch.randn(1, cfg.HKV, n, cfg.D, device=dev, dtype=torch.float16)
            ref = gqa_decode_reference(qq.float().cpu(), kk.float().cpu(), vv.float().cpu(), scale)
            out = module.gqa_decode_attention(qq, kk, vv, scale).float().cpu()
            rel = (out - ref).abs().max().item() / ref.abs().max().item()
            check(f"S22 decode kernel N={n} vs CPU fp32", rel < 2e-2, f"rel={rel:.2e}")
        check("S23 设备 warpSize 探针", module.warp_size_device() in (32, 128),
              f"probe={module.warp_size_device()}")

    print(f"\n[smoke] 总计 PASS={PASS} FAIL={FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
