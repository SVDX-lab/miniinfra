#!/usr/bin/env python3
"""第 28 期 smoke test：边界行为验证（全本地 CPU）。

覆盖：单 token、空桶、全 token 单专家（最深桶）、k=E 退化、k=1、tie-break、
norm=false 变体、FP32/FP64 误差尺度、NaN 传播、形状/k 越界报错、确定性、
零路由器、Zipf 构造统计。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import moe_reference as mr  # noqa: E402

torch.set_num_threads(min(12, torch.get_num_threads()))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name} {detail}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def main() -> int:
    print("== smoke: MoE 路由 + 分组 GEMM 边界行为 ==")

    # 1. 单 token
    w = mr.make_expert_weights(8, 32, 64, 1)
    r = mr.make_router(8, 32, 1)
    x1 = mr.make_inputs(1, 32, 1)
    y1n = mr.naive_moe(x1, w, r, 2)
    y1g = mr.grouped_moe(x1, w, r, 2)
    check("N=1 分组 vs 朴素", (y1n - y1g).abs().max().item() < 1e-5,
          f"max_abs={(y1n - y1g).abs().max().item():.2e}")

    # 2. 空桶（E=64, N=1, k=2 → 至少 62 空桶）
    w64 = mr.make_expert_weights(64, 32, 64, 2)
    r64 = mr.make_router(64, 32, 2)
    _, meta = mr.grouped_moe(x1, w64, r64, 2, return_buckets=True)
    check("空桶计数", meta["n_empty"] >= 62 and int(meta["counts"].sum()) == 2,
          f"empty={meta['n_empty']} sum_counts={int(meta['counts'].sum())}")

    # 3. 全 token 单专家（最深桶 = N×k）
    idx_all = torch.zeros(32, 3, dtype=torch.long)
    w_all = torch.full((32, 3), 1.0 / 3)
    x32 = mr.make_inputs(32, 32, 3)
    yg3, meta3 = mr.grouped_moe_pairs(x32, w, idx_all, w_all, return_buckets=True)
    yn3 = mr.naive_moe_pairs(x32, w, idx_all, w_all)
    check("全单专家桶深", meta3["depth_max"] == 32 * 3,
          f"depth_max={meta3['depth_max']} (N×k=96)")
    check("全单专家 数值", (yg3 - yn3).abs().max().item() < 1e-5,
          f"max_abs={(yg3 - yn3).abs().max().item():.2e}")

    # 4. k=E 退化（top-k=全专家：选择集 = 全部专家、顺序 = 分数降序、Σw=1）
    yg4 = mr.grouped_moe(x1, w, r, 8)
    idx4, w4 = mr.route(x1, r, 8)
    check("k=E 退化：选择=全部专家（顺序=分数降序）",
          sorted(idx4[0].tolist()) == list(range(8)), f"idx={idx4[0].tolist()}")
    check("k=E 退化：权重和=1", abs(float(w4.sum()) - 1.0) < 1e-6,
          f"sum={float(w4.sum()):.8f}")
    check("k=E 与朴素一致", (yg4 - mr.naive_moe(x1, w, r, 8)).abs().max().item() < 1e-5)

    # 5. k=1
    y5g = mr.grouped_moe(x1, w, r, 1)
    y5n = mr.naive_moe(x1, w, r, 1)
    check("k=1", (y5g - y5n).abs().max().item() < 1e-5)

    # 6. tie-break：全同分数 → 选中 {0..k-1}，3 次稳定
    r_tie = torch.zeros(8, 32)
    r_tie[:, 0] = 1.0
    idxs = [mr.route(x1, r_tie, 3)[0] for _ in range(3)]
    expected = torch.arange(3).unsqueeze(0)
    check("tie-break：低索引胜出", all(torch.equal(i, expected.expand_as(i)) for i in idxs),
          f"idx={idxs[0].tolist()}")
    check("tie-break：3 次稳定", all(torch.equal(idxs[0], i) for i in idxs[1:]))

    # 7. norm=False 变体（V2-Lite 口径声明）：权重 = softmax 截断（Σ<1）
    _, w_norm = mr.route(x1, r, 6, norm=False)
    _, w_t = mr.route(x1, r, 6, norm=True)
    check("norm=False：Σ<1 且 ≤ norm=True", float(w_norm.sum()) < 1.0 and
          float(w_norm.sum()) <= float(w_t.sum()) + 1e-6,
          f"sum(norm=False)={float(w_norm.sum()):.4f}")

    # 8. FP32/FP64 误差尺度
    wd = {k: v.double() for k, v in w.items()}
    yd_n = mr.naive_moe(x1.double(), wd, r.double(), 2)
    yd_g = mr.grouped_moe(x1.double(), wd, r.double(), 2)
    rel64 = ((yd_n - yd_g).abs() / yd_n.abs().clamp_min(1e-6)).max().item()
    rel32 = ((y1n - y1g).abs() / y1n.abs().clamp_min(1e-6)).max().item()
    check("FP64 机器精度", rel64 < 1e-12, f"rel_max={rel64:.2e}")
    check("FP32 舍入级", rel32 < 1e-4, f"rel_max={rel32:.2e}")

    # 9. NaN 传播：NaN token 只污染自己
    x_nan = x32.clone()
    x_nan[5] = float("nan")
    yg9 = mr.grouped_moe(x_nan, w, r, 2)
    check("NaN 只污染本 token", bool(torch.isnan(yg9[5]).any())
          and not bool(torch.isnan(yg9[[0, 1, 6, 31]]).any()))

    # 10/11. 形状/k 越界报错
    try:
        mr.grouped_moe(torch.randn(4, 33), w, r, 2)
        check("形状不匹配报错", False)
    except ValueError:
        check("形状不匹配报错", True)
    try:
        mr.grouped_moe(x1, w, r, 9)
        check("k 越界报错", False)
    except ValueError:
        check("k 越界报错", True)

    # 12. 确定性：3 次位级一致（路由 + 输出）
    o1 = mr.grouped_moe(x32, w, r, 2)
    o2 = mr.grouped_moe(x32, w, r, 2)
    o3 = mr.grouped_moe(x32, w, r, 2)
    i_a = mr.route(x32, r, 2)[0]
    i_b = mr.route(x32, r, 2)[0]
    check("确定性 3x 位级", torch.equal(o1, o2) and torch.equal(o2, o3)
          and torch.equal(i_a, i_b))

    # 13. 零路由器：均匀分数，权重恰 1/k（E=8,k=2 → 0.5，2 的幂精确）
    r_zero = torch.zeros(8, 32)
    idx13, w13 = mr.route(x1, r_zero, 2)
    check("零路由器：idx=前 k 个专家", torch.equal(idx13, torch.arange(2).unsqueeze(0)),
          f"idx={idx13.tolist()}")
    check("零路由器：权重=1/k 精确", torch.equal(w13, torch.full_like(w13, 0.5)),
          f"w={w13.tolist()}")

    # 14. 桶内重复 token 的 index_add 合并正确（全 token 单专家已覆盖数值，
    #     此处验证合并顺序无关性：打乱 token 顺序输入，逐 token 输出应随行置换）
    perm = torch.randperm(32, generator=torch.Generator().manual_seed(7))
    yg14 = mr.grouped_moe(x32[perm], w, r, 2)
    inv = torch.argsort(perm)
    check("行置换等变性", (yg14[inv] - o1).abs().max().item() < 1e-5,
          f"max_abs={(yg14[inv] - o1).abs().max().item():.2e}")

    # 15. Zipf 构造统计（N=2048, E=64：热占比 ≈ 0.8；行内无重复专家）
    idx15, _, st15 = mr.make_zipf_selection(2048, 64, 6, 20260450)
    dup = any(len(set(row.tolist())) != 6 for row in idx15)
    check("Zipf 热占比≈0.8", abs(st15["hot_share_measured"] - 0.8) < 0.05,
          f"share={st15['hot_share_measured']:.4f} n_hot={st15['n_hot']}")
    check("Zipf 行内专家唯一", not dup)

    print(f"== smoke 汇总: {PASS} PASS / {FAIL} FAIL ==")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
