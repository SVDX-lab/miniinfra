#!/usr/bin/env python3
"""第 28 期实验 1：正确性主对 + 浮点结合律 + tie-break 显式验证。

口径：
  §9 主对 = 分组 GEMM 路径 vs "逐 token 全量专家再筛选"朴素路径（同一权重同输入，
  同 dtype）——fixtures 5 case + 现场对算矩阵（小 N，朴素路径 E×N 代价受限）；
  fixtures 锚点复算 = --verify-only 同口径位级复核；
  结合律差 = naive（slot 序=分数降序求和） vs grouped（桶序=专家升序 index_add）
  vs 第三序（按 token-row 显式按专家升序 python 求和）——FP32/FP64 两档记录（接 ch08）；
  tie-break = 全同分数构造（零路由器/同行车路由器）+ 近 tie 构造（±1ulp 分数）。
产出 results/correctness-results-cpu.json。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import moe_config as cfg  # noqa: E402
import moe_reference as mr  # noqa: E402
from experiment_utils import comparison_metrics, resolve_device, write_results_json  # noqa: E402

torch.set_num_threads(min(12, torch.get_num_threads()))


def comparison_metrics_native(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """FP64 锚点专用：不做 .float() 降精度，按原 dtype 计算 ch01 同口径指标。"""
    actual_d = actual.detach().double().cpu()
    expected_d = expected.detach().double().cpu()
    abs_err = (actual_d - expected_d).abs()
    denom = expected_d.abs().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(
        actual_d.flatten(), expected_d.flatten(), dim=0).item()
    return {
        "max_abs_error": abs_err.max().item(),
        "mean_abs_error": abs_err.mean().item(),
        "mean_rel_error": (abs_err / denom).mean().item(),
        "cosine_similarity": cos,
        "nan_count": int(torch.isnan(actual_d).sum()),
        "inf_count": int(torch.isinf(actual_d).sum()),
        "max_abs_error_position": int(abs_err.argmax()),
        "metric_dtype": "float64-native",
    }


def combine_token_expert_order(outs: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """第三求和序：每 token 内按专家 ID 升序逐项标量加（显式结合律参照物）。

    outs [E, N, d]；idx/w [N, k]（w 按 slot 存）。权重对齐按 (token, expert) 配对。
    """
    n = idx.shape[0]
    y = torch.zeros_like(outs[0])
    for t in range(n):
        pairs = sorted(zip(idx[t].tolist(), w[t].tolist()))
        acc = outs[pairs[0][0], t] * pairs[0][1]
        for e, wv in pairs[1:]:
            acc = acc + outs[e, t] * wv
        y[t] = acc
    return y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()
    device = resolve_device(args.backend)

    entries: list[dict] = []

    # ------------------------------------------------------------------
    # 1) fixtures 锚点：分组路径复算 vs npz 真值（朴素路径生成时已位级锁定）
    # ------------------------------------------------------------------
    ref = np.load(HERE / "fixtures" / "correctness.npz")
    fixture_cases = [
        ("A_tiny", dict(d=32, d_ff=64, E=8, k=2, N=16, seed=cfg.SEED_BASE), "route"),
        ("B_mid", dict(d=64, d_ff=128, E=16, k=6, N=64, seed=20260400), "route"),
        ("C_zipf", dict(d=64, d_ff=128, E=16, k=6, N=64, seed=20260400), "zipf"),
        ("D_tie", dict(d=32, d_ff=64, E=8, k=3, N=8, seed=20260500), "tie"),
        ("E_real", dict(d=2048, d_ff=1408, E=64, k=6, N=4, seed=20260550), "route"),
    ]
    for tag, prm, mode in fixture_cases:
        w = mr.make_expert_weights(prm["E"], prm["d"], prm["d_ff"], prm["seed"])
        r = mr.make_router(prm["E"], prm["d"], prm["seed"])
        x = mr.make_inputs(prm["N"], prm["d"], prm["seed"])
        if mode == "route":
            idx, wts = mr.route(x, r, prm["k"])
        elif mode == "zipf":
            idx, wts, _ = mr.make_zipf_selection(prm["N"], prm["E"], prm["k"], 20260450)
        else:  # tie
            r = torch.zeros(prm["E"], prm["d"])
            r[:, 0] = 1.0
            idx, wts = mr.route(x, r, prm["k"])
        yg = mr.grouped_moe_pairs(x, w, idx, wts)
        for dtype_key, yv in [("fp32", yg)]:
            truth = torch.from_numpy(np.asarray(ref[f"{tag[0]}_naive_{dtype_key}"]))
            m = comparison_metrics(yv, truth)
            entries.append({"pair": f"grouped({tag}) vs naive-truth[{dtype_key}]",
                            "dtype": "float32", **m})
            print(f"[correctness] {tag} grouped vs truth fp32: max_abs={m['max_abs_error']:.2e} "
                  f"cos={m['cosine_similarity']:.10f}")
        if f"{tag[0]}_naive_fp64" in ref:
            # FP64 锚点权重按 dtype 原生重建（_he 的除法在 fp64 域执行）；
            # 用 fp32 权重 .double() 会引入 ~1e-8 相对差（除法舍入域不同）——
            # 教训记录进 EXPERIMENT_LOG §1。
            w64 = mr.make_expert_weights(prm["E"], prm["d"], prm["d_ff"], prm["seed"],
                                         torch.float64)
            r64 = mr.make_router(prm["E"], prm["d"], prm["seed"], torch.float64)
            if mode == "route":
                idx64, wts64 = mr.route(x.double(), r64, prm["k"])
            elif mode == "zipf":
                idx64, wts64 = idx, wts.double()
            else:  # tie
                r64 = torch.zeros(prm["E"], prm["d"], dtype=torch.float64)
                r64[:, 0] = 1.0
                idx64, wts64 = mr.route(x.double(), r64, prm["k"])
            yg64 = mr.grouped_moe_pairs(x.double(), w64, idx64, wts64)
            truth64 = torch.from_numpy(np.asarray(ref[f"{tag[0]}_naive_fp64"]))
            m = comparison_metrics_native(yg64, truth64)
            entries.append({"pair": f"grouped({tag}) vs naive-truth[fp64]",
                            "dtype": "float64", **m})
            print(f"[correctness] {tag} grouped vs truth fp64: max_abs={m['max_abs_error']:.2e}")

    # ------------------------------------------------------------------
    # 2) 现场主对矩阵：小 N × E × k × dtype（朴素 vs 分组同机对算）
    # ------------------------------------------------------------------
    for (E, d, d_ff) in [(8, 32, 64), (16, 64, 128)]:
        w = mr.make_expert_weights(E, d, d_ff, 20260600 + E)
        r = mr.make_router(E, d, 20260600 + E)
        w64 = {k: v.double() for k, v in w.items()}
        r64 = r.double()
        for N in [1, 8, 33]:
            x = mr.make_inputs(N, d, 20260600 + E)
            for k in ([2, 6, 8] if E == 8 else [2, 6]):
                for dtype in ["fp32", "fp64"]:
                    if dtype == "fp32":
                        xn, wn, rn, wf = x, w, r, w
                    else:
                        xn, wn, rn, wf = x.double(), w64, r64, w64
                    yn = mr.naive_moe(xn, wn, rn, k)
                    yg = mr.grouped_moe(xn, wf, rn, k)
                    m = comparison_metrics(yg, yn)
                    entries.append({"pair": f"live grouped-vs-naive E={E} d={d} N={N} k={k}",
                                    "dtype": dtype, **m})
                    print(f"[correctness] live E={E} N={N} k={k} {dtype}: "
                          f"max_abs={m['max_abs_error']:.2e} rel={m['mean_rel_error']:.2e}")

    # ------------------------------------------------------------------
    # 3) 浮点结合律差（接 ch08）：slot 序 vs 桶序 vs 第三序（专家升序标量加）
    # ------------------------------------------------------------------
    w = mr.make_expert_weights(16, 64, 128, 20260616)
    r = mr.make_router(16, 64, 20260616)
    x = mr.make_inputs(64, 64, 20260616)
    idx, wts = mr.route(x, r, 6)
    assoc = {}
    for dtype in ["fp32", "fp64"]:
        if dtype == "fp32":
            xf, wf, rf, idxf, wtf = x, w, r, idx, wts
        else:
            xf = x.double()
            wf = {k: v.double() for k, v in w.items()}
            rf = r.double()
            idxf, wtf = mr.route(xf, rf, 6)
        E = w["W_gate"].shape[0]
        outs = torch.stack(
            [mr.expert_ffn(xf, wf["W_gate"][e], wf["W_up"][e], wf["W_down"][e]) for e in range(E)],
            dim=0)
        y_slot = mr.naive_moe_pairs(xf, wf, idxf, wtf)          # slot 序（分数降序）
        y_bucket = mr.grouped_moe_pairs(xf, wf, idxf, wtf)      # 桶序（专家升序 index_add）
        y_third = combine_token_expert_order(outs, idxf, wtf)   # 第三序（专家升序标量加）
        assoc[dtype] = {
            "slot_vs_bucket_max_abs": float((y_slot - y_bucket).abs().max()),
            "slot_vs_third_max_abs": float((y_slot - y_third).abs().max()),
            "bucket_vs_third_max_abs": float((y_bucket - y_third).abs().max()),
        }
        print(f"[assoc] {dtype}: {assoc[dtype]}")
    entries.append({"pair": "combine-order associativity (slot/bucket/third-order)",
                    "fp32": assoc["fp32"], "fp64": assoc["fp64"],
                    "note": "三序数学等价，差异=纯浮点结合律（ch08 主题）；"
                            "FP64 级差≈1e-15、FP32 级差≈1e-6"})

    # ------------------------------------------------------------------
    # 4) tie-break 显式验证：全同分数 + 近 tie（±1ulp）
    # ------------------------------------------------------------------
    E, d, d_ff, N = 16, 32, 64, 8
    wt = mr.make_expert_weights(E, d, d_ff, 20260630)
    xt = mr.make_inputs(N, d, 20260630)
    # 全同分数：路由器所有行相同
    r_tie = torch.zeros(E, d)
    r_tie[:, 0] = 1.0
    idx_tie, _ = mr.route(xt, r_tie, 4)
    exp_all = torch.arange(4).unsqueeze(0).repeat(N, 1)
    tie_exact = torch.equal(idx_tie, exp_all)
    # 近 tie：整行构造——专家 1 的分数比专家 0 恰大 1 ulp（fp32），其余专家低 1.0；
    # 期望排序 = [1, 0, 其余按 tie-break 升序]
    base = torch.randn(N, generator=torch.Generator().manual_seed(42)) * 0.1
    logits_lp = torch.empty(N, E)
    logits_lp[:, :] = -1.0
    logits_lp[:, 0] = base
    logits_lp[:, 1] = torch.nextafter(base, torch.tensor(float("inf")))
    idx_lp32, _ = mr.route_topk(logits_lp, 4)
    idx_lp64, _ = mr.route_topk(logits_lp.double(), 4)
    # fp32 softmax 对 1ulp 差是否保序（sub-resolution 塌缩观察）
    p32 = torch.softmax(logits_lp, dim=-1)
    p64_ = torch.softmax(logits_lp.double(), dim=-1)
    collapse32 = bool((p32[:, 0] == p32[:, 1]).all())          # 概率塌缩为相等
    preserved64 = bool((p64_[:, 1] > p64_[:, 0]).all())        # fp64 保序
    # 期望行为：fp32 塌缩后按 tie-break（低索引 0 先）；fp64 尊重 1ulp 优势（1 先）
    near_tie_ok = (collapse32 and (idx_lp32[0, 0].item() == 0)
                   and preserved64 and (idx_lp64[0, 0].item() == 1))
    entries.append({
        "pair": "tie-break explicit",
        "all_equal_scores_selected": idx_tie[0].tolist(),
        "all_equal_matches_lower_index_rule": tie_exact,
        "near_tie_1ulp": "专家1 logits 比专家0大 1ulp、其余专家 -1.0（整行构造）",
        "fp32_softmax_1ulp_collapse_to_equal": collapse32,
        "fp32_selection_after_collapse": idx_lp32[0].tolist(),
        "fp64_1ulp_preserved": preserved64,
        "fp64_selection": idx_lp64[0].tolist(),
        "near_tie_dual_expectation_pass": near_tie_ok,
        "finding": "有效 tie 类 = softmax 舍入后的相等：fp32 下 1ulp(~1e-7 相对) 分数差"
                   "被 softmax 输出分辨率吞掉 → 塌缩为 tie → tie-break 规则（低索引）裁决；"
                   "fp64 保序。路由选择对 sub-resolution 分数差不敏感，但生产实现若用"
                   "sigmoid/不同 softmax 实现会改变塌缩边界（接 ch06/ch08）。",
    })
    print(f"[tie-break] 全同分数 → {idx_tie[0].tolist()}（规则=低索引胜出, exact={tie_exact}）; "
          f"fp32 1ulp 塌缩={collapse32} → 选择 {idx_lp32[0].tolist()} | "
          f"fp64 保序={preserved64} → 选择 {idx_lp64[0].tolist()}")

    # ------------------------------------------------------------------
    # 5) 确定性：路由 + 分组输出 3 次位级
    # ------------------------------------------------------------------
    det = []
    for _ in range(3):
        det.append(mr.grouped_moe(x, w, r, 6))
    det_ok = all(torch.equal(det[0], d_) for d_ in det[1:])
    entries.append({"pair": "determinism 3x bit-level (route+grouped)", "pass": det_ok})
    print(f"[determinism] 3 次位级一致 = {det_ok}")

    payload = {
        "schema_version": "kernel-chapter29.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "device": str(device),
        "main_pair_rule": "grouped(sort/gather/bucket/scatter) vs naive(全量专家再筛选), 同权重同输入",
        "entries": entries,
    }
    write_results_json(HERE / "results" / "correctness-results-cpu.json", payload)

    fails = [e for e in entries if "max_abs_error" in e and
             (e["dtype"] == "float64" and e["max_abs_error"] > 1e-10
              or e["dtype"] == "float32" and e["max_abs_error"] > 2e-4)]
    print(f"[correctness] 汇总: {len(entries) - len(fails)}/{len(entries)} PASS")
    return 0 if not fails and det_ok and tie_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
