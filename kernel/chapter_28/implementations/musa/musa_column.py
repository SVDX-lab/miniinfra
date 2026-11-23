#!/usr/bin/env python3
"""第 28 期 MUSA 列（可选）：torch_musa eager 全链小规模验证（S4000 会话 ≤30min）。

内容（对照任务书）：
  1. 正确性：musa 上路由 + 分组 GEMM 全链 vs musa naive（全量专家再筛选）+ vs
     CPU 同权重（对拍）；确定性 2 次位级；tie-break 行为探针（musa argsort
     stable 支持性显式记录——平台边界素材）；norm=false 变体；
  2. 延迟：grouped vs 单体大 GEMM 等价容量（dense FFN intermediate=k×d_ff，同
     专家段 FLOPs）——N ∈ {1, 8, 64} × E=64 × k=6，V2-Lite 真实 FFN 维
     （d=2048, d_ff=1408）；事件对计时 warmup=3 repeats=10（ch01 口径）；
  3. 环境快照。

M-50 纪律：所有张量构造显式 device=，禁止 CPU/MUSA 混用。
用法（S4000）：python3 implementations/musa/musa_column.py
产出：results/correctness-results-musa.json + benchmark-results-musa.json
     + environment-musa.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / ".." / "cpu"))

import moe_config as cfg  # noqa: E402
import moe_reference as mr  # noqa: E402
from experiment_utils import (OperatorTimer, device_environment_record,  # noqa: E402
                              resolve_device, write_results_json)

ROUTING_SEGS = ["gate_mm", "topk_norm", "sort_pairs", "gather_x", "scatter_combine"]
EXPERT_SEGS = ["expert_gate_up", "silu_mul", "expert_down"]


def to_device(weights: dict, router: torch.Tensor, device) -> tuple[dict, torch.Tensor]:
    return ({k: v.to(device) for k, v in weights.items()}, router.to(device))


ROUTE_STABLE_OK = True


def route_on(x, r, k, norm=True):
    """路由按 stable 探针结果选 musa/cpu 侧执行（选择在 CPU 生成时仍位级同语义）。"""
    if ROUTE_STABLE_OK:
        return mr.route(x, r, k, norm)
    return tuple(t.to(x.device) for t in mr.route(x.cpu(), r.cpu(), k, norm))


def main() -> int:
    device = resolve_device("musa")
    print(f"[musa] device = {torch.musa.get_device_properties(0)}")
    print(f"[musa] torch={torch.__version__}")

    # ------------------------------------------------------------------
    # 平台探针：argsort stable / bincount / index_add / topk 在 musa 的支持性
    # ------------------------------------------------------------------
    probes = {}
    t = torch.randn(8, 16, device=device)
    try:
        o = torch.argsort(t, dim=-1, descending=True, stable=True)
        probes["argsort_stable_descending"] = f"OK (shape={tuple(o.shape)})"
    except Exception as e:  # noqa: BLE001
        probes["argsort_stable_descending"] = f"FAIL: {type(e).__name__}: {e}"
    try:
        b = torch.bincount(torch.tensor([0, 1, 1, 3], device=device), minlength=4)
        probes["bincount"] = "OK" if b.shape[0] == 4 else f"UNEXPECTED {b}"
    except Exception as e:  # noqa: BLE001
        probes["bincount"] = f"FAIL: {type(e).__name__}: {e}"
    try:
        y = torch.zeros(4, 8, device=device)
        y.index_add_(0, torch.tensor([0, 0, 2], device=device),
                     torch.ones(3, 8, device=device))
        probes["index_add_"] = "OK"
    except Exception as e:  # noqa: BLE001
        probes["index_add_"] = f"FAIL: {type(e).__name__}: {e}"
    try:
        v, i = torch.topk(t, 3, dim=-1)
        probes["topk"] = "OK"
    except Exception as e:  # noqa: BLE001
        probes["topk"] = f"FAIL: {type(e).__name__}: {e}"
    for name, r in probes.items():
        print(f"[probe] {name}: {r}")

    # ------------------------------------------------------------------
    # 1) 正确性（小规模全链）
    # ------------------------------------------------------------------
    E, k, N = 16, 6, 32
    d, d_ff = 512, 384
    Wc = mr.make_expert_weights(E, d, d_ff, cfg.SEED_BASE + 900)
    rc = mr.make_router(E, d, cfg.SEED_BASE + 900)
    xc = mr.make_inputs(N, d, cfg.SEED_BASE + 900)
    W, r, x = to_device(Wc, rc, device)[0], to_device(Wc, rc, device)[1], xc.to(device)

    global ROUTE_STABLE_OK
    ROUTE_STABLE_OK = probes["argsort_stable_descending"].startswith("OK")
    entries = []
    idx, w = route_on(x, r, k)
    yg = mr.grouped_moe_pairs(x, W, idx, w)
    yn = mr.naive_moe_pairs(x, W, idx, w)
    abs_err = (yg - yn).abs()
    cos = torch.nn.functional.cosine_similarity(yg.flatten(), yn.flatten(), dim=0).item()
    entries.append({"pair": "musa grouped vs musa naive", "max_abs": abs_err.max().item(),
                    "mean_abs": abs_err.mean().item(), "cos": cos})
    print(f"[correctness] musa grouped vs naive: max_abs={abs_err.max().item():.3e} cos={cos:.10f}")

    ycpu = mr.grouped_moe_pairs(xc, Wc, idx.cpu().long(), w.cpu())
    abs_err_c = (yg.cpu() - ycpu).abs()
    entries.append({"pair": "musa grouped vs cpu grouped", "max_abs": abs_err_c.max().item(),
                    "mean_abs": abs_err_c.mean().item()})
    print(f"[correctness] musa grouped vs cpu grouped: max_abs={abs_err_c.max().item():.3e}")

    y2 = mr.grouped_moe_pairs(x, W, idx, w)
    y3 = mr.grouped_moe_pairs(x, W, idx, w)
    det = torch.equal(y2, y3)
    det_route = torch.equal(route_on(x, r, k)[0], idx)
    entries.append({"pair": "musa determinism 2x bit-level", "output": det, "route": det_route})
    print(f"[determinism] output={det} route={det_route}")

    # tie-break on musa（全同分数构造）
    r_tie = torch.zeros(E, d, device=device)
    r_tie[:, 0] = 1.0
    idx_tie = route_on(x, r_tie, k)[0]
    expected = torch.arange(k, device=device).unsqueeze(0).expand(N, k)
    tie_ok = torch.equal(idx_tie, expected)
    entries.append({"pair": "musa tie-break (equal scores)", "expected": expected[0].tolist(),
                    "got": idx_tie[0].tolist(), "pass": tie_ok})
    print(f"[tie-break] pass={tie_ok} idx0={idx_tie[0].tolist()}")

    # norm=false（V2-Lite 口径）
    _, w_nn = route_on(x, r, k, norm=False)
    entries.append({"pair": "norm=false sum (musa)", "sum_row0": float(w_nn[0].sum())})
    print(f"[norm=false] sum(w) row0 = {float(w_nn[0].sum()):.4f} (应 <1 且 ≤ norm=true)")

    write_results_json(ROOT / "results" / "correctness-results-musa.json", {
        "schema_version": "kernel-chapter29.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "musa",
        "shape": {"E": E, "k": k, "N": N, "d": d, "d_ff": d_ff},
        "platform_probes": probes,
        "entries": entries,
    })

    # ------------------------------------------------------------------
    # 2) 延迟：grouped vs 单体大 GEMM 等价容量（V2-Lite 真实 FFN 维）
    # ------------------------------------------------------------------
    E2, k2 = 64, 6
    Wb = mr.make_expert_weights(E2, cfg.SCAN_D_MODEL, cfg.SCAN_D_FF, cfg.SEED_BASE + 64)
    rb = mr.make_router(E2, cfg.SCAN_D_MODEL, cfg.SEED_BASE + 64)
    Wb, rb = to_device(Wb, rb, device)
    bench = []
    for Nb in [1, 8, 64]:
        xb = mr.make_inputs(Nb, cfg.SCAN_D_MODEL, cfg.SEED_BASE + 2000 + Nb).to(device)
        # 路由在 CPU 侧做（若 stable argsort 探针 OK 则 musa 侧做——按探针结果）
        ib, wb = route_on(xb, rb, k2)
        timer = OperatorTimer(device, warmup=3, repeats=10)
        e2e = timer.measure(lambda: mr.grouped_moe_pairs(xb, Wb, ib, wb))
        # dense equivalent（intermediate = k×d_ff 单体 FFN，MUSA 上现场建权）
        dff_t = k2 * cfg.SCAN_D_FF
        gen = torch.Generator().manual_seed(cfg.SEED_BASE + 7000)
        Wg_d = (torch.randn(dff_t, cfg.SCAN_D_MODEL, generator=gen) / cfg.SCAN_D_MODEL ** 0.5).to(device)
        Wu_d = (torch.randn(dff_t, cfg.SCAN_D_MODEL, generator=gen) / cfg.SCAN_D_MODEL ** 0.5).to(device)
        Wd_d = (torch.randn(cfg.SCAN_D_MODEL, dff_t, generator=gen) / dff_t ** 0.5).to(device)
        dense = timer.measure(lambda: (mr.silu(xb @ Wg_d.T) * (xb @ Wu_d.T)) @ Wd_d.T)
        row = {"tokens": Nb, "experts": E2, "topk": k2,
               "route_on": "musa" if probes["argsort_stable_descending"].startswith("OK") else "cpu",
               "e2e": e2e, "dense_equiv": dense,
               "moe_over_dense": round(e2e["p50_ms"] / dense["p50_ms"], 3)}
        bench.append(row)
        print(f"[bench] N={Nb} E={E2} k={k2}: grouped p50={e2e['p50_ms']:.3f}ms | "
              f"dense p50={dense['p50_ms']:.3f}ms | ratio={row['moe_over_dense']}x")
        del Wg_d, Wu_d, Wd_d
        torch.musa.empty_cache()

    write_results_json(ROOT / "results" / "benchmark-results-musa.json", {
        "schema_version": "kernel-chapter29.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "musa",
        "note": "grouped 路由（视 stable 探针结果在 musa/cpu 侧生成选择，见 route_on 字段）；"
                "dense_equiv = intermediate=k×d_ff 单体 FFN（同专家段 FLOPs）",
        "entries": bench,
    })

    env = device_environment_record(device)
    env["platform_probes"] = probes
    write_results_json(ROOT / "results" / "environment-musa.json", env)
    print("[musa] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
