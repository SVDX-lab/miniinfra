#!/usr/bin/env python3
"""第 28 期 fixtures：固定 seed 真值（朴素路径全量专家锚点）+ manifest 逐张量 SHA256。

FP64 锚点规则：全链原生 fp64（权重/路由器/输入的生成与除法都在 fp64 域执行，
即 make_* 以 dtype=torch.float64 调用），不用 fp32 权重 .double() 上转——
He 缩放除法在 fp32 域先舍入会引入 ~1e-8 相对差（教训见 EXPERIMENT_LOG §1）。

case 划分（权重与 x 按 seed 重建、不进 npz；只存真值输出/路由/桶结构）：
  A_tiny   d=32,  d_ff=64,  E=8,  k=2, N=16，矩阵路由器（uniform 权重分布）
  B_mid    d=64,  d_ff=128, E=16, k=6, N=64，矩阵路由器
  C_zipf   同 B 维度，planted Zipf 选择（80% token top-1 落 20% 热集）
  D_tie    d=32,  d_ff=64,  E=8,  k=3, N=8，路由器各行相同 → 分数全同
           （tie-break 真值：选中专家 = {0..k-1}，升序）
  E_real   V2-Lite 口径 d=2048, d_ff=1408, E=64, k=6, N=4（FP32+FP64 双锚点，
           权重按 seed 重建 ~6.6GB 峰值，顺序生成释放）

用法：
  python generate_fixtures.py               # 生成本地 fixtures/（source of truth）
  python generate_fixtures.py --verify-only # 按 seed 重建全流水，与 npz 对拍
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

FP32_TOL = 2e-4   # 跨 torch 版本 BLAS 舍入漂移容差（ch01 口径）
FP64_TOL = 1e-10


def _sha(a: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(a.shape).encode())
    digest.update(str(a.dtype).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def case_a() -> dict[str, torch.Tensor]:
    w = mr.make_expert_weights(8, 32, 64, cfg.SEED_BASE)
    r = mr.make_router(8, 32, cfg.SEED_BASE)
    x = mr.make_inputs(16, 32, cfg.SEED_BASE)
    idx, wts = mr.route(x, r, 2)
    w64 = mr.make_expert_weights(8, 32, 64, cfg.SEED_BASE, torch.float64)
    r64 = mr.make_router(8, 32, cfg.SEED_BASE, torch.float64)
    idx64, wts64 = mr.route(x.double(), r64, 2)
    return {
        "A_naive_fp32": mr.naive_moe_pairs(x, w, idx, wts),
        "A_naive_fp64": mr.naive_moe_pairs(x.double(), w64, idx64, wts64),
        "A_idx": idx, "A_w": wts,
    }


def case_b() -> dict[str, torch.Tensor]:
    w = mr.make_expert_weights(16, 64, 128, 20260400)
    r = mr.make_router(16, 64, 20260400)
    x = mr.make_inputs(64, 64, 20260400)
    idx, wts = mr.route(x, r, 6)
    w64 = mr.make_expert_weights(16, 64, 128, 20260400, torch.float64)
    r64 = mr.make_router(16, 64, 20260400, torch.float64)
    idx64, wts64 = mr.route(x.double(), r64, 6)
    return {
        "B_naive_fp32": mr.naive_moe_pairs(x, w, idx, wts),
        "B_naive_fp64": mr.naive_moe_pairs(x.double(), w64, idx64, wts64),
        "B_idx": idx, "B_w": wts,
    }


def case_c() -> dict[str, torch.Tensor]:
    w = mr.make_expert_weights(16, 64, 128, 20260400)
    x = mr.make_inputs(64, 64, 20260400)
    idx, wts, stats = mr.make_zipf_selection(64, 16, 6, 20260450)
    _, meta = mr.grouped_moe_pairs(x, w, idx, wts, return_buckets=True)
    w64 = mr.make_expert_weights(16, 64, 128, 20260400, torch.float64)
    return {
        "C_naive_fp32": mr.naive_moe_pairs(x, w, idx, wts),
        "C_naive_fp64": mr.naive_moe_pairs(
            x.double(), w64, idx, wts.double()),
        "C_idx": idx, "C_w": wts, "C_counts": meta["counts"],
        "C_stats_hotshare": torch.tensor([stats["hot_share_measured"], stats["n_hot"]]),
    }


def case_d() -> dict[str, torch.Tensor]:
    w = mr.make_expert_weights(8, 32, 64, 20260500)
    x = mr.make_inputs(8, 32, 20260500)
    r = torch.zeros(8, 32)
    r[:, 0] = 1.0  # 各行相同 → 每 token logits 跨专家全同 → 纯 tie
    idx, wts = mr.route(x, r, 3)
    expected = torch.arange(3).unsqueeze(0).repeat(8, 1)
    assert torch.equal(idx, expected), f"tie-break 真值异常: {idx[0].tolist()}"
    return {
        "D_naive_fp32": mr.naive_moe_pairs(x, w, idx, wts),
        "D_idx": idx, "D_w": wts, "D_expected_idx": expected,
    }


def case_e() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    w32 = mr.make_expert_weights(64, 2048, 1408, 20260550)
    r = mr.make_router(64, 2048, 20260550)
    x = mr.make_inputs(4, 2048, 20260550)
    idx, wts = mr.route(x, r, 6)
    out["E_naive_fp32"] = mr.naive_moe_pairs(x, w32, idx, wts)
    out["E_idx"] = idx
    out["E_w"] = wts
    del w32
    w64 = mr.make_expert_weights(64, 2048, 1408, 20260550, torch.float64)
    r64 = mr.make_router(64, 2048, 20260550, torch.float64)
    idx64, wts64 = mr.route(x.double(), r64, 6)
    assert torch.equal(idx64, idx), "E case FP64/FP32 路由选择不一致（近 tie）"
    out["E_naive_fp64"] = mr.naive_moe_pairs(x.double(), w64, idx64, wts64)
    return out


CASES = {"A": case_a, "B": case_b, "C": case_c, "D": case_d, "E": case_e}


def build_all() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, fn in CASES.items():
        print(f"[fixtures] 生成 case {name} ...")
        tensors.update({f"{k}": v for k, v in fn().items()})
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    fixture_path = HERE / "fixtures" / "correctness.npz"
    manifest_path = HERE / "fixtures" / "manifest.json"

    if args.verify_only:
        ref = np.load(fixture_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fresh = build_all()
        worst = {"fp32": 0.0, "fp64": 0.0, "idx_mismatch": 0}
        for name in manifest:
            base = name.replace(".npz", "")
            if base not in fresh:
                continue
            a = fresh[base].cpu().numpy()
            sha = _sha(a)
            ok_sha = sha == manifest[base]["sha256"]
            d = np.abs(a.astype(np.float64) - ref[name].astype(np.float64)).max() if a.dtype.kind == "f" else 0.0
            if a.dtype.kind == "f":
                tier = "fp64" if a.dtype == np.float64 else "fp32"
                worst[tier] = max(worst[tier], float(d))
            elif "idx" in base and not ok_sha:
                worst["idx_mismatch"] += 1
            status = "SHA256=OK" if ok_sha else f"SHA256=DRIFT(maxdiff={d:.3e})"
            print(f"[verify] {base:22s} {status}")
        print(f"[verify] 汇总: fp32 worst={worst['fp32']:.3e} (tol {FP32_TOL:.0e}) | "
              f"fp64 worst={worst['fp64']:.3e} (tol {FP64_TOL:.0e}) | "
              f"idx 漂移 {worst['idx_mismatch']}")
        passed = worst["fp32"] <= FP32_TOL and worst["fp64"] <= FP64_TOL and worst["idx_mismatch"] == 0
        print(f"[verify] {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1

    tensors = build_all()
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fixture_path, **{k: v.cpu().numpy() for k, v in tensors.items()})
    manifest = {}
    for name, t in tensors.items():
        a = t.cpu().numpy()
        manifest[name] = {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": _sha(a)}
    manifest["__meta__"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_base": cfg.SEED_BASE,
        "torch": torch.__version__,
        "cases": {
            "A_tiny": "d=32,d_ff=64,E=8,k=2,N=16 seed=20260329",
            "B_mid": "d=64,d_ff=128,E=16,k=6,N=64 seed=20260400",
            "C_zipf": "同B + make_zipf_selection(seed=20260450, hot=20%, token=80%, alpha=1)",
            "D_tie": "d=32,d_ff=64,E=8,k=3,N=8 全同分数 seed=20260500",
            "E_real": "V2-Lite dims d=2048,d_ff=1408,E=64,k=6,N=4 seed=20260550",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[fixtures] 写入 {fixture_path}（{len(tensors)} 张量）+ manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
