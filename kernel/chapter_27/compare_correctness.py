#!/usr/bin/env python3
"""第 27 期实验 1：两形式数学等价验证（§9 新结构规则）。

对照设计：
  A. §9 主对（同权重同输入）：chunked(B) vs 逐 Token 递推，同 dtype 现场对算
     ——dtype ∈ {fp64, fp32, fp16io}（fp16io = q/k/v/β/g 存 fp16、递推/矩阵
     乘在 fp32 做——FLA 核真实口径；全程 fp16 的发散行为由 smoke_test 记录），
     B ∈ {16, 32, 64}，N ∈ {1,3,33,127,128,2048}，gate ∈ {scalar, vector}；
  B. 跨机真值锚点：现场重算 vs fixtures/correctness.npz（FP64 < 1e-10、
     FP32 < 2e-4；权重 SHA256 对拍）；
  C. initial_state ≠ 0 情形（decode carry 语义）；
  D. 确定性：fp32 主链 3 次连跑位级一致。

产出 results/correctness-results-cpu.json（后端名跟随 --backend）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import deltanet_chunked as chunked  # noqa: E402
import deltanet_config as cfg  # noqa: E402
import deltanet_reference as ref  # noqa: E402
from experiment_utils import comparison_metrics, sha256_of_tensor, write_results_json  # noqa: E402


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    return comparison_metrics(a, b)


def run_pair(proj: dict, dtype_tag: str, chunk_size: int, initial_state=None) -> dict:
    """同 dtype 下 chunked vs 递推（fp16io：存储 half、运算 float32）。"""
    if dtype_tag == "fp16io":
        cast = lambda t: t.to(torch.float16).to(torch.float32)
    elif dtype_tag == "fp32":
        cast = lambda t: t.to(torch.float32)
    else:
        cast = lambda t: t.to(torch.float64)
    q, k, v = cast(proj["q"]), cast(proj["k"]), cast(proj["v"])
    beta, g = cast(proj["beta"]), cast(proj["g_log"])
    s0 = None if initial_state is None else cast(initial_state)
    o_r, s_r = ref.recurrent_gdn(q, k, v, beta, g, initial_state=s0)
    o_c, s_c = chunked.chunked_gdn(q, k, v, beta, g, chunk_size=chunk_size, initial_state=s0)
    return {
        "output": compare(o_c, o_r),
        "final_state": compare(s_c, s_r),
        "rec_output_absmax": float(o_r.abs().max()),
        "rec_state_absnorm": float(s_r.norm()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()
    if args.backend != "cpu":
        raise SystemExit("本脚本为 CPU 真值列；GPU 列由服务器脚本另行对照同一 fixtures")

    weights = ref.make_weights(cfg.SEED_BASE)
    manifest = json.loads((HERE / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    weight_check = all(
        sha256_of_tensor(weights[name]) == digest
        for name, digest in manifest["weights_sha256"].items()
    )
    print(f"[anchor] 权重 SHA256 对拍：{'PASS' if weight_check else 'FAIL'}")

    groups = []
    # ---- A. §9 主对矩阵 ----
    for variant in ("scalar", "vector"):
        for n in cfg.CORRECTNESS_SEQ_LENS:
            gen = torch.Generator().manual_seed(cfg.SEED_BASE + 7000 + n)
            x = torch.randn(n, cfg.D_MODEL, generator=gen, dtype=torch.float32)
            proj = ref.project_inputs(x, weights, variant)
            for dtype_tag in ("fp64", "fp32", "fp16io"):
                for b in cfg.CHUNK_SWEEP:
                    r = run_pair(proj, dtype_tag, b)
                    groups.append({
                        "experiment": "A_main_pair", "gate_variant": variant,
                        "seq_len": n, "dtype": dtype_tag, "chunk_size": b,
                        **r,
                    })
            print(f"[A] {variant} N={n} done")

    # ---- B. fixtures 真值锚点 ----
    anchor = []
    with np.load(HERE / "fixtures" / "correctness.npz") as data:
        names = list(data.keys())
        for name in names:
            if name.endswith("_s0"):
                continue
            parts = name.split("_n")
            variant = parts[0]
            n = int(parts[1].split("_")[0])
            base = name[(len(variant) + len(f"_n{n}") + 1):]
            gen = torch.Generator().manual_seed(cfg.SEED_BASE + 7000 + n)
            x = torch.randn(n, cfg.D_MODEL, generator=gen, dtype=torch.float32)
            proj = ref.project_inputs(x, weights, variant)
            kind = "fp64" if "_fp64" in base else "fp32"
            cast = (lambda t: t.to(torch.float64)) if kind == "fp64" else (lambda t: t.float())
            q, k, v = cast(proj["q"]), cast(proj["k"]), cast(proj["v"])
            beta, g = cast(proj["beta"]), cast(proj["g_log"])
            if base.startswith("rec_"):
                form = "recurrent"
                o, s = ref.recurrent_gdn(q, k, v, beta, g)
            else:
                form = f"chunked_b64"
                o, s = chunked.chunked_gdn(q, k, v, beta, g, chunk_size=cfg.CHUNK_SIZE_DEFAULT)
            target = torch.from_numpy(data[name])
            if "out" in base:
                m = compare(o, target)
            else:
                m = compare(s, target)
            anchor.append({"key": name, "form": form, "dtype": kind,
                           "fixture_dtype": str(target.dtype), **m})
        # initial_state 情形
        s0_names = [x for x in names if x.endswith("_s0")]
        gen = torch.Generator().manual_seed(cfg.SEED_BASE + 9000 + 127)
        s0 = torch.randn(cfg.NUM_VALUE_HEADS, cfg.HEAD_K_DIM, cfg.HEAD_V_DIM,
                         generator=gen, dtype=torch.float32) * 0.1
        for name in s0_names:
            n = 127
            variant = name.split("_n")[0]
            base = name[len(f"{variant}_n{n}_"):-len("_s0")]
            x = torch.randn(n, cfg.D_MODEL,
                            generator=torch.Generator().manual_seed(cfg.SEED_BASE + 7000 + n),
                            dtype=torch.float32)
            proj = ref.project_inputs(x, weights, variant)
            kind = "fp64" if "_fp64" in base else "fp32"
            cast = (lambda t: t.to(torch.float64)) if kind == "fp64" else (lambda t: t.float())
            q, k, v = cast(proj["q"]), cast(proj["k"]), cast(proj["v"])
            beta, g = cast(proj["beta"]), cast(proj["g_log"])
            if base.startswith("rec_"):
                o, s = ref.recurrent_gdn(q, k, v, beta, g, initial_state=cast(s0))
            else:
                o, s = chunked.chunked_gdn(q, k, v, beta, g, chunk_size=cfg.CHUNK_SIZE_DEFAULT,
                                           initial_state=cast(s0))
            target = torch.from_numpy(data[name])
            m = compare(o, target) if "out" in base else compare(s, target)
            anchor.append({"key": name, "form": "recurrent" if base.startswith("rec_") else "chunked_b64",
                           "dtype": kind, "initial_state": "nonzero", **m})
    anchor_bad = [a for a in anchor if a["max_abs_error"] >= (1e-10 if a["dtype"] == "fp64" else 2e-4)]
    print(f"[B] fixtures 锚点：{len(anchor)} 张量，超容差 {len(anchor_bad)}")

    # ---- C. 确定性 ----
    x = torch.randn(2048, cfg.D_MODEL,
                    generator=torch.Generator().manual_seed(cfg.SEED_BASE + 7000 + 2048),
                    dtype=torch.float32)
    proj = ref.project_inputs(x, weights, "scalar")
    runs = []
    for _ in range(3):
        o, s = chunked.chunked_gdn(proj["q"], proj["k"], proj["v"], proj["beta"], proj["g_log"],
                                   chunk_size=64)
        runs.append((o.clone(), s.clone()))
    det_chunked = all(torch.equal(runs[0][0], r[0]) and torch.equal(runs[0][1], r[1]) for r in runs[1:])
    runs_r = []
    for _ in range(3):
        o, s = ref.recurrent_gdn(proj["q"], proj["k"], proj["v"], proj["beta"], proj["g_log"])
        runs_r.append((o.clone(), s.clone()))
    det_rec = all(torch.equal(runs_r[0][0], r[0]) and torch.equal(runs_r[0][1], r[1]) for r in runs_r[1:])
    print(f"[C] 确定性 chunked={det_chunked} recurrent={det_rec}")

    result = {
        "schema_version": "kernel-chapter28.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "config": {
            "d_model": cfg.D_MODEL, "heads": cfg.NUM_KEY_HEADS,
            "head_k_dim": cfg.HEAD_K_DIM, "head_v_dim": cfg.HEAD_V_DIM,
            "scale": cfg.SCALE, "qk_l2norm": True,
        },
        "weight_sha256_check": weight_check,
        "main_pair": groups,
        "fixture_anchor": anchor,
        "fixture_anchor_over_tolerance": anchor_bad,
        "determinism": {"chunked_bitwise_3runs": det_chunked, "recurrent_bitwise_3runs": det_rec},
        "notes": {
            "fp16io": "fp16 存储半精度 + fp32 运算（FLA 核真实口径）；全程 fp16 的"
                      "状态回路发散（灾难性消除）由 smoke_test §8 记录，不作真值",
            "main_pair_definition": "同权重同输入下 chunked(B) vs 逐 Token 递推，"
                                    "同 dtype 现场对算（§9 线性注意力对照规则）",
        },
    }
    write_results_json(HERE / "results" / f"correctness-results-{args.backend}.json", result)

    # 摘要打印
    print("\n== 等价验证摘要（chunked vs 递推，相对误差 max_abs/|rec|max）==")
    for variant in ("scalar", "vector"):
        for dtype_tag in ("fp64", "fp32", "fp16io"):
            rows = [g for g in groups if g["gate_variant"] == variant and g["dtype"] == dtype_tag]
            worst = max(rows, key=lambda r: r["output"]["max_abs_error"] / max(r["rec_output_absmax"], 1e-30))
            print(f"  {variant:7s} {dtype_tag:7s} worst N={worst['seq_len']} B={worst['chunk_size']} "
                  f"rel_max={worst['output']['max_abs_error']/max(worst['rec_output_absmax'],1e-30):.3e} "
                  f"cos={worst['output']['cosine_similarity']:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
