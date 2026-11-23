#!/usr/bin/env python3
"""第 27 期 fixtures：固定 seed 合成权重/输入 + 两形式真值（跨机器共享锚点）。

内容（全部进 correctness.npz，savez_compressed）：
  每 (gate_variant ∈ {scalar, vector}) × (N ∈ {1,3,33,127,128}；scalar 另含 2048)：
    rec_out_fp32 / rec_state_fp32        逐 Token 递推 FP32（真值锚点）
    rec_out_fp64 / rec_state_fp64        FP64 代数锚点
    chunk_out_fp32_b64 / chunk_state_fp32_b64   分块形式 B=64 FP32（值透明性）
    chunk_out_fp64_b64
  scalar × N=127 另有 initial_state ≠ 0 情形：rec_out_fp32_s0 / rec_state_fp32_s0
  与 chunk_out_fp64_s0_b64（decode carry 语义）。

权重与 x 不进 npz（纯 torch.Generator 固定 seed 可重建）；权重逐张量 SHA256
记入 manifest.json——目标机上 verify-only 重算整条流水与 npz 真值数值复核
（FP64 容差 1e-10、FP32 容差 2e-4 覆盖跨 torch 版本 GEMM 舍入，ep26 同法）。

用法：
  python generate_fixtures.py                # 生成本地（真值源机器）
  python generate_fixtures.py --verify-only  # 目标机重算复核，不写文件
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

import deltanet_config as cfg  # noqa: E402
import deltanet_chunked as chunked  # noqa: E402
import deltanet_reference as ref  # noqa: E402
from experiment_utils import sha256_of_tensor, write_results_json  # noqa: E402

VARIANT_IDX = {"scalar": 0, "vector": 1}
FIXTURE_NS = (1, 3, 33, 127, 128)           # 两变体共用
FIXTURE_NS_SCALAR_EXTRA = (2048,)           # scalar 独占（控制 npz 体积）
INPUT_SEED_OFFSET = 7000                    # x 的 seed = SEED_BASE + offset + N
INIT_STATE_SEED_OFFSET = 9000


def make_case(seq_len: int, variant: str):
    """按合同 seed 约定重建一份用例（权重共享、x 按 N 派生）。"""
    weights = ref.make_weights(cfg.SEED_BASE)
    gen = torch.Generator().manual_seed(cfg.SEED_BASE + INPUT_SEED_OFFSET + seq_len)
    x = torch.randn(seq_len, cfg.D_MODEL, generator=gen, dtype=torch.float32)
    proj = ref.project_inputs(x, weights, variant)
    return weights, x, proj


def case_keys(seq_len: int, variant: str, suffix: str) -> str:
    return f"{variant}_n{seq_len}_{suffix}"


def run_truth(seq_len: int, variant: str, initial_state: torch.Tensor | None = None,
              include_chunked: bool = True):
    """一个用例的真值（递推 FP32/FP64 [+ 分块 B=64 FP32/FP64]）。"""
    _, _, proj = make_case(seq_len, variant)
    out: dict[str, torch.Tensor] = {}
    for tag, cast in (("fp32", lambda t: t.to(torch.float32)),
                      ("fp64", lambda t: t.to(torch.float64))):
        o, s = ref.recurrent_gdn(
            cast(proj["q"]), cast(proj["k"]), cast(proj["v"]),
            cast(proj["beta"]), cast(proj["g_log"]), initial_state=initial_state,
        )
        out[f"rec_out_{tag}"] = o
        out[f"rec_state_{tag}"] = s
        if include_chunked:
            o_c, s_c = chunked.chunked_gdn(
                cast(proj["q"]), cast(proj["k"]), cast(proj["v"]),
                cast(proj["beta"]), cast(proj["g_log"]),
                chunk_size=cfg.CHUNK_SIZE_DEFAULT, initial_state=initial_state,
            )
            out[f"chunk_out_{tag}_b64"] = o_c
            out[f"chunk_state_{tag}_b64"] = s_c
    return out


def build_arrays() -> dict[str, torch.Tensor]:
    arrays: dict[str, torch.Tensor] = {}
    for variant in ("scalar", "vector"):
        for n in FIXTURE_NS:
            for name, tensor in run_truth(n, variant).items():
                arrays[case_keys(n, variant, name)] = tensor
    # N=2048：只存递推真值（体积控制——chunked 等价由 compare_correctness 现场
    # 对算，fixture 真值只承担跨机锚点职责）
    for name, tensor in run_truth(2048, "scalar", include_chunked=False).items():
        arrays[case_keys(2048, "scalar", name)] = tensor
    # initial_state ≠ 0 情形（decode carry）：scalar × N=127
    gen = torch.Generator().manual_seed(cfg.SEED_BASE + INIT_STATE_SEED_OFFSET + 127)
    s0 = torch.randn(cfg.NUM_VALUE_HEADS, cfg.HEAD_K_DIM, cfg.HEAD_V_DIM,
                     generator=gen, dtype=torch.float32) * 0.1
    for name, tensor in run_truth(127, "scalar", initial_state=s0).items():
        arrays[case_keys(127, "scalar", name + "_s0")] = tensor
    return arrays


def verify_only() -> int:
    """目标机重算整条流水，与 npz 真值数值复核（不写文件）。"""
    npz_path = HERE / "fixtures" / "correctness.npz"
    manifest_path = HERE / "fixtures" / "manifest.json"
    if not npz_path.is_file() or not manifest_path.is_file():
        print("[verify] 找不到 fixtures/correctness.npz 或 manifest.json")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as data:
        bad: list[str] = []
        checked = 0
        for name, array in data.items():
            tensor = torch.from_numpy(array)
            kind = "fp64" if "_fp64" in name else "fp32"
            seq_len = int(name.split("_n")[1].split("_")[0])
            variant = name.split("_n")[0]
            prefix = f"{variant}_n{seq_len}_"
            is_s0 = name.endswith("_s0")
            base = name[len(prefix):]
            if is_s0:
                base = base[:-len("_s0")]
                gen = torch.Generator().manual_seed(cfg.SEED_BASE + INIT_STATE_SEED_OFFSET + seq_len)
                s0 = torch.randn(cfg.NUM_VALUE_HEADS, cfg.HEAD_K_DIM, cfg.HEAD_V_DIM,
                                 generator=gen, dtype=torch.float32) * 0.1
                fresh = run_truth(seq_len, variant, initial_state=s0)
            else:
                include_chunked = base.startswith("chunk_") or seq_len != 2048
                fresh = run_truth(seq_len, variant, include_chunked=include_chunked)
            ref_tensor = fresh[base]
            tol = cfg.FIXTURE_TOL_FP64 if kind == "fp64" else cfg.FIXTURE_TOL_FP32
            err = (tensor.double() - ref_tensor.double()).abs().max().item()
            checked += 1
            if err >= tol:
                bad.append(f"{name}: max_abs={err:.3e} >= {tol:.0e}")
            print(f"[verify] {name}: max_abs={err:.3e} tol={tol:.0e} "
                  f"{'PASS' if err < tol else 'FAIL'}")
        if bad:
            print("[verify] 失败：\n  " + "\n  ".join(bad))
            return 1
        print(f"[verify] 通过：{checked} 张量重算与 npz 真值一致")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true",
                        help="重算整条流水并与 npz 真值数值复核（不写文件）")
    parser.add_argument("--out-dir", type=Path, default=HERE / "fixtures")
    args = parser.parse_args()

    if args.verify_only:
        return verify_only()

    arrays = build_arrays()

    # 权重 SHA256 清单（可重建权重的跨机核对锚）
    weights = ref.make_weights(cfg.SEED_BASE)
    weight_hashes = {name: sha256_of_tensor(tensor) for name, tensor in sorted(weights.items())}

    manifest = {
        "schema_version": "kernel-chapter28.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_base": cfg.SEED_BASE,
        "input_seed_offset": INPUT_SEED_OFFSET,
        "init_state_seed_offset": INIT_STATE_SEED_OFFSET,
        "config": {
            "d_model": cfg.D_MODEL, "num_key_heads": cfg.NUM_KEY_HEADS,
            "num_value_heads": cfg.NUM_VALUE_HEADS, "head_k_dim": cfg.HEAD_K_DIM,
            "head_v_dim": cfg.HEAD_V_DIM, "scale": cfg.SCALE,
            "chunk_size_default": cfg.CHUNK_SIZE_DEFAULT,
        },
        "weights_sha256": weight_hashes,
        "tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                   "sha256": sha256_of_tensor(tensor)}
            for name, tensor in sorted(arrays.items())
        },
        "tolerances": {"fp64": cfg.FIXTURE_TOL_FP64, "fp32": cfg.FIXTURE_TOL_FP32},
        "verify_protocol": (
            "npz 是跨机器共享真值；--verify-only 在目标机重算全流水与 npz 真值做"
            "数值复核（FP64 < 1e-10、FP32 < 2e-4，覆盖 GEMM 舍入差），以 npz "
            "分发值为准；权重/x 不进 npz，按 seed 重建并以 weights_sha256 对拍"
        ),
        "volume_control": (
            "vector 变体真值到 N=128（N=2048 的两形式对照在 compare_correctness "
            "现场计算，不依赖 fixture 真值）"
        ),
    }
    write_results_json(args.out_dir / "manifest.json", manifest)
    np.savez_compressed(args.out_dir / "correctness.npz",
                        **{name: t.numpy() for name, t in arrays.items()})
    total = sum(tensor.numel() * tensor.element_size() for tensor in arrays.values())
    print(f"[fixtures] 写出 {args.out_dir}/correctness.npz（{len(arrays)} 张量, "
          f"~{total/1e6:.1f}MB 未压缩口径）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
