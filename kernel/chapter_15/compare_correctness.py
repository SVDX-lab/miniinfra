#!/usr/bin/env python3
"""第 15 期正确性对照（误差分层主实验）。

链路（fixtures 共用一份：本地 CPU 生成，随代码上传服务器）：
1. backend=cpu：fixtures 自洽校验（tiny 三重循环）+ 分层 La1/La2/La3 报告
   （CPU 列的正式产物 = 误差分层的第 (a) 层）；
2. backend=cuda/musa：int8_gemm_bench run 模式跑同一 fixtures：
   - fp16 kernel（物化 FP16 权重）→ Lc 层（kernel FP16 对照）；
   - int8_fused kernel（融合反量化）→ Lb 层（量化 + kernel 实现误差）；
   - int8_fused vs y_a1（同量化权重的 FP32 参考）→ kernel 实现误差叠加量；
   - materialize 模式 → 物化反量化位级一致性；
   - 确定性：int8_fused 连续 3 次运行位级一致。

用法：
    python compare_correctness.py --backend cpu
    python compare_correctness.py --backend musa --binary implementations/musa/int8_gemm_bench
    python compare_correctness.py --backend cuda --binary implementations/cuda/int8_gemm_bench
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiment_utils import (  # noqa: E402
    SCHEMA_VERSION,
    comparison_metrics_np,
    device_environment_record,
    sha256_of_array,
    write_results_json,
)

FIXTURES_DEFAULT = ROOT / "fixtures" / "correctness.npz"

# 正确性扫描的 kernel 配置（覆盖 16/32/64 tile + 两档寄存器分块）
CORRECTNESS_CONFIGS = [
    ("16x16x16", "2x2"),
    ("32x32x8", "2x2"),
    ("32x32x8", "4x4"),
    ("64x64x8", "4x4"),
]


def load_fixtures(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    with path.with_name("manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return arrays, manifest


def run_binary(binary: Path, args: list[str]) -> None:
    result = subprocess.run(
        [str(binary), *args], capture_output=True, text=True, cwd=str(binary.parent)
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"二进制失败 {args}\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )


def save_raw(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(array).tobytes())


def load_raw(path: Path, shape: tuple[int, int], dtype: np.dtype) -> np.ndarray:
    data = np.fromfile(path, dtype=dtype)
    if data.size != shape[0] * shape[1]:
        raise RuntimeError(f"输出尺寸不符 {path}: {data.size} != {shape}")
    return data.reshape(shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--binary", default="")
    parser.add_argument("--fixtures", default=str(FIXTURES_DEFAULT))
    parser.add_argument("--out", default="")
    arguments = parser.parse_args()

    arrays, manifest = load_fixtures(Path(arguments.fixtures))
    results: dict = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "fixtures": str(Path(arguments.fixtures).resolve()),
        "manifest_combos": [],
        "records": [],
    }
    # fixtures 完整性：逐张量 SHA256 对拍（ep21 KI-21-06 教训：上服务器先验完整性）
    mismatch = []
    for combo in manifest["combos"]:
        key = f"{combo['combo']}_{combo['m']}x{combo['k']}x{combo['n']}"
        for base, want in combo["sha256"].items():
            got = sha256_of_array(arrays[f"{key}__{base}"])
            if got != want:
                mismatch.append(f"{key}__{base}")
    results["fixtures_integrity"] = {
        "checked_tensors": sum(len(c["sha256"]) for c in manifest["combos"]),
        "mismatch": mismatch,
        "ok": not mismatch,
    }
    if mismatch:
        print(f"[FATAL] fixtures SHA256 不一致：{mismatch[:5]}...（KI-21-06 教训条款）")
        return 2

    env = None
    if arguments.backend != "cpu":
        if not arguments.binary:
            print("[FATAL] gpu 后端需要 --binary")
            return 2
        binary = Path(arguments.binary).resolve()
        if not binary.exists():
            print(f"[FATAL] 二进制不存在：{binary}")
            return 2
        try:
            import torch

            env = device_environment_record(
                torch.device(arguments.backend if arguments.backend != "musa" else "musa")
            )
        except Exception as exc:  # 环境记录失败不阻塞正确性
            env = {"error": str(exc)}

    if arguments.backend == "cpu":
        run_cpu_column(arrays, manifest, results)
    else:
        run_gpu_column(arrays, manifest, results, binary, env, arguments.backend)

    out_path = (
        Path(arguments.out)
        if arguments.out
        else ROOT / "results" / f"correctness-results-{arguments.backend}.json"
    )
    write_results_json(out_path, results)
    failed = sum(1 for r in results["records"] if r.get("status") == "FAIL")
    print(f"[compare] 完成：{len(results['records'])} 条记录，FAIL {failed}")
    return 1 if failed else 0


def run_cpu_column(arrays: dict, manifest: dict, results: dict) -> None:
    """CPU 列 = 误差分层第 (a) 层的正式产物 + fixtures 自洽校验。"""
    results["column_note"] = (
        "CPU 参考列：La1/La2/La3 分层（量化误差 vs FP16 存储误差），"
        "与 GPU 列的 Lb/Lc（kernel 实现误差）分开归因（大纲 §9 行 476）"
    )
    for combo in manifest["combos"]:
        key = f"{combo['combo']}_{combo['m']}x{combo['k']}x{combo['n']}"
        y_truth = arrays[f"{key}__y_truth"]
        record = {
            "combo": combo["combo"],
            "m": combo["m"],
            "k": combo["k"],
            "n": combo["n"],
            "layers": {
                "La1_quant_only_fp32": comparison_metrics_np(
                    arrays[f"{key}__y_a1"], y_truth
                ),
                "La2_quant_fp16weight_engine": comparison_metrics_np(
                    arrays[f"{key}__y_a2"], y_truth
                ),
                "La3_fp16_no_quant_engine": comparison_metrics_np(
                    arrays[f"{key}__y_a3"], y_truth
                ),
            },
            "status": "PASS",
        }
        # 自洽：La 层余弦相似度必须 ≥ 0.9999（真值链完整性）
        for name, met in record["layers"].items():
            if met["cosine_similarity"] < 0.9999 or met["nan_count"] > 0:
                record["status"] = "FAIL"
        results["records"].append(record)
        print(
            f"[cpu] {key}: La1 max={record['layers']['La1_quant_only_fp32']['max_abs_error']:.3e} "
            f"La3 max={record['layers']['La3_fp16_no_quant_engine']['max_abs_error']:.3e} "
            f"{record['status']}"
        )


def run_gpu_column(
    arrays: dict, manifest: dict, results: dict, binary: Path, env: dict | None, backend: str
) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ep16cmp_"))
    try:
        for combo in manifest["combos"]:
            key = f"{combo['combo']}_{combo['m']}x{combo['k']}x{combo['n']}"
            m, k, n = combo["m"], combo["k"], combo["n"]
            a16 = arrays[f"{key}__A16"]
            w16 = arrays[f"{key}__W16"]
            q = arrays[f"{key}__q"]
            scale = arrays[f"{key}__scale"]
            y_truth = arrays[f"{key}__y_truth"]
            y_a1 = arrays[f"{key}__y_a1"]

            a_file, out_file = tmp / "a.bin", tmp / "c.bin"
            save_raw(a_file, a16)

            record = {
                "combo": combo["combo"],
                "m": m,
                "k": k,
                "n": n,
                "warmup_note": "run 模式单次执行（无计时）",
                "kernels": {},
                "status": "PASS",
            }

            for tile, reg in CORRECTNESS_CONFIGS:
                # ---- fp16 kernel（Lc 层）----
                w_file = tmp / "w16.bin"
                save_raw(w_file, w16)
                run_binary(
                    binary,
                    [
                        "run",
                        "--m", str(m), "--k", str(k), "--n", str(n),
                        "--kernel", "fp16",
                        "--tile", tile, "--reg", reg,
                        "--in-a", str(a_file), "--in-w", str(w_file), "--out-c", str(out_file),
                    ],
                )
                c_fp16 = load_raw(out_file, (m, n), np.float16).astype(np.float32)
                lc = comparison_metrics_np(c_fp16, y_truth)

                # ---- int8_fused kernel（Lb 层 + 对 y_a1 的实现误差）----
                wq_file, scale_file = tmp / "w8.bin", tmp / "scale.bin"
                save_raw(wq_file, q)
                save_raw(scale_file, scale)
                run_binary(
                    binary,
                    [
                        "run",
                        "--m", str(m), "--k", str(k), "--n", str(n),
                        "--kernel", "int8_fused",
                        "--tile", tile, "--reg", reg,
                        "--in-a", str(a_file), "--in-w", str(wq_file),
                        "--in-scale", str(scale_file), "--out-c", str(out_file),
                    ],
                )
                c_q1 = load_raw(out_file, (m, n), np.float16).astype(np.float32)
                run_binary(
                    binary,
                    [
                        "run",
                        "--m", str(m), "--k", str(k), "--n", str(n),
                        "--kernel", "int8_fused",
                        "--tile", tile, "--reg", reg,
                        "--in-a", str(a_file), "--in-w", str(wq_file),
                        "--in-scale", str(scale_file), "--out-c", str(out_file),
                    ],
                )
                c_q2 = load_raw(out_file, (m, n), np.float16).astype(np.float32)
                run_binary(
                    binary,
                    [
                        "run",
                        "--m", str(m), "--k", str(k), "--n", str(n),
                        "--kernel", "int8_fused",
                        "--tile", tile, "--reg", reg,
                        "--in-a", str(a_file), "--in-w", str(wq_file),
                        "--in-scale", str(scale_file), "--out-c", str(out_file),
                    ],
                )
                c_q3 = load_raw(out_file, (m, n), np.float16).astype(np.float32)

                lb = comparison_metrics_np(c_q1, y_truth)
                lb_vs_a1 = comparison_metrics_np(c_q1, y_a1)
                lb_vs_a2 = comparison_metrics_np(
                    c_q1, arrays[f"{key}__y_a2"]
                )
                deterministic = bool(
                    np.array_equal(c_q1, c_q2) and np.array_equal(c_q2, c_q3)
                )

                record["kernels"][f"{tile}_reg{reg}"] = {
                    "Lc_fp16_kernel_vs_truth": lc,
                    "Lb_int8_fused_vs_truth": lb,
                    "Lb_vs_ya1_impl_delta": lb_vs_a1,
                    "Lb_vs_ya2_engine_quant_fp16": lb_vs_a2,
                    "deterministic_3x": deterministic,
                }
                if lb["nan_count"] > 0 or not deterministic or lc["nan_count"] > 0:
                    record["status"] = "FAIL"
                print(
                    f"[{backend}] {key} {tile}_reg{reg}: "
                    f"Lb max={lb['max_abs_error']:.3e} Lc max={lc['max_abs_error']:.3e} "
                    f"impl_delta={lb_vs_a1['max_abs_error']:.3e} det={deterministic} "
                    f"{record['status']}"
                )

            # ---- materialize 模式位级一致性（每 combo 一次，32x32x8_reg2x2 即可）----
            wq_file, scale_file = tmp / "w8.bin", tmp / "scale.bin"
            wmat_file = tmp / "wmat.bin"
            save_raw(wq_file, q)
            save_raw(scale_file, scale)
            run_binary(
                binary,
                [
                    "materialize",
                    "--k", str(k), "--n", str(n),
                    "--in-w", str(wq_file), "--in-scale", str(scale_file),
                    "--out-w", str(wmat_file),
                ],
            )
            w_mat = load_raw(wmat_file, (k, n), np.float16)
            w_ref = (q.astype(np.float32) * scale[None, :]).astype(np.float16)
            bit_exact = bool(np.array_equal(w_mat, w_ref))
            record["materialize_bit_exact"] = bit_exact
            if not bit_exact:
                ulp = np.abs(w_mat.astype(np.int16) - w_ref.astype(np.int16))
                record["materialize_note"] = {
                    "diff_count": int((ulp != 0).sum()),
                    "max_ulp": int(ulp.max()) if ulp.size else 0,
                }
                # 物化路径允许 ±1 ulp（fp32 乘 + RNE 与 kernel 路径的舍入差），>1 ulp 记 FAIL
                if record["materialize_note"]["max_ulp"] > 1:
                    record["status"] = "FAIL"
            print(f"[{backend}] {key} materialize bit_exact={bit_exact} {record['status']}")

            results["records"].append(record)
        if env is not None:
            results["environment"] = env
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
