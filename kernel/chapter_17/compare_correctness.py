#!/usr/bin/env python3
"""第 17 期三后端正确性对照。

链路（fixtures 共用一份：本地 CPU FP32 真值，随代码上传两台服务器）：
1. backend=cpu：naive rows 实现 vs fixtures 真值（自洽校验）+ 字面三重循环抽查；
2. backend=cuda/musa：naive_gemm_bench 二进制按多配置跑同一 fixtures：
   - FP32 主线：tiled 三配置 + 零组织基线（naive kernel）；
   - FP16 对照：FP16 输入 + FP32 累加 + FP16 输出 vs FP32 真值（容差单独记录）；
   - epilogue 融合（bias/gelu/silu）vs 分离 kernel（sep2/sep3）vs numpy 真值；
   - 确定性：同一配置连续 3 次运行的位级一致性（无原子、固定累加序）。

用法：
    python compare_correctness.py --backend cuda
    python compare_correctness.py --backend musa
    python compare_correctness.py --backend cpu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    comparison_metrics_np,
    device_environment_record,
    resolve_device,
    SCHEMA_VERSION,
    write_results_json,
)
from gemm_configs import epilogue_code  # noqa: E402
from naive_gemm import (  # noqa: E402
    fused_bias_gelu,
    fused_bias_silu,
    matmul_naive_rows,
    matmul_triple_loop,
)

FIXTURES_DEFAULT = ROOT / "fixtures" / "correctness.npz"

# 正确性扫描的 kernel 配置：覆盖不同 TK、不同寄存器分块深度 + 零组织基线
TILED_CORRECTNESS_CONFIGS = ["16x16x16_reg2x2", "32x32x8_reg2x2", "64x64x8_reg4x4", "64x64x8_reg8x8"]
EPILOGUE_CONFIG = "32x32x8_reg2x2"


def load_fixtures(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    with path.with_name("manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return arrays, manifest


def run_binary(binary: Path, arguments: list[str]) -> None:
    result = subprocess.run(
        [str(binary), "run", *arguments], capture_output=True, text=True, cwd=str(binary.parent)
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"binary 失败 rc={result.returncode}\nstdout: {result.stdout[-800:]}\nstderr: {result.stderr[-800:]}"
        )


def to_bin(array: np.ndarray, path: Path, dtype: str) -> None:
    data = array.astype(np.float16 if dtype == "fp16" else np.float32, copy=False)
    path.write_bytes(np.ascontiguousarray(data).tobytes())


def from_bin(path: Path, dtype: str) -> np.ndarray:
    raw = path.read_bytes()
    return np.frombuffer(raw, dtype=np.float16 if dtype == "fp16" else np.float32).copy()


def gemm_correctness_chain(binary: Path | None, backend: str, workdir: Path, arrays, manifest) -> dict:
    groups = []
    tmp = workdir / "tmp-correctness"
    tmp.mkdir(exist_ok=True)

    for combo in manifest["combos"]:
        m, k, n = combo["shape"]
        key = f"{m}x{k}x{n}"
        A = arrays[f"A_{key}"]
        B = arrays[f"B_{key}"]
        truth = arrays[f"C_{key}"]

        if backend == "cpu":
            ours = matmul_naive_rows(A, B)
            metrics = comparison_metrics_np(ours, truth)
            if key == "4x8x4":
                triple = matmul_triple_loop(A, B)
                metrics["triple_loop_bitwise_equal"] = bool(np.array_equal(triple, truth))
            metrics.update({"shape": key, "config": "naive_rows_cpu", "dtype": "fp32"})
            groups.append(metrics)
            print(f"[cpu] {key}: max_abs={metrics['max_abs_error']:.3e}")
            continue

        a_path = tmp / f"a_{key}.bin"
        b_path = tmp / f"b_{key}.bin"
        c_path = tmp / f"c_{key}.bin"
        for dtype in ["fp32", "fp16"]:
            to_bin(A, a_path, dtype)
            to_bin(B, b_path, dtype)
            for config in ["naive"] + TILED_CORRECTNESS_CONFIGS:
                tile, reg = ("32x32x8", "2x2") if config == "naive" else (
                    config.split("_reg")[0], config.split("_reg")[1])
                run_binary(binary, [
                    "--dtype", dtype, "--kernel", "naive" if config == "naive" else "tiled",
                    "--tile", tile, "--reg", reg,
                    "--m", str(m), "--n", str(n), "--k", str(k),
                    "--in-a", str(a_path), "--in-b", str(b_path), "--out-c", str(c_path),
                ])
                ours = from_bin(c_path, dtype).reshape(m, n)
                metrics = comparison_metrics_np(ours, truth)
                metrics.update({"shape": key, "config": config, "dtype": dtype})
                groups.append(metrics)
            print(f"[{backend}] {key} {dtype}: 完成 {len(TILED_CORRECTNESS_CONFIGS) + 1} 个配置")

    worst = max(groups, key=lambda item: item["max_abs_error"])
    print(f"[{backend}] GEMM 链共 {len(groups)} 组 | 最差组 {worst['shape']}/{worst['config']}"
          f"({worst['dtype']}) max_abs={worst['max_abs_error']:.3e} cos={worst['cosine_similarity']:.8f}")
    return {"groups": groups, "worst_group": worst}


def epilogue_correctness_chain(binary: Path | None, backend: str, workdir: Path, arrays, manifest) -> dict:
    """融合 epilogue vs 分离 kernel vs numpy 真值（shape ≤ 256 的组合）。"""
    groups = []
    tmp = workdir / "tmp-correctness"
    tmp.mkdir(exist_ok=True)

    for combo in manifest["combos"]:
        if not combo.get("epilogue_truth"):
            continue
        m, k, n = combo["shape"]
        if backend != "cpu" and m > 256:
            continue
        key = f"{m}x{k}x{n}"
        truth_bias = arrays[f"C_bias_{key}"]
        truth_gelu = arrays[f"C_gelu_{key}"]
        truth_silu = arrays[f"C_silu_{key}"]

        if backend == "cpu":
            ours = matmul_naive_rows(arrays[f"A_{key}"], arrays[f"B_{key}"])
            for act, truth in [("bias", truth_bias), ("gelu", truth_gelu), ("silu", truth_silu)]:
                if act == "bias":
                    fused = ours + arrays[f"bias_{key}"][None, :]
                elif act == "gelu":
                    fused = fused_bias_gelu(ours, arrays[f"bias_{key}"])
                else:
                    fused = fused_bias_silu(ours, arrays[f"bias_{key}"])
                metrics = comparison_metrics_np(fused, truth)
                metrics.update({"shape": key, "act": act, "mode": "fused_numpy"})
                groups.append(metrics)
            continue

        a_path = tmp / f"a_{key}.bin"
        b_path = tmp / f"b_{key}.bin"
        bias_path = tmp / f"bias_{key}.bin"
        c_path = tmp / f"c_{key}.bin"
        for dtype in ["fp32", "fp16"]:
            if dtype == "fp16" and key not in ("64x64x64", "4x8x4"):
                continue  # FP16 epilogue 只抽查两个形状（误差性质由 FP32 + GEMM 链覆盖）
            to_bin(arrays[f"A_{key}"], a_path, dtype)
            to_bin(arrays[f"B_{key}"], b_path, dtype)
            bias_path.write_bytes(np.ascontiguousarray(arrays[f"bias_{key}"]).tobytes())  # bias 恒 FP32
            common = ["--dtype", dtype, "--kernel", "tiled", "--tile", "32x32x8", "--reg", "2x2",
                      "--m", str(m), "--n", str(n), "--k", str(k),
                      "--in-a", str(a_path), "--in-b", str(b_path), "--in-bias", str(bias_path),
                      "--out-c", str(c_path)]
            for act_code, truth in [(1, truth_bias), (2, truth_gelu), (3, truth_silu)]:
                act_name = {1: "bias", 2: "gelu", 3: "silu"}[act_code]
                # bias-only 无 sep3 形态（分离 bias 只有一种单 kernel 方式）
                sep_modes = [0, 2] if act_code == 1 else [0, 2, 3]
                outputs = {}
                for sep in sep_modes:
                    run_binary(binary, common + ["--epilogue", str(act_code), "--sep", str(sep)])
                    outputs[sep] = from_bin(c_path, dtype).reshape(m, n)
                    mode_name = {0: "fused", 2: "sep2", 3: "sep3"}[sep]
                    metrics = comparison_metrics_np(outputs[sep], truth)
                    metrics.update({"shape": key, "act": act_name, "mode": mode_name, "dtype": dtype})
                    groups.append(metrics)
                # 融合 vs 分离的直接差（融合省的不是精度，但少一次 FP16 舍入应更贴近）
                metrics = comparison_metrics_np(outputs[0], outputs[2])
                metrics.update({"shape": key, "act": act_name, "mode": "fused_vs_sep2", "dtype": dtype})
                groups.append(metrics)
            print(f"[{backend}] epilogue {key} {dtype}: bias/gelu/silu × fused/sep2/sep3 完成")

    worst = max(groups, key=lambda item: item["max_abs_error"])
    print(f"[{backend}] epilogue 链共 {len(groups)} 组 | 最差 {worst['shape']}/{worst['act']}"
          f"/{worst['mode']} max_abs={worst['max_abs_error']:.3e}")
    return {"groups": groups, "worst_group": worst}


def determinism_chain(binary: Path, backend: str, workdir: Path, arrays) -> dict | None:
    if backend == "cpu":
        return None
    m = k = n = 512
    key = f"{m}x{k}x{n}"
    tmp = workdir / "tmp-correctness"
    a_path = tmp / f"a_{key}.bin"
    b_path = tmp / f"b_{key}.bin"
    to_bin(arrays[f"A_{key}"], a_path, "fp32")
    to_bin(arrays[f"B_{key}"], b_path, "fp32")
    runs = []
    for i in range(3):
        c_path = tmp / f"det_{i}.bin"
        run_binary(binary, ["--dtype", "fp32", "--kernel", "tiled", "--tile", "32x32x8",
                            "--reg", "2x2", "--m", str(m), "--n", str(n), "--k", str(k),
                            "--in-a", str(a_path), "--in-b", str(b_path), "--out-c", str(c_path)])
        runs.append(c_path.read_bytes())
    bitwise = all(runs[0] == run for run in runs[1:])
    print(f"[{backend}] 确定性：3 次运行位级一致 = {bitwise}")
    return {"runs": 3, "bitwise_identical": bool(bitwise),
            "note": "无原子操作、固定 k 升序累加顺序 → 位级确定"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--fixtures", default=str(FIXTURES_DEFAULT))
    arguments = parser.parse_args()

    arrays, manifest = load_fixtures(Path(arguments.fixtures))
    workdir = ROOT
    binary = None
    device = None

    if arguments.backend == "cpu":
        device = "cpu"
    else:
        device = resolve_device(arguments.backend)
        binary = ROOT / "implementations" / arguments.backend / "naive_gemm_bench"
        if not binary.exists():
            raise SystemExit(f"[ERROR] 找不到 {binary}——先运行 implementations/{arguments.backend}/build.sh")

    env_record = device_environment_record(device) if arguments.backend != "cpu" else {
        "device_type": "cpu",
        "note": "本地/服务器 CPU；完整环境见 benchmark.py 的 environment 段",
    }

    results = {
        "schema_version": SCHEMA_VERSION + ".correctness",
        "backend": arguments.backend,
        "gemm_fixtures": gemm_correctness_chain(binary, arguments.backend, workdir, arrays, manifest),
        "epilogue_fusion": epilogue_correctness_chain(binary, arguments.backend, workdir, arrays, manifest),
        "determinism": determinism_chain(binary, arguments.backend, workdir, arrays) if binary else None,
        "environment": env_record,
    }
    write_results_json(ROOT / "results" / f"correctness-results-{arguments.backend}.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
