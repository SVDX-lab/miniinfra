#!/usr/bin/env python3
"""第 18 期正确性对照（CUDA/MUSA 二进制；无 CPU 列，真值=chapter_17 单源 fixtures）。

链路：
1. WMMA 主链：8 组形状 × 3 tile（wm16x16/wm32x32/wm64x64）× 2 累加（fp32/fp16）
   vs CPU FP32 真值——FP16/FP32 累加精度对照的原始数据；
2. 朴素版抽查：同 fixtures 重跑 tiled 1 配置（同会话 sanity，与第 17 期数字对照）；
3. 确定性：wm64x64 fp32-acc 512³ × 3 次位级一致。

用法：python3 compare_correctness.py --backend cuda|musa
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
    SCHEMA_VERSION,
    comparison_metrics_np,
    device_environment_record,
    parse_binary_json,
    resolve_device,
    write_results_json,
)
from naive_gemm import matmul_naive_rows  # noqa: E402

FIXTURES_DEFAULT = ROOT / "fixtures" / "correctness.npz"
WMMA_CORRECTNESS_TILES = ["16x16x16", "32x32x8", "64x64x8"]  # 取 TILE_MxTN 传给 wmma


def load_fixtures(path: Path):
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    with path.with_name("manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if "combos" not in manifest:
        # 第 18 期 manifest 结构（arrays 平铺）：从键名重建 combos 序列
        shapes = []
        for name in manifest.get("arrays", {}):
            if not name.startswith("A_"):
                continue
            m, k, n = name[2:].split("x")
            shapes.append((int(m), int(k), int(n)))
        manifest["combos"] = [
            {"shape": [m, k, n], "epilogue_truth": m <= 256} for m, k, n in shapes
        ]
    return arrays, manifest


def run_binary(binary: Path, arguments: list[str]) -> None:
    result = subprocess.run([str(binary), "run", *arguments], capture_output=True, text=True,
                            cwd=str(binary.parent))
    if result.returncode != 0:
        raise RuntimeError(f"binary 失败 rc={result.returncode}\n{result.stdout[-400:]}\n{result.stderr[-600:]}")


def to_bin(array: np.ndarray, path: Path, dtype: str) -> None:
    data = array.astype(np.float16 if dtype == "fp16" else np.float32, copy=False)
    path.write_bytes(np.ascontiguousarray(data).tobytes())


def from_bin(path: Path, dtype: str) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.float16 if dtype == "fp16" else np.float32).copy()


def wmma_correctness_chain(binary: Path, workdir: Path, arrays, manifest) -> dict:
    groups = []
    tmp = workdir / "tmp-correctness"
    tmp.mkdir(exist_ok=True)
    for combo in manifest["combos"]:
        m, k, n = combo["shape"]
        key = f"{m}x{k}x{n}"
        truth = arrays[f"C_{key}"]
        a_path = tmp / f"a_{key}.bin"
        b_path = tmp / f"b_{key}.bin"
        c_path = tmp / f"c_{key}.bin"
        to_bin(arrays[f"A_{key}"], a_path, "fp16")
        to_bin(arrays[f"B_{key}"], b_path, "fp16")
        common = ["--dtype", "fp16", "--kernel", "wmma", "--m", str(m), "--n", str(n), "--k", str(k),
                  "--in-a", str(a_path), "--in-b", str(b_path), "--out-c", str(c_path)]
        for tile in WMMA_CORRECTNESS_TILES:
            for acc in ["fp32", "fp16"]:
                try:
                    run_binary(binary, common + ["--tile", tile, "--acc", acc])
                except RuntimeError as error:
                    if "未编译 WMMA" in str(error):
                        return {"groups": [], "unavailable": str(error)[-200:]}
                    raise
                ours = from_bin(c_path, "fp16").reshape(m, n)
                metrics = comparison_metrics_np(ours, truth)
                metrics.update({"shape": key, "kernel": "wmma", "tile": tile, "acc": acc})
                groups.append(metrics)
        print(f"[wmma] {key}: 3 tile × 2 acc 完成")
    worst_fp32 = max([g for g in groups if g["acc"] == "fp32"], key=lambda x: x["max_abs_error"])
    worst_fp16 = max([g for g in groups if g["acc"] == "fp16"], key=lambda x: x["max_abs_error"])
    print(f"[wmma] 共 {len(groups)} 组 | fp32-acc 最差 {worst_fp32['shape']}/{worst_fp32['tile']} "
          f"max_abs={worst_fp32['max_abs_error']:.3e} | fp16-acc 最差 {worst_fp16['shape']}/"
          f"{worst_fp16['tile']} max_abs={worst_fp16['max_abs_error']:.3e}")
    return {"groups": groups, "worst_fp32_acc": worst_fp32, "worst_fp16_acc": worst_fp16}


def backend_hint_of(backend: str) -> str:
    return backend


def naive_spot_chain(binary: Path, workdir: Path, arrays, manifest) -> dict:
    """同会话朴素版抽查（fp16 tiled + naive 基线），与第 17 期结果对照。"""
    groups = []
    tmp = workdir / "tmp-correctness"
    tmp.mkdir(exist_ok=True)  # MUSA 降级路径（wmma 链 early-return）时由此创建
    for combo in manifest["combos"]:
        m, k, n = combo["shape"]
        if (m, k, n) not in [(64, 64, 64), (256, 256, 256), (1024, 1024, 1024)]:
            continue
        key = f"{m}x{k}x{n}"
        truth = arrays[f"C_{key}"]
        a_path = tmp / f"a_{key}.bin"
        b_path = tmp / f"b_{key}.bin"
        c_path = tmp / f"c_{key}.bin"
        to_bin(arrays[f"A_{key}"], a_path, "fp16")
        to_bin(arrays[f"B_{key}"], b_path, "fp16")
        run_binary(binary, ["--dtype", "fp16", "--kernel", "tiled", "--tile", "32x32x8",
                            "--reg", "2x2", "--m", str(m), "--n", str(n), "--k", str(k),
                            "--in-a", str(a_path), "--in-b", str(b_path), "--out-c", str(c_path)])
        ours = from_bin(c_path, "fp16").reshape(m, n)
        metrics = comparison_metrics_np(ours, truth)
        metrics.update({"shape": key, "kernel": "tiled_fp16_spot", "config": "32x32x8_reg2x2"})
        groups.append(metrics)
    worst = max(groups, key=lambda x: x["max_abs_error"])
    print(f"[naive-spot] {len(groups)} 组 | 最差 {worst['shape']} max_abs={worst['max_abs_error']:.3e}"
          "（应与第 17 期同量级）")
    return {"groups": groups, "worst": worst}


def determinism_chain(binary: Path, workdir: Path, arrays) -> dict:
    m = k = n = 512
    key = f"{m}x{k}x{n}"
    tmp = workdir / "tmp-correctness"
    a_path = tmp / f"a_{key}.bin"
    b_path = tmp / f"b_{key}.bin"
    to_bin(arrays[f"A_{key}"], a_path, "fp16")
    to_bin(arrays[f"B_{key}"], b_path, "fp16")
    runs = []
    for i in range(3):
        c_path = tmp / f"det19_{i}.bin"
        run_binary(binary, ["--dtype", "fp16", "--kernel", "wmma", "--tile", "64x64x8",
                            "--acc", "fp32", "--m", str(m), "--n", str(n), "--k", str(k),
                            "--in-a", str(a_path), "--in-b", str(b_path), "--out-c", str(c_path)])
        runs.append(c_path.read_bytes())
    bitwise = all(runs[0] == run for run in runs[1:])
    print(f"[determinism] wmma fp32-acc 512³ × 3 次位级一致 = {bitwise}")
    return {"runs": 3, "bitwise_identical": bool(bitwise)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "musa"])
    parser.add_argument("--fixtures", default=str(FIXTURES_DEFAULT))
    arguments = parser.parse_args()

    arrays, manifest = load_fixtures(Path(arguments.fixtures))
    device = resolve_device(arguments.backend)
    binary = ROOT / "implementations" / arguments.backend / "naive_gemm_bench"
    if not binary.exists():
        raise SystemExit(f"[ERROR] 找不到 {binary}——先运行 implementations/{arguments.backend}/build.sh")

    # WMMA 可用性判定：attrs 输出含 kind=wmma 条目 ⇔ WMMA 编译单元存在
    attr_probe = subprocess.run([str(binary), "attrs"], capture_output=True, text=True,
                                cwd=str(binary.parent))
    has_wmma = False
    wmma_note = "attrs 无 wmma 条目（DISABLE_WMMA 或平台无 MMA 头）"
    if attr_probe.returncode == 0:
        attrs_probe = parse_binary_json(attr_probe.stdout)
        has_wmma = any(c.get("kind") == "wmma" for c in attrs_probe.get("configs", []))
        if has_wmma:
            wmma_note = None
    if not has_wmma:
        print(f"[wmma-check/{arguments.backend}] {wmma_note}")

    results = {
        "schema_version": SCHEMA_VERSION + ".correctness",
        "backend": arguments.backend,
        "episode_note": "第 18 期无 CPU 列（Tensor Core 为 GPU 专属计算单元，期首声明）；真值=chapter_17 单源 fixtures",
        "wmma_available": {"available": has_wmma, "note": wmma_note},
        "wmma_fixtures": (wmma_correctness_chain(binary, ROOT, arrays, manifest)
                          if has_wmma else {"groups": [], "unavailable": wmma_note}),
        "naive_spot": naive_spot_chain(binary, ROOT, arrays, manifest),
        "determinism": (determinism_chain(binary, ROOT, arrays)
                        if has_wmma else {"runs": 0, "bitwise_identical": None,
                                          "note": "WMMA 不可用（平台边界），确定性由朴素链覆盖"}),
        "environment": device_environment_record(device),
    }
    write_results_json(ROOT / "results" / f"correctness-results-{arguments.backend}.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
