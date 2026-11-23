#!/usr/bin/env python3
"""第 02 期三平台正确性对照：手写 Kernel vs CPU FP32 串行真值。

对照链（对全部 30 组合同 Shape：6 H × 5 rows）：
1. fixture 完整性：逐张量 SHA256 与 manifest 核对（跨机器共用同一份真值的前提）；
2. serial 自洽（仅 CPU）：naive_rmsnorm_serial(x) 应与真值位级一致（真值本体）；
3. composite vs 真值：FP32（三后端）/ FP16（GPU 后端）—— 平台原生路径对照（第 01 期口径延续）；
4. kernel vs 真值：FP32 + FP16 各 30 组 —— 本期手写 Kernel 的主对照；
   kernel 路径自动选择：torch 扩展（首选）或 独立可执行程序+文件 IO（退路，
   MUSA 扩展路径不通时自动启用，结果标注 runner）；
5. 确定性：同 Tile 3 次位级一致；跨 Tile（128 vs 256）位级差异记录（第 07 期引子）；
6. 边界行为：全零行、NaN 行、FP16 大数、H=0 空 Tensor、非连续输入、非法 tile。

用法：
    python compare_correctness.py --backend cuda
    python compare_correctness.py --backend musa --kernel-backend auto
    python compare_correctness.py --backend cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    comparison_metrics,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from naive_rmsnorm import RMS_NORM_EPS, composite_rmsnorm_torch, naive_rmsnorm_serial  # noqa: E402

FIXTURES_DEFAULT = HERE / "fixtures" / "correctness.npz"


def load_fixtures(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    with path.with_name("manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return arrays, manifest


def fixture_integrity_chain(arrays: dict[str, np.ndarray], manifest: dict) -> dict:
    mismatches = []
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        # 权重按 H 共享（w_{H}），x/y 按 combos（x_{rows}x{H}）。
        tensor_keys = {
            "x": f"x_{rows}x{hidden_size}",
            "w": f"w_{hidden_size}",
            "y": f"y_{rows}x{hidden_size}",
        }
        for name, tensor_key in tensor_keys.items():
            array = arrays[tensor_key]
            digest = hashlib.sha256()
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(array.tobytes())
            if digest.hexdigest() != combo["sha256"][name]:
                mismatches.append(tensor_key)
    total = len(manifest["combos"]) * 3
    print(f"[fixtures] SHA256 完整性：{total - len(mismatches)}/{total} 通过"
          + (f"，不一致: {mismatches}" if mismatches else ""))
    return {"tensors_checked": total, "mismatches": mismatches, "passed": not mismatches}


class ExtRunner:
    """torch 扩展路径（CUDA: cpp_extension.load；MUSA: torch_musa musa_extension）。

    两个平台的构建器都叫 build_ext.py（迁移要求同名同结构），用
    importlib 按路径显式加载避免 sys.path 命名冲突。
    """

    name = "torch-extension"

    def __init__(self, backend: str):
        import importlib.util

        path = HERE / "implementations" / backend / "build_ext.py"
        spec = importlib.util.spec_from_file_location(f"build_ext_{backend}", path)
        module_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_loader)
        self.module = module_loader.get_module()

    def run(self, x: torch.Tensor, w: torch.Tensor, tile: int) -> torch.Tensor:
        return self.module.rmsnorm(x, w, RMS_NORM_EPS, tile)


class BenchRunner:
    """独立可执行程序 + 文件 IO 路径（扩展路径不通时的退路，协议 §6）。"""

    name = "standalone-fileio"

    def __init__(self, backend: str):
        impl_dir = HERE / "implementations" / backend
        self.binary = impl_dir / "build" / "rmsnorm_bench"
        if not self.binary.is_file():
            raise RuntimeError(f"rmsnorm_bench 未构建：{self.binary}")
        self.backend = backend

    def run(self, x: torch.Tensor, w: torch.Tensor, tile: int) -> torch.Tensor:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            x_np = x.detach().float().cpu().contiguous().numpy()
            w_np = w.detach().float().cpu().contiguous().numpy()
            np.asarray(x_np, dtype=np.float32).tofile(tmp_path / "x.bin")
            np.asarray(w_np, dtype=np.float32).tofile(tmp_path / "w.bin")
            dtype = "fp16" if x.dtype == torch.float16 else "fp32"
            command = [
                str(self.binary),
                "--h", str(x.shape[1]), "--rows", str(x.shape[0]),
                "--dtype", dtype, "--tiles", str(tile),
                "--load", str(tmp_path / "x.bin"), str(tmp_path / "w.bin"),
                "--out", str(tmp_path / "y.bin"),
                "--warmup", "1", "--repeats", "1", "--check", "false",
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"rmsnorm_bench 失败（rc={result.returncode}）：{result.stderr[-2000:]}")
            y_np = np.fromfile(tmp_path / "y.bin", dtype=np.float32)
        return torch.from_numpy(y_np.reshape(x.shape)).to(x.device, dtype=x.dtype)


def make_kernel_runner(backend: str, preference: str, device: torch.device):
    if preference == "bench":
        return BenchRunner(backend)
    try:
        runner = ExtRunner(backend)
        # 设备上冒烟：触发真实的扩展加载与一次 kernel 发射。
        x = torch.randn(4, 128, device=device, dtype=torch.float16)
        w = torch.randn(128, device=device, dtype=torch.float16)
        runner.run(x, w, 256)
        return runner
    except Exception as error:  # noqa: BLE001
        if preference == "ext":
            raise
        print(f"[kernel] torch 扩展路径不可用（{type(error).__name__}: {error}），"
              f"退到独立可执行程序 + 文件 IO 对比路径")
        return BenchRunner(backend)


def kernel_chain(arguments, device: torch.device, arrays, runner, dtype) -> dict:
    weight_cache: dict[int, torch.Tensor] = {}
    groups = []
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{hidden_size}"
        x_fp32 = torch.from_numpy(arrays[f"x_{key}"])
        expected = torch.from_numpy(arrays[f"y_{key}"])
        if hidden_size not in weight_cache:
            weight_cache[hidden_size] = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(
                device=device, dtype=dtype)
        x = x_fp32.to(device=device, dtype=dtype)
        tile = arguments.tile
        start = time.perf_counter()
        result = runner.run(x, weight_cache[hidden_size], tile)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = comparison_metrics(result, expected)
        vec_width = 8 if dtype == torch.float16 else 4
        metrics.update({
            "rows": rows,
            "hidden_size": hidden_size,
            "dtype": "fp16" if dtype == torch.float16 else "fp32",
            "tile": tile,
            "vec_path": hidden_size % vec_width == 0,
            "call_overhead_ms": round(elapsed_ms, 2),
        })
        groups.append(metrics)
    worst = max(groups, key=lambda item: item["max_abs_error"])
    nan_total = sum(item["nan_count"] for item in groups)
    print(f"[kernel/{arguments.backend}/{dtype}] {len(groups)} 组（runner={runner.name}, tile={arguments.tile}）| "
          f"最差组 {worst['rows']}x{worst['hidden_size']} max_abs={worst['max_abs_error']:.3e} "
          f"cos={worst['cosine_similarity']:.8f} NaN 总数={nan_total}")
    return {"dtype": "fp16" if dtype == torch.float16 else "fp32",
            "runner": runner.name, "tile": arguments.tile,
            "groups": groups, "worst_group": worst, "nan_total": nan_total}


def composite_chain(arguments, device: torch.device, arrays, dtype) -> dict:
    weight_cache: dict[int, torch.Tensor] = {}
    groups = []
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{hidden_size}"
        x_fp32 = torch.from_numpy(arrays[f"x_{key}"])
        expected = torch.from_numpy(arrays[f"y_{key}"])
        if hidden_size not in weight_cache:
            weight_cache[hidden_size] = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(
                device=device, dtype=dtype)
        x = x_fp32.to(device=device, dtype=dtype)
        result = composite_rmsnorm_torch(x, weight_cache[hidden_size], RMS_NORM_EPS)
        metrics = comparison_metrics(result, expected)
        metrics.update({"rows": rows, "hidden_size": hidden_size,
                        "dtype": "fp16" if dtype == torch.float16 else "fp32"})
        groups.append(metrics)
    worst = max(groups, key=lambda item: item["max_abs_error"])
    print(f"[composite/{arguments.backend}/{dtype}] {len(groups)} 组 | "
          f"最差组 {worst['rows']}x{worst['hidden_size']} max_abs={worst['max_abs_error']:.3e} "
          f"cos={worst['cosine_similarity']:.8f}")
    return {"dtype": "fp16" if dtype == torch.float16 else "fp32",
            "groups": groups, "worst_group": worst}


def serial_selfcheck_chain(arrays) -> dict:
    """CPU 真值本体自洽校验：串行实现重算应与 fixture 真值位级一致。"""
    mismatches = 0
    max_diff = 0.0
    for combo in manifest["combos"]:
        key = f"{combo['rows']}x{combo['hidden_size']}"
        y_recomputed = naive_rmsnorm_serial(
            arrays[f"x_{key}"], arrays[f"w_{combo['hidden_size']}"], RMS_NORM_EPS)
        diff = float(np.abs(y_recomputed - arrays[f"y_{key}"]).max())
        max_diff = max(max_diff, diff)
        if diff != 0.0:
            mismatches += 1
    print(f"[serial] 真值自洽：{len(manifest['combos']) - mismatches}/{len(manifest['combos'])} 组位级一致"
          f"（max_diff={max_diff:.1e}）")
    return {"groups": len(manifest["combos"]), "bitwise_mismatch_groups": mismatches,
            "max_abs_diff": max_diff}


def determinism_chain(arguments, device: torch.device, arrays, runner) -> dict:
    rows, hidden_size = 128, 1024
    key = f"{rows}x{hidden_size}"
    x = torch.from_numpy(arrays[f"x_{key}"]).to(device=device, dtype=torch.float16)
    w = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(device=device, dtype=torch.float16)
    runs = [runner.run(x, w, arguments.tile).detach().float().cpu().contiguous() for _ in range(3)]
    bitwise = all(
        bytes(runs[0].numpy().tobytes()) == bytes(run.numpy().tobytes())
        for run in runs[1:]
    )
    cross_tile = None
    try:
        other = runner.run(x, w, 128 if arguments.tile != 128 else 256)
        cross_tile = {
            "tiles": [arguments.tile, 128 if arguments.tile != 128 else 256],
            "bitwise_identical": bool(torch.equal(runs[0], other.detach().float().cpu())),
            "max_abs_diff": float((runs[0] - other.detach().float().cpu()).abs().max().item()),
            "note": "跨 Tile 归约分组不同，FP32 结果可能位级不同 —— 第 07 期确定性主题引子",
        }
    except Exception as error:  # noqa: BLE001
        cross_tile = {"error": str(error)}
    print(f"[determinism] 同 tile 3 次位级一致: {bitwise} | 跨 tile 位级一致: {cross_tile.get('bitwise_identical')}")
    return {"runs": 3, "tile": arguments.tile, "bitwise_identical": bool(bitwise),
            "cross_tile": cross_tile}


def edge_chain(arguments, device: torch.device, arrays, runner) -> dict:
    if isinstance(runner, BenchRunner):
        print("[edge] 文件 IO 退路不支持逐条边界注入（bench 仅整张量路径），跳过并记录")
        return {"runner": runner.name, "skipped": True}
    checks = {}
    w = torch.randn(1024, device=device, dtype=torch.float16)

    # 全零行：输出全零（eps 在根号内），无 NaN。
    x = torch.randn(3, 1024, device=device, dtype=torch.float16)
    x[1] = 0
    y = runner.run(x, w, arguments.tile)
    checks["zero_row"] = {
        "all_zero": bool((y[1] == 0).all()),
        "no_nan": bool(~torch.isnan(y[1]).any()),
        "neighbors_finite": bool(torch.isfinite(y[[0, 2]]).all()),
    }

    # NaN 行：按行传播。
    x = torch.randn(2, 1024, device=device, dtype=torch.float16)
    x[0, 0] = float("nan")
    y = runner.run(x, w, arguments.tile)
    checks["nan_row"] = {
        "row0_all_nan": bool(torch.isnan(y[0]).all()),
        "row1_finite": bool(torch.isfinite(y[1]).all()),
    }

    # FP16 大数：平方和升 FP32 不溢出。
    x = torch.full((1, 1024), 65504.0, device=device, dtype=torch.float16)
    y = runner.run(x, w, arguments.tile)
    checks["fp16_large"] = {"finite": bool(torch.isfinite(y).all())}

    # H=0 空 Tensor：显式报错。
    try:
        runner.run(torch.randn(2, 0, device=device, dtype=torch.float16),
                   torch.zeros(0, device=device, dtype=torch.float16), arguments.tile)
        checks["h_zero"] = {"raised": False}
    except Exception as error:  # noqa: BLE001
        checks["h_zero"] = {"raised": True, "error_type": type(error).__name__,
                            "message": str(error)[:300]}

    # rows=0：返回空 Tensor（合同条目）。
    y = runner.run(torch.randn(0, 1024, device=device, dtype=torch.float16), w, arguments.tile)
    checks["rows_zero"] = {"shape": list(y.shape)}

    # 非连续输入：显式报错（合同：第 06 期主题，本期不支持）。
    base = torch.randn(1024, 128, device=device, dtype=torch.float16)
    try:
        runner.run(base.t(), torch.randn(128, device=device, dtype=torch.float16), arguments.tile)
        checks["non_contiguous"] = {"raised": False}
    except Exception as error:  # noqa: BLE001
        checks["non_contiguous"] = {"raised": True, "error_type": type(error).__name__,
                                    "message": str(error)[:300]}

    # 非法 tile：显式报错（编译期实例化集合之外）。
    try:
        runner.run(torch.randn(2, 1024, device=device, dtype=torch.float16), w, 100)
        checks["invalid_tile"] = {"raised": False}
    except Exception as error:  # noqa: BLE001
        checks["invalid_tile"] = {"raised": True, "error_type": type(error).__name__}

    ok = (checks["zero_row"]["all_zero"] and checks["zero_row"]["no_nan"]
          and checks["nan_row"]["row0_all_nan"] and checks["nan_row"]["row1_finite"]
          and checks["fp16_large"]["finite"]
          and checks["h_zero"]["raised"] and checks["non_contiguous"]["raised"]
          and checks["invalid_tile"]["raised"] and checks["rows_zero"]["shape"] == [0, 1024])
    print(f"[edge] 边界行为{'全部符合合同' if ok else '存在不符合项（见 JSON）'}")
    return {"runner": runner.name, "checks": checks, "all_pass": ok}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--kernel-backend", default="auto", choices=["auto", "ext", "bench"])
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--fixtures", default=str(FIXTURES_DEFAULT))
    arguments = parser.parse_args()

    global manifest
    arrays, manifest = load_fixtures(Path(arguments.fixtures))
    device = resolve_device(arguments.backend) if arguments.backend != "cpu" else torch.device("cpu")

    integrity = fixture_integrity_chain(arrays, manifest)
    if not integrity["passed"]:
        raise SystemExit("fixture SHA256 校验失败，终止（跨机器真值不可信）")

    results = {
        "schema_version": "kernel-chapter03.correctness.v1",
        "backend": arguments.backend,
        "tile": arguments.tile,
        "fixture_integrity": integrity,
        "serial_selfcheck": serial_selfcheck_chain(arrays) if arguments.backend == "cpu" else None,
        "composite_fp32": composite_chain(arguments, device, arrays, torch.float32),
        "environment": device_environment_record(device),
    }
    if arguments.backend in ("cuda", "musa"):
        runner = make_kernel_runner(arguments.backend, arguments.kernel_backend, device)
        results["composite_fp16"] = composite_chain(arguments, device, arrays, torch.float16)
        results["kernel_fp32"] = kernel_chain(arguments, device, arrays, runner, torch.float32)
        results["kernel_fp16"] = kernel_chain(arguments, device, arrays, runner, torch.float16)
        results["determinism"] = determinism_chain(arguments, device, arrays, runner)
        results["edge"] = edge_chain(arguments, device, arrays, runner)

    output = HERE / "results" / f"correctness-results-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
