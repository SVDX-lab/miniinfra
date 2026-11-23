#!/usr/bin/env python3
"""第 07 期正确性比较：三实现（det / atomic_shared / atomic_global）× 两 dtype
× 两累加 dtype（atomic）vs CPU FP32 串行真值（fixtures 30 组）。

比较口径复用第 01 期（experiment_utils.comparison_metrics）：max/mean 绝对
误差、相对误差、余弦相似度、NaN/Inf 计数、最大误差位置。

本期新增口径：
- 每实现跑 3 次记录"3 次位级一致"（ep03/ep06 基线的延续：本期把它升格为
  100 次正式主题，3 次是正确性链内的快速哨兵）；
- atomic 版 vs 确定性版在同一输入下的位级差（同 tile）——"同一数学、不同
  归约顺序"的直接证据；
- ACC=half 档（atomic 专属）接第 05 期卡死区口径：H=1024 不卡死但误差
  FP16 级（1e-2~1e-3），H=65536 的卡死行为在 benchmark.py --curve。

用法：
  python compare_correctness.py --backend cpu
  python compare_correctness.py --backend cuda
  python compare_correctness.py --backend musa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import torch  # noqa: E402

from experiment_utils import (  # noqa: E402
    backend_sync,
    comparison_metrics,
    device_environment_record,
    resolve_device,
    sha256_of_tensor,
    write_results_json,
)
from serial_rmsnorm import RMS_NORM_EPS, naive_rmsnorm_serial  # noqa: E402

MAIN_TILE = 256
GLOBAL_SPLITS = 4


def load_kernels_for(backend: str):
    if backend == "cuda":
        sys.path.insert(0, str(HERE / "implementations" / "cuda"))
        from ops_binding import load_kernels  # noqa: E402
        return load_kernels("cuda", "libmini_detatomic_cuda.so")
    if backend == "musa":
        sys.path.insert(0, str(HERE / "implementations" / "musa"))
        from ops_binding import load_kernels  # noqa: E402
        return load_kernels("musa", "libmini_detatomic_musa.so")
    raise ValueError(backend)


def run_impl(kernels, impl: str, x, w, y, g_acc, eps, tile, warp_size, acc_is_half=False):
    if impl == "det":
        kernels.det(x, w, y, eps, tile, warp_size)
    elif impl == "shared":
        kernels.atomic_shared(x, w, y, eps, tile, warp_size, acc_is_half=acc_is_half)
    elif impl == "global":
        kernels.atomic_global(x, w, g_acc, y, eps, tile, warp_size,
                              splits=GLOBAL_SPLITS, acc_is_half=acc_is_half)
    else:
        raise ValueError(impl)


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.contiguous(), b.contiguous()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "musa"])
    parser.add_argument("--tile", type=int, default=MAIN_TILE)
    parser.add_argument("--repeats-per-combo", type=int, default=3)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    backend_sync(device)

    fixtures = np.load(HERE / "fixtures" / "correctness.npz")
    manifest = json.loads((HERE / "fixtures" / "manifest.json").read_text(encoding="utf-8"))
    eps = manifest["eps"]

    results: dict = {
        "schema_version": "kernel-chapter08.correctness.v1",
        "backend": arguments.backend,
        "tile": arguments.tile,
        "eps": eps,
        "truth": "fixtures y（CPU FP32 串行，自然顺序累加）",
        "combos": [],
        "order_pairs": [],
        "edge": {},
        "three_run_sentinel": {},
        "environment": device_environment_record(device),
    }

    kernels = None
    warp_size = None
    if arguments.backend != "cpu":
        kernels = load_kernels_for(arguments.backend)
        warp_size = kernels.probe_warp_size(device)
        results["warp_size_probe"] = warp_size
        print(f"[probe] warpSize(device builtin) = {warp_size}")

    half_atomic_ok = True
    if arguments.backend != "cpu":
        status_file = (HERE / "implementations" / arguments.backend /
                       "build" / "half_atomic_status.txt")
        if status_file.exists() and "disabled" in status_file.read_text():
            half_atomic_ok = False
            results["half_atomic_platform_boundary"] = (
                "ACC=half 编译开关关闭（half atomicAdd 缺失，见 build 日志）")

    impl_list = ["det", "shared", "global"]
    acc_list = {"det": ["fp32"], "shared": ["fp32"] + (["half"] if half_atomic_ok else []),
                "global": ["fp32"] + (["half"] if half_atomic_ok else [])}

    combo_index = 0
    for combo in manifest["combos"]:
        rows, h = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{h}"
        x_np = fixtures[f"x_{key}"]
        w_np = fixtures[f"w_{h}"]
        y_np = fixtures[f"y_{key}"]
        x = torch.from_numpy(x_np).to(device)
        w = torch.from_numpy(w_np).to(device)
        entry = {"rows": rows, "hidden_size": h,
                 "truth_sha256": combo["sha256"]["y"], "impls": {}}

        for impl in impl_list:
            for acc in acc_list[impl]:
                if arguments.backend == "cpu":
                    # CPU 列：串行实现（真值口径，零误差）+ NumPy/torch 可读
                    # 实现（归约顺序不同 → 有漂移；3 次运行各自位级确定）。
                    truth_t = torch.from_numpy(y_np)
                    y_serial = torch.from_numpy(naive_rmsnorm_serial(x_np, w_np, eps))
                    metrics_s = comparison_metrics(y_serial, truth_t.float())
                    metrics_s["runs_bit_identical"] = arguments.repeats_per_combo
                    entry["impls"]["cpu_serial_fp32"] = metrics_s
                    if "cpu_numpy_rowwise_fp32" not in entry["impls"]:
                        from serial_rmsnorm import naive_rmsnorm_rowwise  # noqa: E402
                        y_np_impl = torch.from_numpy(naive_rmsnorm_rowwise(x_np, w_np, eps))
                        metrics_n = comparison_metrics(y_np_impl, truth_t.float())
                        metrics_n["runs_bit_identical"] = arguments.repeats_per_combo
                        entry["impls"]["cpu_numpy_rowwise_fp32"] = metrics_n
                        print(f"[{key}] cpu_numpy_rowwise: max={metrics_n['max_abs_error']:.3e}")
                        if torch is not None:
                            try:
                                from serial_rmsnorm import composite_rmsnorm_torch  # noqa: E402
                                y_torch = composite_rmsnorm_torch(torch.from_numpy(x_np),
                                                                  torch.from_numpy(w_np), eps)
                                metrics_t = comparison_metrics(y_torch.float(), truth_t.float())
                                metrics_t["runs_bit_identical"] = arguments.repeats_per_combo
                                entry["impls"]["cpu_torch_composite_fp32"] = metrics_t
                                print(f"[{key}] cpu_torch_composite: max={metrics_t['max_abs_error']:.3e}")
                            except RuntimeError as error:
                                entry["impls"]["cpu_torch_composite_fp32"] = (
                                    f"unavailable: {error}")
                    continue

                y = torch.empty_like(x)
                g_acc = torch.zeros(rows, dtype=torch.float16 if acc == "half" else torch.float32,
                                    device=device)
                run_impl(kernels, impl, x, w, y, g_acc, eps, arguments.tile, warp_size,
                         acc_is_half=(acc == "half"))
                backend_sync(device)
                metrics = comparison_metrics(y, torch.from_numpy(y_np))
                # 3 次位级一致哨兵
                first = y.clone()
                same = 0
                for _ in range(arguments.repeats_per_combo - 1):
                    run_impl(kernels, impl, x, w, y, g_acc, eps, arguments.tile, warp_size,
                             acc_is_half=(acc == "half"))
                    backend_sync(device)
                    same += int(bit_equal(y, first))
                metrics["runs_bit_identical"] = same + 1
                tag = f"{impl}_{acc}"
                entry["impls"][tag] = metrics
                print(f"[{key}] {tag}: max={metrics['max_abs_error']:.3e} "
                      f"mean={metrics['mean_abs_error']:.3e} cos={metrics['cosine_similarity']:.8f} "
                      f"3run_bitsame={metrics['runs_bit_identical']}/3")

                # det vs atomic 同输入位级差（fp32 acc、fp32 输入档才有意义）
                if impl != "det" and acc == "fp32":
                    y_det = torch.empty_like(x)
                    run_impl(kernels, "det", x, w, y_det, g_acc, eps, arguments.tile, warp_size)
                    backend_sync(device)
                    diff_bits = int((y_det.view(torch.int32) != y.view(torch.int32)).sum().item()) \
                        if x.dtype == torch.float32 else None
                    max_diff = float((y_det.float() - y.float()).abs().max().item())
                    results["order_pairs"].append({
                        "rows": rows, "hidden_size": h, "dtype": "fp16" if x.dtype == torch.float16 else "fp32",
                        "impl": impl,
                        "bit_diff_elements": diff_bits,
                        "max_abs_diff": max_diff,
                    })
        results["combos"].append(entry)
        combo_index += 1

    # ---- 边界行为（kernel 侧错误处理 + 空 Tensor）----
    if arguments.backend != "cpu":
        edge = {}
        try:
            x_bad = torch.randn(4, 128, device=device)
            w_bad = torch.randn(128, device=device)
            y_bad = torch.empty_like(x_bad)
            # tile=48 不是任何平台 warpSize（32）的整数倍 → 必须显式拒绝
            kernels.det(x_bad, w_bad, y_bad, eps, 48, warp_size)
            edge["tile_not_multiple"] = "no_error(UNEXPECTED: 48 应被拒绝)"
        except RuntimeError as error:
            edge["tile_not_multiple"] = str(error)[:200]
        try:
            x_empty = torch.empty(0, 1024, device=device)
            w_empty = torch.randn(1024, device=device)
            y_empty = torch.empty_like(x_empty)
            kernels.det(x_empty, w_empty, y_empty, eps, arguments.tile, warp_size)
            edge["rows_zero"] = "no_error(rows=0 不发射 kernel)"
        except RuntimeError as error:
            edge["rows_zero"] = str(error)[:200]
        results["edge"] = edge
        print(f"[edge] {json.dumps(edge, ensure_ascii=False)[:300]}")

    out_path = Path(arguments.out) if arguments.out else (
        HERE / f"results/correctness-results-{arguments.backend}.json")
    write_results_json(out_path, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
