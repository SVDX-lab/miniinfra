#!/usr/bin/env python3
"""第 07 期统一 benchmark：性能对照 + 100 次确定性 + 漂移曲线 + 竞争强度扫描。

四块实验（与大纲 §13 第 07 期"实验"清单一一对应）：
1. --perf：合同 Shape rows={1,4,128,512,2048}×H=1024 × dtype{fp32,fp16} ×
   三实现（det/shared/global-splits4）× acc{fp32,half(atomic)}；
   标注 L2 区/DRAM 区（workset vs 4090 L2=72MB / S4000 L2=24MB）与
   launch 数（det/shared=1，global=3：zero+sqsum+scale）。
2. --determinism：每实现 100 次重复，统计位级一致率（torch.equal vs
   首次）、唯一输出数（SHA256）、漂移带（每次运行 vs CPU FP32 串行真值的
   max abs error 的 min/max）。
3. --curve：H ∈ {1024,4096,16384,65536} × rows ∈ {128,2048} 的确定性 +
   性能曲线；ACC=half 档接第 05 期卡死区（atomic 组织下增量粒度变化 →
   卡死/幸存行为是本期天然实验点）。
4. --splits-scan：global atomic splits ∈ {1,2,4,8,16} + shared atomic
   tile ∈ {128,256,512}——竞争强度 vs 漂移/性能。

计时口径：experiment_utils.OperatorTimer（Event 对 + 末尾一次同步，
P50/P95/mean/min），与第 02 期完全一致；MUSA 侧同口径（M-12：~28µs
launch 地板，小形状读数为端到端含发射口径）。

用法：
  python benchmark.py --backend cpu  --perf --determinism
  python benchmark.py --backend cuda --perf --determinism --curve --splits-scan
  python benchmark.py --backend musa --perf --determinism --curve --splits-scan
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

import torch  # noqa: E402

from experiment_utils import (  # noqa: E402
    OperatorTimer,
    backend_sync,
    device_environment_record,
    resolve_device,
)
from serial_rmsnorm import RMS_NORM_EPS, naive_rmsnorm_serial  # noqa: E402

SEED_CURVE = 20260809
CURVE_H = (1024, 4096, 16384, 65536)
CURVE_ROWS = (128, 2048)
CONTRACT_ROWS = (1, 4, 128, 512, 2048)
CONTRACT_H = 1024
MAIN_TILE = 256
GLOBAL_SPLITS_DEFAULT = 4
SPLITS_SCAN = (1, 2, 4, 8, 16)
ATOMIC_TILES = (128, 256, 512)
# Python 串行真值的元素数上限（超出部分真值漂移由 det_atomic_bench C 速度通道承担）
PYTHON_TRUTH_CAP = 10_000_000
# 每次运行回传 SHA256 的元素数上限（hash 成本约束）
HASH_CAP = 4_000_000
L2_BYTES = {"cuda": 72 * 1024 * 1024, "musa": 24 * 1024 * 1024, "cpu": None}


def region_label(workset_bytes: int, backend: str) -> str:
    l2 = L2_BYTES.get(backend)
    if l2 is None:
        return "cpu"
    return "L2" if workset_bytes <= l2 else "DRAM"


def make_curve_inputs(rows: int, h: int, dtype: str, device: torch.device):
    rng = np.random.default_rng(SEED_CURVE + h * 10 + rows)
    x_np = rng.standard_normal((rows, h), dtype=np.float32)
    w_np = rng.standard_normal(h, dtype=np.float32)
    if dtype == "fp16":
        x_np = x_np.astype(np.float16).astype(np.float32)  # 设备看到的量化输入
        w_np = w_np.astype(np.float16).astype(np.float32)
    x = torch.from_numpy(x_np.astype(np.float16 if dtype == "fp16" else np.float32)).to(device)
    w = torch.from_numpy(w_np.astype(np.float16 if dtype == "fp16" else np.float32)).to(device)
    return x, w, x_np, w_np


class KernelRunner:
    """三实现的统一调用闭包（ctypes；cpu 后端不用）。"""

    def __init__(self, kernels, warp_size: int):
        self.kernels = kernels
        self.warp_size = warp_size

    def __call__(self, impl: str, x, w, y, g_acc, acc_is_half: bool, splits: int, tile: int):
        if impl == "det":
            self.kernels.det(x, w, y, RMS_NORM_EPS, tile, self.warp_size)
        elif impl == "shared":
            self.kernels.atomic_shared(x, w, y, RMS_NORM_EPS, tile, self.warp_size,
                                       acc_is_half=acc_is_half)
        elif impl == "global":
            self.kernels.atomic_global(x, w, g_acc, y, RMS_NORM_EPS, tile, self.warp_size,
                                       splits=splits, acc_is_half=acc_is_half)
        else:
            raise ValueError(impl)


def sha256_of_torch(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def determinism_run(runner, impl: str, acc: str, splits: int, tile: int,
                    x, w, truth_np, device, repeats: int, y_ref_hash: str | None):
    """100 次重复的位级一致率 + 唯一输出数 + 漂移带。"""
    n_elems = x.numel()
    acc_is_half = acc == "half"
    g_acc = torch.zeros(x.shape[0], dtype=torch.float16 if acc_is_half else torch.float32,
                        device=device)
    y = torch.empty_like(x)
    runner(impl, x, w, y, g_acc, acc_is_half, splits, tile)
    backend_sync(device)
    y0 = y.clone()
    identical = 1
    hashes = {sha256_of_torch(y0)} if n_elems <= HASH_CAP else None
    truth = torch.from_numpy(truth_np) if truth_np is not None else None
    y_cpu = y.float().cpu()
    drift_max_first = (float((y_cpu - truth).abs().max().item())
                       if truth is not None else None)
    band = [drift_max_first]
    for _ in range(repeats - 1):
        runner(impl, x, w, y, g_acc, acc_is_half, splits, tile)
        backend_sync(device)
        if bool(torch.equal(y, y0)):
            identical += 1
        if hashes is not None:
            hashes.add(sha256_of_torch(y))
        if truth is not None and n_elems <= 64_000_000:
            y_cpu = y.float().cpu()
            band.append(float((y_cpu - truth).abs().max().item()))
    band = [b for b in band if b is not None]
    return {
        "repeats": repeats,
        "identical_vs_first": identical,
        "unique_outputs": len(hashes) if hashes is not None else None,
        "hash_scope": "per_run_sha256" if hashes is not None else "skip_large",
        "drift_band_min": min(band) if band else None,
        "drift_band_max": max(band) if band else None,
        "drift_vs_truth_run0": drift_max_first,
    }


def perf_run(runner, impl: str, acc: str, splits: int, tile: int,
             x, w, device, warmup: int, repeats: int) -> dict:
    acc_is_half = acc == "half"
    g_acc = torch.zeros(x.shape[0], dtype=torch.float16 if acc_is_half else torch.float32,
                        device=device)
    y = torch.empty_like(x)
    timer = OperatorTimer(device, warmup, repeats)
    return timer.measure(lambda: runner(impl, x, w, y, g_acc, acc_is_half, splits, tile))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "musa"])
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--determinism", action="store_true")
    parser.add_argument("--curve", action="store_true")
    parser.add_argument("--splits-scan", action="store_true")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--tile", type=int, default=MAIN_TILE)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    backend_sync(device)
    flags = (arguments.perf, arguments.determinism, arguments.curve, arguments.splits_scan)
    if not any(flags):
        arguments.perf = arguments.determinism = True

    results: dict = {
        "schema_version": "kernel-chapter08.benchmark.v1",
        "backend": arguments.backend,
        "tile": arguments.tile,
        "warmup": arguments.warmup,
        "repeats": arguments.repeats,
        "eps": RMS_NORM_EPS,
        "environment": device_environment_record(device),
        "perf": [],
        "determinism": [],
        "curve": [],
        "splits_scan": [],
        "notes": [],
    }

    runner = None
    warp_size = None
    if arguments.backend != "cpu":
        if arguments.backend == "cuda":
            sys.path.insert(0, str(HERE / "implementations" / "cuda"))
            from ops_binding import load_kernels  # noqa: E402
            kernels = load_kernels("cuda", "libmini_detatomic_cuda.so")
        else:
            sys.path.insert(0, str(HERE / "implementations" / "musa"))
            from ops_binding import load_kernels  # noqa: E402
            kernels = load_kernels("musa", "libmini_detatomic_musa.so")
        warp_size = kernels.probe_warp_size(device)
        runner = KernelRunner(kernels, warp_size)
        results["warp_size_probe"] = warp_size
        status_file = (HERE / "implementations" / arguments.backend /
                       "build" / "half_atomic_status.txt")
        if status_file.exists() and "disabled" in status_file.read_text():
            results["half_atomic_platform_boundary"] = "ACC=half 编译开关关闭（平台边界）"
            print("[half-atomic] 平台边界：ACC=half 不可用，跳过该档")

    def acc_list_for(impl: str) -> list:
        if arguments.backend == "cpu" or impl == "det":
            return ["fp32"]
        if results.get("half_atomic_platform_boundary"):
            return ["fp32"]
        return ["fp32", "half"]

    # ================= 1. perf：合同 Shape =================
    if arguments.perf and arguments.backend != "cpu":
        h = CONTRACT_H
        for dtype in ("fp32", "fp16"):
            x, w, _, _ = make_curve_inputs(CONTRACT_ROWS[-1], h, dtype, device)
            for rows in CONTRACT_ROWS:
                x_r = x[:rows].contiguous()
                workset = rows * h * (2 if dtype == "fp16" else 4) * 2
                region = region_label(workset, arguments.backend)
                for impl in ("det", "shared", "global"):
                    for acc in acc_list_for(impl):
                        timing = perf_run(runner, impl, acc, GLOBAL_SPLITS_DEFAULT,
                                          arguments.tile, x_r, w, device,
                                          arguments.warmup, arguments.repeats)
                        entry = {
                            "shape": f"{rows}x{h}", "dtype": dtype, "impl": impl,
                            "acc": acc, "splits": GLOBAL_SPLITS_DEFAULT,
                            "launches": 3 if impl == "global" else 1,
                            "region": region,
                            "workset_bytes": workset, **timing,
                        }
                        results["perf"].append(entry)
                        print(f"[perf] {rows}x{h} {dtype} {impl}/{acc} [{region}] "
                              f"p50={timing['p50_ms']}ms p95={timing['p95_ms']}ms "
                              f"launches={entry['launches']}")

    # ================= 2. determinism：合同主形状 100 次 =================
    if arguments.determinism and arguments.backend != "cpu":
        h = CONTRACT_H
        rows = 2048
        for dtype in ("fp32", "fp16"):
            x, w, x_np_q, w_np_q = make_curve_inputs(rows, h, dtype, device)
            truth = naive_rmsnorm_serial(x_np_q, w_np_q, RMS_NORM_EPS)
            for impl in ("det", "shared", "global"):
                for acc in acc_list_for(impl):
                    stats = determinism_run(runner, impl, acc, GLOBAL_SPLITS_DEFAULT,
                                            arguments.tile, x, w, truth, device,
                                            arguments.repeats, None)
                    entry = {"shape": f"{rows}x{h}", "dtype": dtype, "impl": impl,
                             "acc": acc, "splits": GLOBAL_SPLITS_DEFAULT, **stats}
                    results["determinism"].append(entry)
                    print(f"[det100] {dtype} {impl}/{acc}: identical="
                          f"{stats['identical_vs_first']}/{stats['repeats']} "
                          f"unique={stats['unique_outputs']} "
                          f"band=[{stats['drift_band_min']:.2e},{stats['drift_band_max']:.2e}]")

    # ================= 3. curve：H × rows 增长曲线 =================
    if arguments.curve and arguments.backend != "cpu":
        for h in CURVE_H:
            for rows in CURVE_ROWS:
                for dtype in ("fp16",):   # 曲线主口径 fp16 输入（第 05 期主线延续）
                    x, w, x_np_q, w_np_q = make_curve_inputs(rows, h, dtype, device)
                    can_truth = rows * h <= PYTHON_TRUTH_CAP
                    truth = naive_rmsnorm_serial(x_np_q, w_np_q, RMS_NORM_EPS) if can_truth else None
                    workset = rows * h * 2 * 2
                    region = region_label(workset, arguments.backend)
                    for impl in ("det", "shared", "global"):
                        for acc in acc_list_for(impl):
                            stats = determinism_run(runner, impl, acc, GLOBAL_SPLITS_DEFAULT,
                                                    arguments.tile, x, w, truth, device,
                                                    arguments.repeats, None)
                            timing = perf_run(runner, impl, acc, GLOBAL_SPLITS_DEFAULT,
                                              arguments.tile, x, w, device,
                                              arguments.warmup, min(arguments.repeats, 50))
                            entry = {"shape": f"{rows}x{h}", "dtype": dtype, "impl": impl,
                                     "acc": acc, "splits": GLOBAL_SPLITS_DEFAULT,
                                     "region": region, "workset_bytes": workset,
                                     "truth_scope": "python_serial" if can_truth
                                     else "bench_channel(see det_atomic_bench JSON)", **stats,
                                     "perf": timing}
                            results["curve"].append(entry)
                            print(f"[curve] {rows}x{h} {dtype} {impl}/{acc} [{region}]: "
                                  f"identical={stats['identical_vs_first']}/{stats['repeats']} "
                                  f"unique={stats['unique_outputs']} "
                                  f"band=[{stats['drift_band_min'] if stats['drift_band_min'] is not None else 'NA'},"
                                  f"{stats['drift_band_max']}] p50={timing['p50_ms']}ms")

    # ================= 4. splits/tile 扫描：竞争强度 =================
    if arguments.splits_scan:
        h = CONTRACT_H
        rows = 2048
        for dtype in ("fp16",):
            x, w, x_np_q, w_np_q = make_curve_inputs(rows, h, dtype, device)
            truth = naive_rmsnorm_serial(x_np_q, w_np_q, RMS_NORM_EPS)
            # global atomic splits 扫描
            for splits in SPLITS_SCAN:
                for acc in acc_list_for("global"):
                    stats = determinism_run(runner, "global", acc, splits, arguments.tile,
                                            x, w, truth, device, arguments.repeats, None)
                    timing = perf_run(runner, "global", acc, splits, arguments.tile,
                                      x, w, device, arguments.warmup, arguments.repeats)
                    entry = {"impl": "global", "splits": splits, "acc": acc, "dtype": dtype,
                             "competitors_per_row": splits * arguments.tile, **stats,
                             "perf": timing}
                    results["splits_scan"].append(entry)
                    print(f"[splits] global s={splits} {acc}: "
                          f"identical={stats['identical_vs_first']}/{stats['repeats']} "
                          f"unique={stats['unique_outputs']} "
                          f"band=[{stats['drift_band_min']:.2e},{stats['drift_band_max']:.2e}] "
                          f"p50={timing['p50_ms']}ms")
            # shared atomic tile 扫描（竞争强度 = tile 线程）
            for tile in ATOMIC_TILES:
                for acc in acc_list_for("shared"):
                    stats = determinism_run(runner, "shared", acc, GLOBAL_SPLITS_DEFAULT,
                                            tile, x, w, truth, device, arguments.repeats, None)
                    timing = perf_run(runner, "shared", acc, GLOBAL_SPLITS_DEFAULT, tile,
                                      x, w, device, arguments.warmup, arguments.repeats)
                    entry = {"impl": "shared", "tile": tile, "acc": acc, "dtype": dtype,
                             "competitors_per_row": tile, **stats, "perf": timing}
                    results["splits_scan"].append(entry)
                    print(f"[tile] shared t={tile} {acc}: "
                          f"identical={stats['identical_vs_first']}/{stats['repeats']} "
                          f"unique={stats['unique_outputs']} "
                          f"band=[{stats['drift_band_min']:.2e},{stats['drift_band_max']:.2e}] "
                          f"p50={timing['p50_ms']}ms")

    # ================= CPU 后端：串行实现参照 =================
    if arguments.backend == "cpu":
        h = CONTRACT_H
        for dtype in ("fp32",):
            for rows in CONTRACT_ROWS:
                _, _, x_np_q, w_np_q = make_curve_inputs(rows, h, dtype, device)
                timer = OperatorTimer(device, 3, arguments.repeats)
                timing = timer.measure(lambda: naive_rmsnorm_serial(x_np_q, w_np_q, RMS_NORM_EPS))
                truth = naive_rmsnorm_serial(x_np_q, w_np_q, RMS_NORM_EPS)
                first = truth.copy()
                identical = 0
                hashes = set()
                for _ in range(arguments.repeats):
                    out = naive_rmsnorm_serial(x_np_q, w_np_q, RMS_NORM_EPS)
                    identical += int((out.tobytes() == first.tobytes()))
                    hashes.add(hashlib.sha256(out.tobytes()).hexdigest())
                results["perf"].append({"shape": f"{rows}x{h}", "dtype": dtype,
                                        "impl": "cpu_serial", "acc": "fp32",
                                        "launches": 0, "region": "cpu", **timing})
                results["determinism"].append({
                    "shape": f"{rows}x{h}", "dtype": dtype, "impl": "cpu_serial",
                    "acc": "fp32", "repeats": arguments.repeats,
                    "identical_vs_first": identical, "unique_outputs": len(hashes),
                    "hash_scope": "per_run_sha256",
                    "drift_band_min": 0.0, "drift_band_max": 0.0, "drift_vs_truth_run0": 0.0,
                    "note": "串行实现同一顺序重复运行：位级确定是构造保证（顺序固定），"
                            "100 次一致率 100/100 由实测确认",
                })
                print(f"[cpu] {rows}x{h} serial p50={timing['p50_ms']}ms "
                      f"identical={identical}/{arguments.repeats}")

    out_path = Path(arguments.out) if arguments.out else (
        HERE / f"results/benchmark-results-{arguments.backend}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    print(f"[done] 已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
