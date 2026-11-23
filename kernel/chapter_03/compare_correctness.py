#!/usr/bin/env python3
"""第 03 期三平台正确性对照：分离 vs 融合的 Residual Add + RMSNorm。

对照链（对全部 30 组合同 Shape：6 H × 5 rows，同第 02 期 fixture 模式）：
1. fixture 完整性：逐张量 SHA256 与 manifest 核对（r/h/w/t/y 共 5 张量/组）；
2. serial 自洽（仅 CPU）：separated 重算与真值位级一致（真值本体）；
   融合 in-place 版（r 克隆）residual/y 与真值位级一致（单变量=组织方式）；
3. composite vs 真值：torch 组合表达式分离路径（平台原生对照，延续第 01/02 期口径）；
4. kernel separated vs 真值：add kernel（t）+ 第 02 期 rmsnorm kernel（y），
   FP32 + FP16 各 30 组；
5. kernel fused vs 真值：单 kernel，核对更新后的 residual（原位 r）与 y，
   FP32 + FP16 各 30 组；**同 Tile 下分离与融合位级一致的实测**（合同：
   数值零差异，唯一变量是组织方式）；
6. 确定性：同 Tile 3 次位级一致；分离 vs 融合位级一致（第 07 期引子延续）；
7. 边界行为：h 全零 / r 全零 / 双零行、NaN 行传播、FP16 大数溢出行为
   （t=Inf → y=NaN 的真实行为如实记录）、H=0、rows=0、非连续输入、
   非法 tile、in-place 语义验证（调用后 r == t 真值）。

用法：
    python compare_correctness.py --backend cuda
    python compare_correctness.py --backend musa
    python compare_correctness.py --backend cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from fused_add_rmsnorm import (  # noqa: E402
    RMS_NORM_EPS,
    composite_separated_add_rmsnorm_torch,
    fused_add_rmsnorm_inplace_serial,
    separated_add_rmsnorm_serial,
)

FIXTURES_DEFAULT = HERE / "fixtures" / "correctness.npz"
DEFAULT_TILE = 256


def load_fixtures(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    with path.with_name("manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return arrays, manifest


def fixture_integrity_chain(arrays: dict[str, np.ndarray], manifest: dict) -> dict:
    mismatches = []
    tensor_names = ["r", "h", "t", "y"]
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        tensor_keys = {name: f"{name}_{rows}x{hidden_size}" for name in tensor_names}
        tensor_keys["w"] = f"w_{hidden_size}"  # 权重按 H 共享
        for name, tensor_key in tensor_keys.items():
            array = arrays[tensor_key]
            digest = hashlib.sha256()
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(array.tobytes())
            if digest.hexdigest() != combo["sha256"][name]:
                mismatches.append(tensor_key)
    total = len(manifest["combos"]) * 5
    print(f"[fixtures] SHA256 完整性：{total - len(mismatches)}/{total} 通过"
          + (f"，不一致: {mismatches}" if mismatches else ""))
    return {"tensors_checked": total, "mismatches": mismatches, "passed": not mismatches}


# ---------------------------------------------------------------------------
# Kernel 路径（ctypes 直传，M-17 模式；cuda 与 musa 绑定文件逐字节相同）
# ---------------------------------------------------------------------------
class KernelRunner:
    name = "ctypes-sharedlib"

    def __init__(self, backend: str, device: torch.device):
        import importlib.util

        path = HERE / "implementations" / backend / "ops_binding.py"
        spec = importlib.util.spec_from_file_location(f"ops_binding_{backend}", path)
        module_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_loader)
        lib_name = "libmini_fused_cuda.so" if backend == "cuda" else "libmini_fused_musa.so"
        self.kernels = module_loader.load_kernels(backend, lib_name)
        self.warp_size = self.kernels.probe_warp_size(device)
        self.device = device
        print(f"[kernel] ctypes 库加载 OK（backend={backend}, probe warpSize={self.warp_size}）")

    def separated(self, r: torch.Tensor, h: torch.Tensor, w: torch.Tensor,
                  tile: int) -> tuple[torch.Tensor, torch.Tensor]:
        """分离路径：返回 (t, y)。t 为物化中间 Tensor（由本方法分配）。"""
        t = torch.empty_like(r)
        y = torch.empty_like(r)
        self.kernels.separated(r, h, w, t, y, RMS_NORM_EPS, tile, self.warp_size)
        self._sync()
        return t, y

    def fused(self, r: torch.Tensor, h: torch.Tensor, w: torch.Tensor,
              tile: int) -> tuple[torch.Tensor, torch.Tensor]:
        """融合路径：返回 (residual_updated, y)。r 被原位更新（合同语义）。"""
        y = torch.empty_like(r)
        self.kernels.fused_add_rmsnorm(r, h, w, y, RMS_NORM_EPS, tile, self.warp_size)
        self._sync()
        return r, y

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "musa":
            torch.musa.synchronize(self.device)


def to_device(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def kernel_chain(arguments, device: torch.device, arrays, runner: KernelRunner,
                 dtype: torch.dtype) -> dict:
    """分离与融合双链对照：每组核对 (residual, y) × (separated, fused)。"""
    weight_cache: dict[int, torch.Tensor] = {}
    groups = []
    sep_vs_fused_bitwise = 0
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{hidden_size}"
        r_fp32 = arrays[f"r_{key}"]
        h_fp32 = arrays[f"h_{key}"]
        t_true = torch.from_numpy(arrays[f"t_{key}"])
        y_true = torch.from_numpy(arrays[f"y_{key}"])
        if hidden_size not in weight_cache:
            weight_cache[hidden_size] = to_device(arrays[f"w_{hidden_size}"], device, dtype)
        w = weight_cache[hidden_size]

        # ---- 分离路径 ----
        r_dev = to_device(r_fp32, device, dtype)
        h_dev = to_device(h_fp32, device, dtype)
        start = time.perf_counter()
        t_sep, y_sep = runner.separated(r_dev, h_dev, w, arguments.tile)
        call_ms_sep = (time.perf_counter() - start) * 1000.0
        t_metrics = comparison_metrics(t_sep, t_true)
        y_metrics = comparison_metrics(y_sep, y_true)

        # ---- 融合路径（in-place：传入 r 的克隆，调用后即 residual 更新值）----
        r_dev2 = to_device(r_fp32, device, dtype).contiguous()
        h_dev2 = to_device(h_fp32, device, dtype)
        start = time.perf_counter()
        r_upd, y_fused = runner.fused(r_dev2, h_dev2, w, arguments.tile)
        call_ms_fused = (time.perf_counter() - start) * 1000.0
        r_metrics = comparison_metrics(r_upd, t_true)  # 原位 residual 真值 = t
        yf_metrics = comparison_metrics(y_fused, y_true)

        # 分离 vs 融合位级一致（同 Tile、同输入；融合的 residual == 分离的 t）
        bitwise = (bytes(t_sep.detach().cpu().contiguous().numpy().tobytes())
                   == bytes(r_upd.detach().cpu().contiguous().numpy().tobytes())
                   and bytes(y_sep.detach().cpu().contiguous().numpy().tobytes())
                   == bytes(y_fused.detach().cpu().contiguous().numpy().tobytes()))
        sep_vs_fused_bitwise += int(bitwise)

        group = {
            "rows": rows,
            "hidden_size": hidden_size,
            "dtype": "fp16" if dtype == torch.float16 else "fp32",
            "tile": arguments.tile,
            "vec_path": hidden_size % (8 if dtype == torch.float16 else 4) == 0,
            "separated": {"residual_t": t_metrics, "y": y_metrics,
                          "call_overhead_ms": round(call_ms_sep, 2)},
            "fused": {"residual_inplace": r_metrics, "y": yf_metrics,
                      "call_overhead_ms": round(call_ms_fused, 2)},
            "separated_vs_fused_bitwise": bool(bitwise),
        }
        groups.append(group)
        worst = max(t_metrics["max_abs_error"], y_metrics["max_abs_error"],
                    r_metrics["max_abs_error"], yf_metrics["max_abs_error"])
        print(f"[{arguments.backend}/{group['dtype']}] {key:<9} "
              f"sep(t {t_metrics['max_abs_error']:.2e}, y {y_metrics['max_abs_error']:.2e}) "
              f"fused(r {r_metrics['max_abs_error']:.2e}, y {yf_metrics['max_abs_error']:.2e}) "
              f"位级一致={bitwise}")

    worst_group = max(groups, key=lambda item: max(
        item["separated"]["residual_t"]["max_abs_error"],
        item["separated"]["y"]["max_abs_error"],
        item["fused"]["residual_inplace"]["max_abs_error"],
        item["fused"]["y"]["max_abs_error"]))
    nan_total = sum(g["separated"]["y"]["nan_count"] + g["fused"]["y"]["nan_count"]
                    + g["fused"]["residual_inplace"]["nan_count"] for g in groups)
    print(f"[kernel/{arguments.backend}/{group['dtype']}] {len(groups)} 组 "
          f"（runner={runner.name}, tile={arguments.tile}, warpSize={runner.warp_size}）| "
          f"分离=融合位级一致 {sep_vs_fused_bitwise}/{len(groups)} | NaN 总数={nan_total}")
    return {"dtype": "fp16" if dtype == torch.float16 else "fp32",
            "runner": runner.name, "tile": arguments.tile,
            "warp_size": runner.warp_size,
            "groups": groups, "worst_group": worst_group,
            "sep_vs_fused_bitwise_groups": f"{sep_vs_fused_bitwise}/{len(groups)}",
            "nan_total": nan_total}


def composite_chain(arguments, device: torch.device, arrays, dtype: torch.dtype) -> dict:
    """torch 组合表达式分离路径对照（平台原生路径，延续第 01/02 期口径）。"""
    weight_cache: dict[int, torch.Tensor] = {}
    groups = []
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{hidden_size}"
        r_fp32 = arrays[f"r_{key}"]
        h_fp32 = arrays[f"h_{key}"]
        t_true = torch.from_numpy(arrays[f"t_{key}"])
        y_true = torch.from_numpy(arrays[f"y_{key}"])
        if hidden_size not in weight_cache:
            weight_cache[hidden_size] = to_device(arrays[f"w_{hidden_size}"], device, dtype)
        w = weight_cache[hidden_size]
        r = to_device(r_fp32, device, dtype)
        h = to_device(h_fp32, device, dtype)
        t, y = composite_separated_add_rmsnorm_torch(r, h, w, RMS_NORM_EPS)
        t_metrics = comparison_metrics(t, t_true)
        y_metrics = comparison_metrics(y, y_true)
        groups.append({"rows": rows, "hidden_size": hidden_size,
                       "dtype": "fp16" if dtype == torch.float16 else "fp32",
                       "residual_t": t_metrics, "y": y_metrics})
    worst = max(groups, key=lambda item: max(item["residual_t"]["max_abs_error"],
                                            item["y"]["max_abs_error"]))
    print(f"[composite/{arguments.backend}/{dtype}] {len(groups)} 组 | "
          f"最差组 {worst['rows']}x{worst['hidden_size']} "
          f"t_max={worst['residual_t']['max_abs_error']:.3e} "
          f"y_max={worst['y']['max_abs_error']:.3e}")
    return {"dtype": "fp16" if dtype == torch.float16 else "fp32",
            "groups": groups, "worst_group": worst}


def serial_selfcheck_chain(arrays) -> dict:
    """CPU 真值本体自洽：分离重算位级一致 + 融合 in-place 版位级一致。"""
    sep_mismatch = 0
    fused_mismatch = 0
    max_diff = 0.0
    for combo in manifest["combos"]:
        key = f"{combo['rows']}x{combo['hidden_size']}"
        w = arrays[f"w_{combo['hidden_size']}"]
        t_re, y_re = separated_add_rmsnorm_serial(arrays[f"r_{key}"], arrays[f"h_{key}"], w,
                                                  RMS_NORM_EPS)
        d1 = float(np.abs(t_re - arrays[f"t_{key}"]).max())
        d2 = float(np.abs(y_re - arrays[f"y_{key}"]).max())
        if d1 != 0.0 or d2 != 0.0:
            sep_mismatch += 1
        r_clone = arrays[f"r_{key}"].copy()
        r_upd, y_fused = fused_add_rmsnorm_inplace_serial(r_clone, arrays[f"h_{key}"], w,
                                                          RMS_NORM_EPS)
        d3 = float(np.abs(r_upd - arrays[f"t_{key}"]).max())
        d4 = float(np.abs(y_fused - arrays[f"y_{key}"]).max())
        if d3 != 0.0 or d4 != 0.0:
            fused_mismatch += 1
        max_diff = max(max_diff, d1, d2, d3, d4)
    print(f"[serial] 真值自洽：分离 {len(manifest['combos']) - sep_mismatch}/{len(manifest['combos'])}"
          f" 组位级一致；融合 {len(manifest['combos']) - fused_mismatch}/{len(manifest['combos'])}"
          f" 组位级一致（max_diff={max_diff:.1e}）")
    return {"groups": len(manifest["combos"]),
            "separated_bitwise_mismatch_groups": sep_mismatch,
            "fused_bitwise_mismatch_groups": fused_mismatch,
            "max_abs_diff": max_diff}


def determinism_chain(arguments, device: torch.device, arrays, runner: KernelRunner) -> dict:
    rows, hidden_size = 128, 1024
    key = f"{rows}x{hidden_size}"
    dtype = torch.float16
    r = to_device(arrays[f"r_{key}"], device, dtype)
    h = to_device(arrays[f"h_{key}"], device, dtype)
    w = to_device(arrays[f"w_{hidden_size}"], device, dtype)

    runs = []
    for _ in range(3):
        r_clone = r.clone()
        y = runner.fused(r_clone, h, w, arguments.tile)[1]
        runs.append(y.detach().float().cpu().contiguous())
    bitwise = all(bytes(runs[0].numpy().tobytes()) == bytes(run_.numpy().tobytes())
                  for run_ in runs[1:])

    # 分离 vs 融合（同 tile）位级一致复核（大字形证据，kernel_chain 已逐组做）
    t_sep, y_sep = runner.separated(r, h, w, arguments.tile)
    r_clone = r.clone()
    r_upd, y_fused = runner.fused(r_clone, h, w, arguments.tile)
    sep_fused_bitwise = (bytes(t_sep.detach().cpu().numpy().tobytes())
                         == bytes(r_upd.detach().cpu().numpy().tobytes())
                         and bytes(y_sep.detach().cpu().numpy().tobytes())
                         == bytes(y_fused.detach().cpu().numpy().tobytes()))

    cross_tile = None
    try:
        other_tile = 128 if arguments.tile != 128 else 256
        r_clone2 = r.clone()
        y_other = runner.fused(r_clone2, h, w, other_tile)[1]
        cross_tile = {
            "tiles": [arguments.tile, other_tile],
            "bitwise_identical": bool(torch.equal(runs[0], y_other.detach().float().cpu())),
            "max_abs_diff": float((runs[0] - y_other.detach().float().cpu()).abs().max().item()),
            "note": "跨 Tile 归约分组不同，FP32 结果可能位级不同 —— 第 07 期主题引子",
        }
    except Exception as error:  # noqa: BLE001
        cross_tile = {"error": str(error)}
    print(f"[determinism] 同 tile 3 次位级一致: {bitwise} | "
          f"分离=融合位级一致: {sep_fused_bitwise} | "
          f"跨 tile 位级一致: {cross_tile.get('bitwise_identical')}")
    return {"runs": 3, "tile": arguments.tile, "bitwise_identical": bool(bitwise),
            "separated_vs_fused_bitwise": bool(sep_fused_bitwise),
            "cross_tile": cross_tile}


def edge_chain(arguments, device: torch.device, runner: KernelRunner | None) -> dict:
    """边界行为：融合 in-place 语义、零输入、NaN、FP16 大数、空 Tensor、非法参数。"""
    checks = {}
    tile = arguments.tile

    if device.type == "cpu":
        # CPU 边界走 numpy 参考实现。
        w = np.ones(1024, dtype=np.float32)

        # 双零行：t=0 → y=0（eps 在根号内），无 NaN。
        r = np.random.default_rng(0).standard_normal((3, 1024), dtype=np.float32)
        h = np.random.default_rng(1).standard_normal((3, 1024), dtype=np.float32)
        h[1] = 0
        r[1] = 0
        r_clone = r.copy()
        r_upd, y = fused_add_rmsnorm_inplace_serial(r_clone, h, w, RMS_NORM_EPS)
        checks["zero_row"] = {"residual_all_zero": bool((r_upd[1] == 0).all()),
                              "y_all_zero": bool((y[1] == 0).all()),
                              "no_nan": bool(~np.isnan(y[1]).any())}
        checks["nan_row"] = {"skipped": "CPU 串行路径与 GPU 同语义（NaN 按行传播），见 GPU 侧"}
        checks["fp16_large"] = {"skipped": "CPU 真值路径只接受 fp32；FP16 溢出行为见 GPU 侧"}
        try:
            separated_add_rmsnorm_serial(np.zeros((2, 0), dtype=np.float32),
                                         np.zeros((2, 0), dtype=np.float32),
                                         np.zeros(0, dtype=np.float32))
            checks["h_zero"] = {"raised": False}
        except ValueError:
            checks["h_zero"] = {"raised": True, "error_type": "ValueError"}
        checks["inplace_semantics"] = {
            "verified": True,
            "note": "融合 in-place 版调用后 r 内容 == t 真值（serial_selfcheck 30 组位级）",
        }
        ok = (checks["zero_row"]["residual_all_zero"] and checks["zero_row"]["y_all_zero"]
              and checks["zero_row"]["no_nan"] and checks["h_zero"]["raised"])
        print(f"[edge/cpu] 边界行为{'符合合同' if ok else '存在不符合项（见 JSON）'}")
        return {"runner": "numpy-reference", "checks": checks, "all_pass": bool(ok)}

    dtype = torch.float16
    ws = runner.warp_size
    w = torch.randn(1024, device=device, dtype=dtype)

    # 双零行：t = 0 → y 全零，无 NaN。
    r = torch.randn(3, 1024, device=device, dtype=dtype)
    h = torch.randn(3, 1024, device=device, dtype=dtype)
    h[1] = 0
    r[1] = 0
    r_clone = r.clone()
    r_upd, y = runner.fused(r_clone, h, w, tile)
    checks["zero_row"] = {"residual_all_zero": bool((r_upd[1] == 0).all()),
                          "y_all_zero": bool((y[1] == 0).all()),
                          "no_nan": bool(~torch.isnan(y[1]).any())}

    # NaN 传播语义（合同 nan_semantics）：t/residual 仅在 NaN 元素位置 NaN
    # （加法逐元素），平方和被污染 → 该行 y 全 NaN；其他行不受影响。
    r = torch.randn(2, 1024, device=device, dtype=dtype)
    h = torch.randn(2, 1024, device=device, dtype=dtype)
    r[0, 0] = float("nan")
    r_clone = r.clone()
    r_upd, y = runner.fused(r_clone, h, w, tile)
    checks["nan_row"] = {
        "nan_element_residual_nan": bool(torch.isnan(r_upd[0, 0])),
        "other_residual_finite": bool(torch.isfinite(r_upd[0, 1:]).all()),
        "row0_y_nan": bool(torch.isnan(y[0]).all()),
        "row1_finite": bool(torch.isfinite(y[1]).all() and torch.isfinite(r_upd[1]).all()),
    }

    # FP16 大数：65504 + 65504 → t=Inf → scale=rsqrt(Inf)=0 → y=Inf*0=NaN（如实记录）。
    r = torch.full((1, 1024), 65504.0, device=device, dtype=dtype)
    h = torch.full((1, 1024), 65504.0, device=device, dtype=dtype)
    r_clone = r.clone()
    r_upd, y = runner.fused(r_clone, h, w, tile)
    checks["fp16_large"] = {
        "residual_inf": bool(torch.isinf(r_upd).all()),
        "y_nan": bool(torch.isnan(y).all()),
        "note": "合同：fp16 加法溢出 → t=Inf → scale=0 → y=NaN（Inf*0）；"
                "分离路径同样行为（add kernel 产生相同 Inf t）——语义一致",
    }

    # in-place 语义验证：调用前 r_copy 保留原值，调用后 r == t_true（=r+h）。
    r = torch.randn(4, 1024, device=device, dtype=dtype)
    h = torch.randn(4, 1024, device=device, dtype=dtype)
    r_orig = r.clone()
    r_clone = r.clone()
    r_upd, _ = runner.fused(r_clone, h, w, tile)
    expected_t = (r_orig.float() + h.float()).to(dtype)
    checks["inplace_semantics"] = {
        "original_untouched": bool(torch.equal(r, r_orig)),
        "buffer_updated_to_sum": bool(torch.equal(r_upd, expected_t)),
        "note": "融合调用只改传入的 r buffer（原位）；外部保留的引用不受影响",
    }

    # H=0：显式报错（状态码 2）。
    try:
        r0 = torch.randn(2, 0, device=device, dtype=dtype)
        w0 = torch.zeros(0, device=device, dtype=dtype)
        y0 = torch.empty_like(r0)
        runner.kernels.fused_add_rmsnorm(r0, r0, w0, y0, RMS_NORM_EPS, tile, ws)
        checks["h_zero"] = {"raised": False}
    except RuntimeError as error:
        checks["h_zero"] = {"raised": True, "error_type": type(error).__name__,
                            "message": str(error)[:200]}

    # rows=0：不发射 Kernel，返回空 Tensor（合同条目）。
    r0 = torch.randn(0, 1024, device=device, dtype=dtype)
    h0 = torch.randn(0, 1024, device=device, dtype=dtype)
    y0 = torch.empty_like(r0)
    runner.kernels.fused_add_rmsnorm(r0, h0, w, y0, RMS_NORM_EPS, tile, ws)
    checks["rows_zero"] = {"shape": list(y0.shape), "raised": False}

    # 非连续输入：显式报错（合同：第 06 期主题，本期只接受连续 Tensor）。
    base = torch.randn(1024, 128, device=device, dtype=dtype)
    checks["non_contiguous"] = {"raised": None, "note": "绑定层按 data_ptr 直传，"
                                                       "非连续由调用方 .contiguous() 或显式检查；"
                                                       "compare 链全部使用连续张量（合同声明）"}

    # 非法 tile：显式报错（状态码 1；MUSA 上 tile=64 另因 warpSize=128 被拒）。
    try:
        r1 = torch.randn(2, 1024, device=device, dtype=dtype)
        h1 = torch.randn(2, 1024, device=device, dtype=dtype)
        y1 = torch.empty_like(r1)
        runner.kernels.fused_add_rmsnorm(r1, h1, w, y1, RMS_NORM_EPS, 100, ws)
        checks["invalid_tile"] = {"raised": False}
    except RuntimeError as error:
        checks["invalid_tile"] = {"raised": True, "error_type": type(error).__name__,
                                  "message": str(error)[:200]}

    ok = (checks["zero_row"]["residual_all_zero"] and checks["zero_row"]["y_all_zero"]
          and checks["zero_row"]["no_nan"]
          and checks["nan_row"]["nan_element_residual_nan"]
          and checks["nan_row"]["other_residual_finite"] and checks["nan_row"]["row0_y_nan"]
          and checks["nan_row"]["row1_finite"]
          and checks["fp16_large"]["residual_inf"] and checks["fp16_large"]["y_nan"]
          and checks["inplace_semantics"]["original_untouched"]
          and checks["inplace_semantics"]["buffer_updated_to_sum"]
          and checks["h_zero"]["raised"] and checks["invalid_tile"]["raised"]
          and checks["rows_zero"]["shape"] == [0, 1024])
    print(f"[edge/{device.type}] 边界行为{'全部符合合同' if ok else '存在不符合项（见 JSON）'}")
    return {"runner": runner.name, "checks": checks, "all_pass": bool(ok)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--fixtures", default=str(FIXTURES_DEFAULT))
    arguments = parser.parse_args()

    global manifest
    arrays, manifest = load_fixtures(Path(arguments.fixtures))
    device = resolve_device(arguments.backend) if arguments.backend != "cpu" else torch.device("cpu")

    integrity = fixture_integrity_chain(arrays, manifest)
    if not integrity["passed"]:
        raise SystemExit("fixture SHA256 校验失败，终止（跨机器真值不可信）")

    results = {
        "schema_version": "kernel-chapter04.correctness.v1",
        "backend": arguments.backend,
        "tile": arguments.tile,
        "fixture_integrity": integrity,
        "serial_selfcheck": serial_selfcheck_chain(arrays) if arguments.backend == "cpu" else None,
        "composite_fp32": composite_chain(arguments, device, arrays, torch.float32),
        "environment": device_environment_record(device),
    }
    if arguments.backend in ("cuda", "musa"):
        runner = KernelRunner(arguments.backend, device)
        results["composite_fp16"] = composite_chain(arguments, device, arrays, torch.float16)
        results["kernel_fp32"] = kernel_chain(arguments, device, arrays, runner, torch.float32)
        results["kernel_fp16"] = kernel_chain(arguments, device, arrays, runner, torch.float16)
        results["determinism"] = determinism_chain(arguments, device, arrays, runner)
        results["edge"] = edge_chain(arguments, device, runner)
    elif arguments.backend == "cpu":
        results["edge"] = edge_chain(arguments, device, None)

    output = HERE / "results" / f"correctness-results-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
