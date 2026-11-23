#!/usr/bin/env python3
"""第 05 期三平台正确性对照：三 dtype Kernel/组合表达式 vs CPU FP32 串行真值。

对照链（30 组合同 Shape = 6 H × 5 rows，与第 02 期同一份数据口径）：
1. fixture 完整性：逐张量 SHA256 与 manifest 核对（含实验 4 的 acc 构造输入）；
2. serial 自洽（仅 CPU）：真值位级重算；
3. composite vs 真值：FP32 / FP16 / BF16（平台原生 torch 路径，30 组）；
4. kernel vs 真值：FP32 / FP16 / BF16 各 30 组（本期手写 Kernel 三条编译路径，
   累加固定 FP32 —— 合同默认口径；CPU 后端的 kernel 角色由模拟路径承担）；
5. 累加 dtype 实验（实验 4，acc 构造输入）：kernel ACC=FP32 vs ACC=FP16
   （存储固定 FP32 以隔离变量）；CPU 侧另有位精确累加模拟器交叉验证；
6. 宽动态行链（"bf16 何时更准"）：小值分支 fp16 vs bf16 相对误差；
7. 确定性：同 Tile 3 次位级一致；
8. 边界行为：全零行、NaN 行、FP16 大数、BF16 大数/小数、H=0、rows=0、
   非连续输入、非法 tile、FP16 累加溢出行（ACC=fp16 输出全零语义）。

用法：
    python compare_correctness.py --backend cpu
    python compare_correctness.py --backend cuda
    python compare_correctness.py --backend musa --kernel-backend auto
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
from naive_rmsnorm import RMS_NORM_EPS, naive_rmsnorm_serial  # noqa: E402

FIXTURES_DEFAULT = HERE / "fixtures" / "correctness.npz"
TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_fixtures(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    with path.with_name("manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    return arrays, manifest


def _sha(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def fixture_integrity_chain(arrays: dict[str, np.ndarray], manifest: dict) -> dict:
    mismatches = []
    checked = 0
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        tensor_keys = {
            "x": f"x_{rows}x{hidden_size}",
            "w": f"w_{hidden_size}",
            "y": f"y_{rows}x{hidden_size}",
        }
        for name, tensor_key in tensor_keys.items():
            checked += 1
            if _sha(arrays[tensor_key]) != combo["sha256"][name]:
                mismatches.append(tensor_key)
    for entry in manifest.get("acc_inputs", []):
        for name in ("x", "w", "y"):
            checked += 1
            tensor_key = f"acc{name}_{entry['tag']}"
            if _sha(arrays[tensor_key]) != entry["sha256"][name]:
                mismatches.append(tensor_key)
    print(f"[fixtures] SHA256 完整性：{checked - len(mismatches)}/{checked} 通过"
          + (f"，不一致: {mismatches}" if mismatches else ""))
    return {"tensors_checked": checked, "mismatches": mismatches, "passed": not mismatches}


class ExtRunner:
    """torch 扩展路径（CUDA: cpp_extension.load；MUSA: torch_musa musa_extension）。"""

    name = "torch-extension"

    def __init__(self, backend: str):
        import importlib.util

        path = HERE / "implementations" / backend / "build_ext.py"
        spec = importlib.util.spec_from_file_location(f"build_ext_{backend}", path)
        module_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_loader)
        self.module = module_loader.get_module()

    def run(self, x: torch.Tensor, w: torch.Tensor, tile: int, acc: str = "fp32") -> torch.Tensor:
        return self.module.rmsnorm(x, w, RMS_NORM_EPS, tile, acc)


class BenchRunner:
    """独立可执行程序 + 文件 IO 路径（扩展路径不通时的退路）。"""

    name = "standalone-fileio"

    def __init__(self, backend: str):
        impl_dir = HERE / "implementations" / backend
        self.binary = impl_dir / "build" / "rmsnorm_bench"
        if not self.binary.is_file():
            raise RuntimeError(f"rmsnorm_bench 未构建：{self.binary}")
        self.backend = backend

    def run(self, x: torch.Tensor, w: torch.Tensor, tile: int, acc: str = "fp32") -> torch.Tensor:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            x.detach().float().cpu().contiguous().numpy().tofile(tmp_path / "x.bin")
            w.detach().float().cpu().contiguous().numpy().tofile(tmp_path / "w.bin")
            dtype = {torch.float32: "fp32", torch.float16: "fp16",
                     torch.bfloat16: "bf16"}[x.dtype]
            command = [
                str(self.binary),
                "--h", str(x.shape[1]), "--rows", str(x.shape[0]),
                "--dtype", dtype, "--acc", acc, "--tiles", str(tile),
                "--load", str(tmp_path / "x.bin"), str(tmp_path / "w.bin"),
                "--out", str(tmp_path / "y.bin"),
                "--warmup", "1", "--repeats", "1", "--check", "false",
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(
                    f"rmsnorm_bench 失败（rc={result.returncode}）：{result.stderr[-2000:]}")
            y_np = np.fromfile(tmp_path / "y.bin", dtype=np.float32)
        return torch.from_numpy(y_np.reshape(x.shape)).to(x.device, dtype=x.dtype)


class CpuSimRunner:
    """CPU 模拟 Kernel 路径（CPU 后端的 kernel 角色：模拟三条编译路径的数值语义）。

    与 GPU Kernel 对应：读 T -> 转 FP32 -> 平方和在 ACC 算术内累加 -> 写回 T。
    """

    name = "cpu-simulated"

    def run_array(self, x: np.ndarray, w: np.ndarray, dtype: str, acc: str) -> np.ndarray:
        from dtype_sim import rmsnorm_simulated_kernel

        return rmsnorm_simulated_kernel(x, w, RMS_NORM_EPS, dtype, acc)["y"]


def make_kernel_runner(backend: str, preference: str, device: torch.device):
    if backend == "cpu":
        return CpuSimRunner()
    if preference == "bench":
        return BenchRunner(backend)
    try:
        runner = ExtRunner(backend)
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.randn(4, 128, device=device, dtype=dt)
            w = torch.randn(128, device=device, dtype=dt)
            runner.run(x, w, 256)
        return runner
    except Exception as error:  # noqa: BLE001
        if preference == "ext":
            raise
        print(f"[kernel] torch 扩展路径不可用（{type(error).__name__}: {error}），"
              f"退到独立可执行程序 + 文件 IO 对比路径")
        return BenchRunner(backend)


def composite_chain(arguments, device: torch.device, arrays, dtype_name: str) -> dict:
    """平台原生组合表达式（torch 三 dtype）vs 真值，30 组。"""
    dtype = TORCH_DTYPES[dtype_name]
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
        float_states = x.float()
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        result = weight_cache[hidden_size] * (
            float_states * torch.rsqrt(variance + RMS_NORM_EPS)).to(dtype)
        metrics = comparison_metrics(result, expected)
        metrics.update({"rows": rows, "hidden_size": hidden_size, "dtype": dtype_name})
        groups.append(metrics)
    worst = max(groups, key=lambda item: item["max_abs_error"])
    nan_inf_total = sum(item["nan_count"] + item["inf_count"] for item in groups)
    print(f"[composite/{arguments.backend}/{dtype_name}] {len(groups)} 组 | "
          f"最差组 {worst['rows']}x{worst['hidden_size']} max_abs={worst['max_abs_error']:.3e} "
          f"cos={worst['cosine_similarity']:.8f} NaN/Inf={nan_inf_total}")
    return {"dtype": dtype_name, "groups": groups, "worst_group": worst,
            "nan_inf_total": nan_inf_total}


def kernel_chain(arguments, device: torch.device, arrays, runner, dtype_name: str) -> dict:
    """本期手写 Kernel 三条编译路径 vs 真值，30 组（acc=fp32 合同默认）。"""
    dtype = TORCH_DTYPES[dtype_name]
    weight_cache: dict[int, torch.Tensor] = {}
    groups = []
    for combo in manifest["combos"]:
        rows, hidden_size = combo["rows"], combo["hidden_size"]
        key = f"{rows}x{hidden_size}"
        expected = torch.from_numpy(arrays[f"y_{key}"])
        tile = arguments.tile
        vec_path = hidden_size % (8 if dtype_name != "fp32" else 4) == 0
        if isinstance(runner, CpuSimRunner):
            result_np = runner.run_array(arrays[f"x_{key}"], arrays[f"w_{hidden_size}"],
                                         dtype_name, "fp32")
            result = torch.from_numpy(np.asarray(result_np, dtype=np.float32))
        else:
            x = torch.from_numpy(arrays[f"x_{key}"]).to(device=device, dtype=dtype)
            if hidden_size not in weight_cache:
                weight_cache[hidden_size] = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(
                    device=device, dtype=dtype)
            result = runner.run(x, weight_cache[hidden_size], tile)
        metrics = comparison_metrics(result, expected)
        metrics.update({
            "rows": rows, "hidden_size": hidden_size, "dtype": dtype_name,
            "acc": "fp32", "tile": tile, "vec_path": vec_path,
        })
        groups.append(metrics)
    worst = max(groups, key=lambda item: item["max_abs_error"])
    nan_total = sum(item["nan_count"] for item in groups)
    inf_total = sum(item["inf_count"] for item in groups)
    print(f"[kernel/{arguments.backend}/{dtype_name}] {len(groups)} 组（runner={runner.name}, "
          f"tile={arguments.tile}）| 最差组 {worst['rows']}x{worst['hidden_size']} "
          f"max_abs={worst['max_abs_error']:.3e} cos={worst['cosine_similarity']:.8f} "
          f"NaN={nan_total} Inf={inf_total}")
    return {"dtype": dtype_name, "acc": "fp32", "runner": runner.name, "tile": arguments.tile,
            "groups": groups, "worst_group": worst, "nan_total": nan_total,
            "inf_total": inf_total}


def acc_chain(arguments, device: torch.device, arrays, runner) -> dict:
    """实验 4：累加 dtype（FP32 vs FP16）对 RMSNorm 误差的影响。

    存储固定 FP32（隔离变量：唯一差异是平方和累加链的算术）。
    构造输入 = fixtures acc_inputs（大范数行 scale 扫描 + 长 H 扫描 + 宽动态行）。
    """
    from dtype_sim import SERIAL_SQSUM

    entries = []
    for spec in manifest["acc_inputs"]:
        tag = spec["tag"]
        x_np = arrays[f"accx_{tag}"]
        w_np = arrays[f"accw_{tag}"]
        expected = torch.from_numpy(arrays[f"accy_{tag}"])
        entry = {"tag": tag, "kind": spec["kind"], "rows": spec["rows"],
                 "hidden_size": spec["hidden_size"],
                 "fp16_acc_overflow_expected": spec.get("fp16_acc_overflow_expected"),
                 "overflow_via_element_sq": spec.get("element_sq_overflow_expected"),
                 "overflow_via_sum": spec.get("sum_overflow_expected")}
        if spec["kind"] == "large_norm":
            entry["scale"] = spec["scale"]
            entry["theoretical_sqsum"] = spec["theoretical_sqsum"]
        # CPU 位精确模拟器（三格式平方和诊断，标量链，廉价）
        sq_diag = {}
        for acc_name, fn in SERIAL_SQSUM.items():
            acc_val, ovf = fn(x_np[0])
            sq_diag[acc_name] = {"sqsum_row0": acc_val, "overflow_row0": ovf}
        entry["cpu_sim_sqsum_diag"] = sq_diag
        if isinstance(runner, CpuSimRunner):
            out_fp32acc = torch.from_numpy(np.asarray(
                runner.run_array(x_np, w_np, "fp32", "fp32"), dtype=np.float32))
            out_fp16acc = torch.from_numpy(np.asarray(
                runner.run_array(x_np, w_np, "fp32", "fp16"), dtype=np.float32))
        else:
            x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            w = torch.from_numpy(w_np).to(device=device, dtype=torch.float32)
            out_fp32acc = runner.run(x, w, arguments.tile, acc="fp32").float().cpu()
            try:
                out_fp16acc = runner.run(x, w, arguments.tile, acc="fp16").float().cpu()
            except Exception as error:  # noqa: BLE001 —— 平台边界（如 ACC=half 蝶形被拒）
                entry["acc_fp16_error"] = f"{type(error).__name__}: {str(error)[:300]}"
                entry["platform_boundary"] = True
                out_fp16acc = None
        m32 = comparison_metrics(out_fp32acc, expected)
        m32.update({"acc": "fp32"})
        entry["acc_fp32"] = m32
        if out_fp16acc is not None:
            m16 = comparison_metrics(out_fp16acc, expected)
            m16.update({"acc": "fp16"})
            zeroed_rows = int((out_fp16acc.abs().sum(dim=-1) == 0).sum().item())
            m16["zeroed_rows"] = zeroed_rows
            m16["rows_total"] = spec["rows"]
            m16["output_all_zero"] = bool((out_fp16acc == 0).all())
            # 先验对账口径：expected 溢出 ⇔ 实测至少一行输出全零（scale=rsqrtf(inf)=0）
            m16["overflow_agrees_with_prior"] = (
                (zeroed_rows > 0) == bool(spec.get("fp16_acc_overflow_expected")))
            # 先验不一致的归因：sum 路径 + 串行累加 → "卡死区"（ulp(acc) > 增量，
            # 累加器冻结在 max 以下而不溢出 —— 树形归约（GPU 蝶形）无此现象）
            m16["sticking_suspected"] = (
                zeroed_rows == 0
                and bool(spec.get("sum_overflow_expected"))
                and not bool(spec.get("element_sq_overflow_expected")))
            entry["acc_fp16"] = m16
        entries.append(entry)
        m16s = entry.get("acc_fp16")
        print(f"[acc/{arguments.backend}] {tag:<12} H={spec['hidden_size']:<6} "
              f"fp32acc max={m32['max_abs_error']:.3e} | "
              + (f"fp16acc max={m16s['max_abs_error']:.3e} zeroed={m16s['zeroed_rows']}/{m16s['rows_total']} "
                 f"prior_agree={m16s['overflow_agrees_with_prior']}"
                 f"{'(sticking)' if m16s['sticking_suspected'] else ''}"
                 if m16s else f"fp16acc 平台边界（{entry.get('acc_fp16_error', '')[:60]}）"))
    checked = [e for e in entries if "acc_fp16" in e]
    bad = [e["tag"] for e in checked if not e["acc_fp16"]["overflow_agrees_with_prior"]]
    sticking = [e["tag"] for e in checked if e["acc_fp16"]["sticking_suspected"]]
    print(f"[acc/{arguments.backend}] 溢出先验与实测一致: "
          f"{len(checked) - len(bad)}/{len(checked)}"
          + (f"（不一致: {bad}；其中卡死区归因: {sticking}）" if bad else ""))
    return {"entries": entries, "runner": runner.name,
            "prior_agree": f"{len(checked) - len(bad)}/{len(checked)}",
            "sticking_tags": sticking}


def wide_dynamic_chain(arguments, device: torch.device, arrays, runner) -> dict:
    """宽动态范围行（"bf16 何时更准"）：小值分支的 fp16 vs bf16 相对误差。"""
    entries = []
    for i in range(4):
        tag = f"wide_r{i}"
        x_np = arrays[f"accx_{tag}"]
        w_np = arrays[f"accw_{tag}"]
        expected = torch.from_numpy(arrays[f"accy_{tag}"]).float()
        small_mask = torch.from_numpy(np.abs(x_np[0]) < 1e-5)
        row = {"tag": tag}
        for dtype_name in ("fp16", "bf16"):
            dtype = TORCH_DTYPES[dtype_name]
            if isinstance(runner, CpuSimRunner):
                out = torch.from_numpy(np.asarray(
                    runner.run_array(x_np, w_np, dtype_name, "fp32"), dtype=np.float32))
            else:
                x = torch.from_numpy(x_np).to(device=device, dtype=dtype)
                w = torch.from_numpy(w_np).to(device=device, dtype=dtype)
                out = runner.run(x, w, arguments.tile).float().cpu()
            err = (out - expected).abs()
            small_branch_rel = float((err[0][small_mask] /
                                      expected[0][small_mask].abs().clamp_min(1e-30)).max())
            big_branch_rel = float((err[0][~small_mask] /
                                    expected[0][~small_mask].abs().clamp_min(1e-30)).max())
            row[dtype_name] = {"small_branch_max_rel": small_branch_rel,
                               "big_branch_max_rel": big_branch_rel,
                               "overall_max_abs": float(err.max())}
        row["bf16_wins_small_branch"] = (
            row["bf16"]["small_branch_max_rel"] < row["fp16"]["small_branch_max_rel"])
        entries.append(row)
        print(f"[wide/{arguments.backend}] {tag}: fp16 small_rel={row['fp16']['small_branch_max_rel']:.3e} "
              f"bf16 small_rel={row['bf16']['small_branch_max_rel']:.3e} "
              f"bf16_wins={row['bf16_wins_small_branch']}")
    return {"entries": entries}


def determinism_chain(arguments, device: torch.device, arrays, runner) -> dict:
    rows, hidden_size = 128, 1024
    key = f"{rows}x{hidden_size}"
    results = {}
    for dtype_name in ("fp16", "bf16"):
        dtype = TORCH_DTYPES[dtype_name]
        if isinstance(runner, CpuSimRunner):
            runs = [torch.from_numpy(np.asarray(
                runner.run_array(arrays[f"x_{key}"], arrays[f"w_{hidden_size}"],
                                 dtype_name, "fp32"), dtype=np.float32)) for _ in range(3)]
        else:
            x = torch.from_numpy(arrays[f"x_{key}"]).to(device=device, dtype=dtype)
            w = torch.from_numpy(arrays[f"w_{hidden_size}"]).to(device=device, dtype=dtype)
            runs = [runner.run(x, w, arguments.tile).detach().float().cpu().contiguous()
                    for _ in range(3)]
        bitwise = all(bytes(runs[0].numpy().tobytes()) == bytes(r.numpy().tobytes())
                      for r in runs[1:])
        results[dtype_name] = {"runs": 3, "bitwise_identical": bool(bitwise)}
    print(f"[determinism] 同 tile 3 次位级一致: {results}")
    return results


def edge_chain(arguments, device: torch.device, arrays, runner) -> dict:
    checks = {}

    def run2(x: torch.Tensor, w: torch.Tensor, acc: str = "fp32") -> torch.Tensor:
        if isinstance(runner, CpuSimRunner):
            dtype_name = {torch.float32: "fp32", torch.float16: "fp16",
                          torch.bfloat16: "bf16"}[x.dtype]
            # 存储舍入已发生在 torch 的 dtype 转换里；float() 即 Kernel 的
            # "读 T 转 FP32"步骤（bf16 无 numpy 原生类型，走 float 中转）。
            return torch.from_numpy(np.asarray(
                runner.run_array(x.float().numpy(), w.float().numpy(), dtype_name, acc),
                dtype=np.float32))
        return runner.run(x, w, arguments.tile, acc).float().cpu()

    w16 = torch.randn(1024, device=device, dtype=torch.float16)
    # 全零行
    x = torch.randn(3, 1024, device=device, dtype=torch.float16)
    x[1] = 0
    y = run2(x, w16)
    checks["zero_row"] = {"all_zero": bool((y[1] == 0).all()),
                          "no_nan": bool(~torch.isnan(y[1]).any())}
    # NaN 行
    x = torch.randn(2, 1024, device=device, dtype=torch.float16)
    x[0, 0] = float("nan")
    y = run2(x, w16)
    checks["nan_row"] = {"row0_all_nan": bool(torch.isnan(y[0]).all()),
                         "row1_finite": bool(torch.isfinite(y[1]).all())}
    # FP16 大数（平方和在 FP32 累加下不溢出）
    x = torch.full((1, 1024), 65504.0, device=device, dtype=torch.float16)
    y = run2(x, w16)
    checks["fp16_large"] = {"finite": bool(torch.isfinite(y).all())}
    # BF16 大数（fp16 表示不了的量级）
    wbf = torch.randn(1024, device=device, dtype=torch.bfloat16)
    x = torch.full((1, 1024), 1e30, device=device, dtype=torch.bfloat16)
    y = run2(x, wbf)
    checks["bf16_large"] = {"finite": bool(torch.isfinite(y).all()),
                            "note": "x²=1e60 在 FP32 累加下不溢出（FP32 max≈3.4e38，"
                                    "平方和 1e30²×1024≈1e33 < 3.4e38）"}
    # BF16 小数（fp16 会下溢为 0 的量级）
    x = torch.full((1, 1024), 1e-30, device=device, dtype=torch.bfloat16)
    y = run2(x, wbf)
    checks["bf16_tiny"] = {"finite": bool(torch.isfinite(y).all()),
                           "any_nonzero": bool((y != 0).any()),
                           "note": "1e-30 在 bf16 normal 范围内；fp16 存储会下溢为 0"}
    # FP16 存储同量级小数 → 下溢为 0 → 输出全零（对照）
    x = torch.full((1, 1024), 1e-30, device=device, dtype=torch.float16)
    y = run2(x, w16)
    checks["fp16_tiny"] = {"all_zero": bool((y == 0).all()),
                           "note": "1e-30 < fp16 min subnormal 2^-24：输入已是 0，"
                                   "scale=1/sqrt(eps)~1000，y=0*1000=0"}
    # H=0 显式报错
    try:
        run2(torch.randn(2, 0, device=device, dtype=torch.float16),
             torch.zeros(0, device=device, dtype=torch.float16))
        checks["h_zero"] = {"raised": False}
    except Exception as error:  # noqa: BLE001
        checks["h_zero"] = {"raised": True, "error_type": type(error).__name__,
                            "message": str(error)[:300]}
    # rows=0
    y = run2(torch.randn(0, 1024, device=device, dtype=torch.float16), w16)
    checks["rows_zero"] = {"shape": list(y.shape)}
    # 非连续输入
    base = torch.randn(1024, 128, device=device, dtype=torch.float16)
    try:
        run2(base.t(), torch.randn(128, device=device, dtype=torch.float16))
        checks["non_contiguous"] = {"raised": False}
    except Exception as error:  # noqa: BLE001
        checks["non_contiguous"] = {"raised": True, "error_type": type(error).__name__,
                                    "message": str(error)[:300]}
    # 非法 tile
    try:
        if isinstance(runner, CpuSimRunner):
            raise ValueError("CPU 模拟路径无 tile 维度")
        runner.run(torch.randn(2, 1024, device=device, dtype=torch.float16), w16, 100)
        checks["invalid_tile"] = {"raised": False}
    except Exception as error:  # noqa: BLE001
        checks["invalid_tile"] = {"raised": True, "error_type": type(error).__name__}
    # ACC=fp16 溢出行：|x|~300 → 平方逐项 > 65504 → inf → 输出全零
    x = torch.full((1, 1024), 300.0, device=device, dtype=torch.float32)
    y16acc = run2(x, torch.randn(1024, device=device, dtype=torch.float32), acc="fp16")
    checks["acc_fp16_overflow"] = {"output_all_zero": bool((y16acc == 0).all()),
                                   "note": "FP16 累加下 x²=90000 > 65504 逐项溢出"}
    # 不支持的 dtype（int8）显式报错
    try:
        if isinstance(runner, CpuSimRunner):
            raise ValueError("CPU 模拟路径 dtype 只支持三浮点")
        runner.run(torch.randn(2, 1024, device=device, dtype=torch.int8),
                   torch.randn(1024, device=device, dtype=torch.int8), arguments.tile)
        checks["invalid_dtype"] = {"raised": False}
    except Exception as error:  # noqa: BLE001
        checks["invalid_dtype"] = {"raised": True, "error_type": type(error).__name__}

    if isinstance(runner, CpuSimRunner):
        checks["cpu_sim_note"] = ("CPU 模拟路径不模拟 shape/tile 合同检查"
                                  "（见 naive_rmsnorm._validate_inputs）")
    ok = (checks["zero_row"]["all_zero"] and checks["zero_row"]["no_nan"]
          and checks["nan_row"]["row0_all_nan"] and checks["nan_row"]["row1_finite"]
          and checks["fp16_large"]["finite"] and checks["bf16_large"]["finite"]
          and checks["bf16_tiny"]["any_nonzero"] and checks["fp16_tiny"]["all_zero"]
          and checks["acc_fp16_overflow"]["output_all_zero"]
          and checks["rows_zero"]["shape"] == [0, 1024])
    print(f"[edge] 边界行为{'全部符合合同' if ok else '存在不符合项（见 JSON）'}")
    return {"runner": runner.name, "checks": checks, "all_pass": ok}


def serial_selfcheck_chain(arrays) -> dict:
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
    print(f"[serial] 真值自洽：{len(manifest['combos']) - mismatches}/"
          f"{len(manifest['combos'])} 组位级一致（max_diff={max_diff:.1e}）")
    return {"groups": len(manifest["combos"]), "bitwise_mismatch_groups": mismatches,
            "max_abs_diff": max_diff}


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

    runner = make_kernel_runner(arguments.backend, arguments.kernel_backend, device)

    results = {
        "schema_version": "kernel-chapter06.correctness.v1",
        "backend": arguments.backend,
        "tile": arguments.tile,
        "fixture_integrity": integrity,
        "serial_selfcheck": serial_selfcheck_chain(arrays) if arguments.backend == "cpu" else None,
        "composite_fp32": composite_chain(arguments, device, arrays, "fp32"),
        "environment": device_environment_record(device),
    }
    results["composite_fp16"] = composite_chain(arguments, device, arrays, "fp16")
    results["composite_bf16"] = composite_chain(arguments, device, arrays, "bf16")
    results["kernel_fp32"] = kernel_chain(arguments, device, arrays, runner, "fp32")
    results["kernel_fp16"] = kernel_chain(arguments, device, arrays, runner, "fp16")
    results["kernel_bf16"] = kernel_chain(arguments, device, arrays, runner, "bf16")
    results["acc_experiment"] = acc_chain(arguments, device, arrays, runner)
    results["wide_dynamic"] = wide_dynamic_chain(arguments, device, arrays, runner)
    results["determinism"] = determinism_chain(arguments, device, arrays, runner)
    results["edge"] = edge_chain(arguments, device, arrays, runner)

    output = HERE / "results" / f"correctness-results-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
