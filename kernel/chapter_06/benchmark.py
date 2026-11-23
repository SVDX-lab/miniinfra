#!/usr/bin/env python3
"""第 06 期性能测量：三合同策略 × 三布局的延迟与有效带宽（统一口径）。

实验映射（大纲第 06 期实验清单）：
  E1 转置输入喂 RMSNorm 三策略延迟对照（显式报错无延迟，只有 A 拒绝；
     B=拷贝+kernel；C=stride 寻址 kernel）+ .contiguous() 隐性拷贝成本占比；
  E2 非连续输入有效带宽损失（对连续基线 A 的比值）；
  E3 "先转置回来再算"组织方式对照（layout 归一化拷贝成本 vs 摊销，
     含 transpose-back 陷阱的零成本证据）；
  copyops：布局归一化拷贝的裸成本（transpose-copy / 行 gather / 连续 clone）。

形状：H=1024 × rows {1, 128, 2048, 32768(=DRAM 区探针，128MB 工作集)}
     + H=127/129 抽查（rows=2048）。dtype：GPU fp32 主口径 + fp16 对照；
     CPU fp32。
非连续构造：
  - 转置视图 tv = base[H,rows].t() → 逻辑 [rows,H]、stride=(1,rows)：
    归一化维（H 维）步长 = rows —— 合并访存完全失效的最坏布局；
    rows=1 时退化连续（教学点，实测记录）。
  - 行切片视图 rs = big[:, :H] → stride=(H+pad,1)：行内合并访存保留，
    行间跳步 —— 大模型 slice 的常见布局。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from experiment_utils import (  # noqa: E402
    OperatorTimer,
    backend_sync,
    device_environment_record,
    resolve_device,
    write_results_json,
)
from stride_layout import RMS_NORM_EPS  # noqa: E402

HIDDEN = 1024
ROWS = [1, 128, 2048, 32768]
SPOT_HIDDEN = [127, 129]
PAD = 8
TILE = 256
PEAK_BANDWIDTH = {"cuda": 918.8e9, "musa": 711.3e9, "cpu": None}  # ch02 实测 D2D
N_ORG = 50  # 摊销实验的调用次数


def theoretical_bytes(rows: int, h: int, itemsize: int) -> int:
    return (rows * h * 2 + h) * itemsize  # 读 x + 写 y + 读 w


def repeats_for(rows: int) -> int:
    return {1: 200, 128: 100, 2048: 50, 32768: 8}[rows]


def warmup_for(rows: int) -> int:
    return 10 if rows <= 128 else 5


def seeded(shape: tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    seed = 20260701 + shape[0] * 100000 + shape[1]
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal(shape, dtype=np.float32)).to(device=device, dtype=dtype)


def build_inputs(rows: int, h: int, device: torch.device, dtype: torch.dtype):
    x = seeded((rows, h), device, dtype)              # L0 连续
    base = seeded((h, rows), device, dtype)           # 转置基：base.t() = 转置视图
    tv = base.t()
    big = seeded((rows, h + PAD), device, dtype)
    rs = big[:, :h]                                   # L2 行切片视图
    w = seeded((1, h), device, dtype)[0]
    return x, tv, rs, w


def layout_flags(tv: torch.Tensor, rs: torch.Tensor) -> dict:
    return {
        "tv_shape": list(tv.shape), "tv_stride": list(tv.stride()),
        "tv_is_contiguous": tv.is_contiguous(),
        "rs_stride": list(rs.stride()), "rs_is_contiguous": rs.is_contiguous(),
    }


def add_bandwidth(entry: dict, total_bytes: int, p50_ms: float, device_type: str) -> None:
    effective = total_bytes / (p50_ms / 1000.0)
    entry["effective_bandwidth_gb_s"] = round(effective / 1e9, 2)
    if PEAK_BANDWIDTH[device_type] is not None:
        entry["achievement_ratio"] = round(effective / PEAK_BANDWIDTH[device_type], 4)
        entry["peak_source"] = "chapter_01/performance 实测 D2D 拷贝峰值"


# ---------------------------------------------------------------------------
# GPU/MUSA：三策略链（E1 + E2）
# ---------------------------------------------------------------------------
def policy_chain(module, device: torch.device, dtype: torch.dtype, warmup: int) -> dict:
    entries = []
    dtype_str = str(dtype).replace("torch.", "")
    for rows in ROWS:
        x, tv, rs, w = build_inputs(rows, HIDDEN, device, dtype)
        itemsize = x.element_size()
        total = theoretical_bytes(rows, HIDDEN, itemsize)
        repeats = repeats_for(rows)
        timer = OperatorTimer(device, warmup=warmup_for(rows), repeats=repeats)

        entry = {"rows": rows, "hidden_size": HIDDEN, "dtype": dtype_str,
                 "theoretical_bytes": total, "layout": layout_flags(tv, rs)}

        p50s: dict[str, float] = {}
        ops = {
            "A_contig_baseline": lambda: module.rmsnorm_policy_a(x, w, RMS_NORM_EPS, TILE),
            "B_tv": lambda: module.rmsnorm_policy_b(tv, w, RMS_NORM_EPS, TILE),
            "C_tv": lambda: module.rmsnorm_policy_c(tv, w, RMS_NORM_EPS, TILE),
            "B_rs": lambda: module.rmsnorm_policy_b(rs, w, RMS_NORM_EPS, TILE),
            "C_rs": lambda: module.rmsnorm_policy_c(rs, w, RMS_NORM_EPS, TILE),
        }
        if tv.is_contiguous():
            # rows=1 退化：转置视图自动连续，策略 A 也能直接吃（教学点）。
            ops["A_tv_degenerate_contiguous"] = lambda: module.rmsnorm_policy_a(tv, w, RMS_NORM_EPS, TILE)
        for name, op in ops.items():
            timing = timer.measure(op)
            p50s[name] = timing["p50_ms"]
            sub = {"warmup": timing["warmup"], "repeats": timing["repeats"],
                   "p50_ms": timing["p50_ms"], "p95_ms": timing["p95_ms"],
                   "min_ms": timing["min_ms"]}
            add_bandwidth(sub, total, timing["p50_ms"], device.type)
            entry[name] = sub

        # 派生指标。
        kernel_only = p50s["A_contig_baseline"]
        entry["derived"] = {
            "copy_share_B_tv": round((p50s["B_tv"] - kernel_only) / p50s["B_tv"], 4)
            if p50s["B_tv"] > 0 else None,
            "copy_share_B_rs": round((p50s["B_rs"] - kernel_only) / p50s["B_rs"], 4)
            if p50s["B_rs"] > 0 else None,
            "C_tv_vs_A_latency_ratio": round(p50s["C_tv"] / kernel_only, 3) if kernel_only > 0 else None,
            "C_tv_vs_A_bandwidth_ratio": round(
                entry["A_contig_baseline"]["effective_bandwidth_gb_s"]
                / entry["C_tv"]["effective_bandwidth_gb_s"], 3),
            "C_rs_vs_A_bandwidth_ratio": round(
                entry["A_contig_baseline"]["effective_bandwidth_gb_s"]
                / entry["C_rs"]["effective_bandwidth_gb_s"], 3),
            "C_tv_slowdown_note": "stride_col=rows：归一化维步长=rows，合并访存失效"
            if not tv.is_contiguous() else "rows=1 转置视图退化为连续",
        }
        entries.append(entry)
        d = entry["derived"]
        print(f"[policy/{dtype_str}] rows={rows:<6} A={kernel_only:.3f}ms "
              f"B_tv={p50s['B_tv']:.3f}ms(拷贝占{d['copy_share_B_tv']:.1%}) "
              f"C_tv={p50s['C_tv']:.3f}ms({d['C_tv_vs_A_latency_ratio']}x, "
              f"带宽比={d['C_tv_vs_A_bandwidth_ratio']}) "
              f"C_rs带宽比={d['C_rs_vs_A_bandwidth_ratio']}")
    return {"dtype": dtype_str, "entries": entries}


def spot_chain(module, device: torch.device, warmup: int) -> dict:
    """H=127/129 非整 Tile 抽查（标量路径）× 转置视图。"""
    entries = []
    for h in SPOT_HIDDEN:
        rows = 2048
        x = seeded((rows, h), device, torch.float32)
        base = seeded((h, rows), device, torch.float32)
        tv = base.t()
        w = seeded((1, h), device, torch.float32)[0]
        timer = OperatorTimer(device, warmup=warmup, repeats=50)
        itemsize = 4
        total = theoretical_bytes(rows, h, itemsize)
        entry = {"rows": rows, "hidden_size": h, "dtype": "float32",
                 "theoretical_bytes": total, "layout": layout_flags(tv, tv)}
        p50s = {}
        for name, op in {
            "A_contig_baseline": lambda: module.rmsnorm_policy_a(x, w, RMS_NORM_EPS, TILE),
            "B_tv": lambda: module.rmsnorm_policy_b(tv, w, RMS_NORM_EPS, TILE),
            "C_tv": lambda: module.rmsnorm_policy_c(tv, w, RMS_NORM_EPS, TILE),
        }.items():
            timing = timer.measure(op)
            p50s[name] = timing["p50_ms"]
            sub = {"p50_ms": timing["p50_ms"], "p95_ms": timing["p95_ms"], "min_ms": timing["min_ms"]}
            add_bandwidth(sub, total, timing["p50_ms"], device.type)
            entry[name] = sub
        entry["derived"] = {
            "copy_share_B_tv": round((p50s["B_tv"] - p50s["A_contig_baseline"]) / p50s["B_tv"], 4),
            "C_tv_vs_A_latency_ratio": round(p50s["C_tv"] / p50s["A_contig_baseline"], 3),
        }
        entries.append(entry)
        print(f"[spot] H={h}: A={p50s['A_contig_baseline']:.3f}ms B_tv={p50s['B_tv']:.3f}ms "
              f"C_tv={p50s['C_tv']:.3f}ms ({entry['derived']['C_tv_vs_A_latency_ratio']}x)")
    return {"entries": entries}


# ---------------------------------------------------------------------------
# copyops：布局归一化拷贝裸成本（E3 的对照面 + transpose-back 陷阱）
# ---------------------------------------------------------------------------
def copyops_chain(device: torch.device, dtype: torch.dtype) -> dict:
    entries = []
    dtype_str = str(dtype).replace("torch.", "")
    for rows in [128, 2048, 32768]:
        x, tv, rs, _w = build_inputs(rows, HIDDEN, device, dtype)
        itemsize = x.element_size()
        copy_bytes = rows * HIDDEN * itemsize
        timer = OperatorTimer(device, warmup=10 if rows <= 128 else 5,
                              repeats=repeats_for(rows))
        entry = {"rows": rows, "dtype": dtype_str, "copy_bytes": copy_bytes}

        def timed(name, op):
            timing = timer.measure(op)
            sub = {"p50_ms": timing["p50_ms"], "p95_ms": timing["p95_ms"], "min_ms": timing["min_ms"]}
            add_bandwidth(sub, copy_bytes, timing["p50_ms"], device.type)
            entry[name] = sub
            return timing["p50_ms"]

        p_contig = timed("transpose_copy_tv_contiguous", lambda: tv.contiguous())
        p_gather = timed("row_gather_copy_rs_contiguous", lambda: rs.contiguous())
        p_clone = timed("contig_clone_baseline", lambda: x.clone())
        p_explicit = timed("explicit_copy_kernel", lambda: torch.empty(
            rows, HIDDEN, device=device, dtype=dtype).copy_(tv))
        trap_ptr_equal = tv.t().contiguous().data_ptr() == tv.t().data_ptr()
        p_trap = timed("transpose_back_trap_tv_t_contiguous", lambda: tv.t().contiguous())
        entry["trap"] = {
            "tv_t_is_contiguous": tv.t().is_contiguous(),
            "contiguous_returns_same_ptr": trap_ptr_equal,
            "p50_ms_zero_cost_evidence": p_trap,
        }
        entry["derived"] = {
            "transpose_copy_vs_clone_ratio": round(p_contig / p_clone, 3) if p_clone > 0 else None,
            "gather_copy_vs_clone_ratio": round(p_gather / p_clone, 3) if p_clone > 0 else None,
        }
        entries.append(entry)
        print(f"[copyops/{dtype_str}] rows={rows:<6} transpose={p_contig:.3f}ms "
              f"gather={p_gather:.3f}ms clone={p_clone:.3f}ms explicit={p_explicit:.3f}ms "
              f"trap={p_trap:.3f}ms(same_ptr={trap_ptr_equal})")
    return {"dtype": dtype_str, "entries": entries}


# ---------------------------------------------------------------------------
# org：先转置回来再算（一次性归一化 + N 次连续 kernel）vs 策略 B（每次拷贝）
# ---------------------------------------------------------------------------
def org_chain(module, device: torch.device, dtype: torch.dtype) -> dict:
    entries = []
    dtype_str = str(dtype).replace("torch.", "")
    for rows in [2048, 32768]:
        _x, tv, _rs, w = build_inputs(rows, HIDDEN, device, dtype)
        xc = tv.contiguous()  # 组织方式：生产侧一次性归一化（setup，不入被测循环）
        timer = OperatorTimer(device, warmup=3, repeats=10)

        def loop_b():
            for _ in range(N_ORG):
                module.rmsnorm_policy_b(tv, w, RMS_NORM_EPS, TILE)

        def loop_a():
            for _ in range(N_ORG):
                module.rmsnorm_policy_a(xc, w, RMS_NORM_EPS, TILE)

        t_b = timer.measure(loop_b)
        t_a = timer.measure(loop_a)
        one_copy = timer.measure(lambda: tv.contiguous())
        per_call_b = t_b["p50_ms"] / N_ORG
        per_call_a = t_a["p50_ms"] / N_ORG
        copy_p50 = one_copy["p50_ms"]
        savings_per_call = per_call_b - per_call_a
        entry = {
            "rows": rows, "dtype": dtype_str, "n_calls": N_ORG,
            "strategy_B_loop_total_p50_ms": t_b["p50_ms"],
            "explicit_org_loop_total_p50_ms": t_a["p50_ms"],
            "single_layout_copy_p50_ms": copy_p50,
            "per_call_B_ms": round(per_call_b, 4),
            "per_call_explicit_org_ms": round(per_call_a, 4),
            "savings_per_call_ms": round(savings_per_call, 4),
            "model_check_(N-1)*copy_ms": round((N_ORG - 1) * copy_p50, 4),
            "amortization_note": "策略 B 每次调用都付拷贝；显式组织只付一次，"
                                 "每次调用节省 ≈ 一份 layout copy（第 N-1 次起为纯收益）",
        }
        entries.append(entry)
        print(f"[org/{dtype_str}] rows={rows:<6} N={N_ORG}: B_total={t_b['p50_ms']:.2f}ms "
              f"explicit_total={t_a['p50_ms']:.2f}ms single_copy={copy_p50:.3f}ms "
              f"per_call_savings={savings_per_call:.4f}ms "
              f"(model {(N_ORG - 1) * copy_p50:.2f}ms)")
    return {"dtype": dtype_str, "entries": entries}


# ---------------------------------------------------------------------------
# CPU 列（本地全跑）：串行布局敏感度 + torch CPU 组合表达式 + numpy 拷贝
# ---------------------------------------------------------------------------
def cpu_chains(warmup: int) -> dict:
    from stride_layout import StridedTensor, naive_rmsnorm_strided

    results: dict = {"serial": {"entries": []}, "torch": {"entries": []}, "copyops": {"entries": []}}

    # 1) 朴素串行（策略 C）：连续 vs 转置视图 vs 行切片 —— 纯 Python，cache 局部性。
    for rows in [1, 128]:
        h = HIDDEN
        rng = np.random.default_rng(20260701 + rows * 100000 + h)
        x = rng.standard_normal((rows, h), dtype=np.float32)
        w = np.random.default_rng(20260701 + h).standard_normal(h, dtype=np.float32)
        entry = {"rows": rows, "hidden_size": h, "dtype": "float32", "repeats": 3}
        for name, st in {
            "L0_contig": StridedTensor.row_major(x),
            "L1_transposed": StridedTensor.transposed_view(x),
            "L2_row_sliced": StridedTensor.row_sliced_view(x, PAD),
        }.items():
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                naive_rmsnorm_strided(st, w, RMS_NORM_EPS)
                samples.append((time.perf_counter() - start) * 1000.0)
            entry[name + "_p50_ms"] = round(sorted(samples)[1], 2)
        entry["transposed_slowdown_ratio"] = round(
            entry["L1_transposed_p50_ms"] / entry["L0_contig_p50_ms"], 3)
        results["serial"]["entries"].append(entry)
        print(f"[cpu-serial] rows={rows}: contig={entry['L0_contig_p50_ms']}ms "
              f"tv={entry['L1_transposed_p50_ms']}ms "
              f"({entry['transposed_slowdown_ratio']}x) rs={entry['L2_row_sliced_p50_ms']}ms")

    # 2) torch CPU 组合表达式：连续 vs 转置视图 vs 行切片 + B 式（拷贝后算）。
    device = torch.device("cpu")
    for rows in [1, 128, 2048, 32768]:
        x, tv, rs, w = build_inputs(rows, HIDDEN, device, torch.float32)
        itemsize = x.element_size()
        total = theoretical_bytes(rows, HIDDEN, itemsize)
        repeats = 20 if rows <= 128 else (10 if rows == 2048 else 5)
        timer = OperatorTimer(device, warmup=3, repeats=repeats)

        def composite(inp):
            float_states = inp.float()
            variance = float_states.pow(2).mean(dim=-1, keepdim=True)
            return w * (float_states * torch.rsqrt(variance + RMS_NORM_EPS)).to(inp.dtype)

        entry = {"rows": rows, "hidden_size": HIDDEN, "dtype": "float32",
                 "theoretical_bytes": total, "layout": layout_flags(tv, rs)}
        p50s = {}
        ops = {"composite_L0_contig": lambda: composite(x),
               "composite_L1_transposed": lambda: composite(tv),
               "composite_L2_row_sliced": lambda: composite(rs)}
        for name, op in ops.items():
            timing = timer.measure(op)
            p50s[name] = timing["p50_ms"]
            sub = {"p50_ms": timing["p50_ms"], "p95_ms": timing["p95_ms"], "min_ms": timing["min_ms"]}
            add_bandwidth(sub, total, timing["p50_ms"], device.type)
            entry[name] = sub
        copy_timing = timer.measure(lambda: tv.contiguous())
        entry["layout_copy_p50_ms"] = copy_timing["p50_ms"]
        copy_bytes = rows * HIDDEN * itemsize
        add_bandwidth({"p50_ms": copy_timing["p50_ms"]}, copy_bytes, copy_timing["p50_ms"], device.type)
        entry["layout_copy_bandwidth_gb_s"] = round(copy_bytes / (copy_timing["p50_ms"] / 1000.0) / 1e9, 2)
        entry["derived"] = {
            "transposed_slowdown_ratio": round(p50s["composite_L1_transposed"] / p50s["composite_L0_contig"], 3),
            "row_sliced_slowdown_ratio": round(p50s["composite_L2_row_sliced"] / p50s["composite_L0_contig"], 3),
            "B_style_copy_share": round(copy_timing["p50_ms"] / (p50s["composite_L0_contig"] + copy_timing["p50_ms"]), 4),
        }
        results["torch"]["entries"].append(entry)
        print(f"[cpu-torch] rows={rows:<6}: contig={p50s['composite_L0_contig']:.3f}ms "
              f"tv={p50s['composite_L1_transposed']:.3f}ms"
              f"({entry['derived']['transposed_slowdown_ratio']}x) "
              f"rs={p50s['composite_L2_row_sliced']:.3f}ms copy={entry['layout_copy_p50_ms']:.3f}ms")

    # 3) numpy ascontiguousarray 拷贝带宽（policy B CPU 裸成本）。
    for rows in [2048, 32768]:
        h = HIDDEN
        rng = np.random.default_rng(20260701 + rows * 100000 + h)
        x = rng.standard_normal((rows, h), dtype=np.float32)
        st = StridedTensor.transposed_view(x)
        copy_bytes = rows * h * 4
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            np.ascontiguousarray(st.to_numpy_via_as_strided())
            samples.append((time.perf_counter() - start) * 1000.0)
        p50 = sorted(samples)[len(samples) // 2]
        results["copyops"]["entries"].append({
            "rows": rows, "dtype": "float32", "copy_bytes": copy_bytes,
            "ascontiguousarray_p50_ms": round(p50, 3),
            "effective_copy_bandwidth_gb_s": round(copy_bytes / (p50 / 1000.0) / 1e9, 2),
        })
        print(f"[cpu-copy] rows={rows}: ascontiguousarray={p50:.3f}ms "
              f"({results['copyops']['entries'][-1]['effective_copy_bandwidth_gb_s']}GB/s)")
    return results


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend) if arguments.backend != "cpu" else torch.device("cpu")
    if device.type != "cpu":
        memory_api = getattr(torch, device.type)
        memory_api.empty_cache()
        if hasattr(memory_api, "reset_peak_memory_stats"):
            memory_api.reset_peak_memory_stats()

    environment = device_environment_record(device)
    write_results_json(HERE / "results" / f"environment-{arguments.backend}.json",
                       {"schema_version": "kernel-chapter07.environment.v1",
                        "backend": arguments.backend, **environment})

    results = {
        "schema_version": "kernel-chapter07.benchmark.v1",
        "backend": arguments.backend,
        "tile": TILE,
        "pad": PAD,
        "rows": ROWS,
        "org_calls": N_ORG,
        "peak_bandwidth_source": "chapter_01/performance 实测 D2D（cuda 918.8 / musa 711.3 GB/s）",
        "environment": environment,
    }

    if arguments.backend == "cpu":
        results["cpu_chains"] = cpu_chains(warmup=3)
    else:
        import importlib.util

        path = HERE / "implementations" / arguments.backend / "build_ext.py"
        spec = importlib.util.spec_from_file_location(f"build_ext_{arguments.backend}", path)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        start = time.perf_counter()
        module = loader.get_module()
        build_seconds = time.perf_counter() - start
        results["kernel_build_seconds"] = round(build_seconds, 1)

        results["policy_fp32"] = policy_chain(module, device, torch.float32, warmup=10)
        results["policy_fp16"] = policy_chain(module, device, torch.float16, warmup=10)
        results["spot_h127_h129"] = spot_chain(module, device, warmup=10)
        results["copyops_fp32"] = copyops_chain(device, torch.float32)
        results["copyops_fp16"] = copyops_chain(device, torch.float16)
        results["org_fp32"] = org_chain(module, device, torch.float32)
        results["org_fp16"] = org_chain(module, device, torch.float16)

        if device.type == "cuda":
            results["peak_memory_bytes"] = torch.cuda.max_memory_allocated()

    output = HERE / "results" / f"benchmark-results-{arguments.backend}.json"
    write_results_json(output, results)
    backend_sync(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
