"""第 14 期 CUDA 列（回填会话）：torch.compile GPU 延迟 + mode 阶梯。

回填版 v2（vs 初稿的差异）：
- 每个 (mode, dtype) 共享一个 compiled 实例跨 rows 序列（自动 dynamic 处理
  shape 变化）——编译次数从 24 降到 ~8，会话预算内；每次首调墙钟单独记录
  （含编译/重编译成本，重编译行为数据顺带产出）；
- mode 阶梯 = eager / inductor（default） / reduce-overhead（自动 CUDA Graph，
  接 ch10 Graph 化数字：ch10 intercept 2.4µs/slope 0.84µs·node、小形状 graph
  地板 ~14µs；ch04 launch 事件对地板 ~21–24µs）；
  max-autotune-no-cudagraphs 默认不跑（整点 autotune 耗时失控，会话预算取舍，
  如实记录；--with-max-autotune 可选开启）；
- DRAM 区（32768×1024）额外算有效带宽与达成率（分母 = ch02 实测拷贝峰值
  918.8 GB/s，单源口径）；
- ch04 手写融合锚点直接引用（不重测）：fp32@2048 0.0237 / fp16@2048 0.0269 /
  fp32@32768 0.5873 / fp16@32768 0.2946 ms（兑现率 1.05–1.14）；
  ch14 Triton 手写融合锚点：fp32@32768 0.5878 / fp16@32768 0.2929 ms（DRAM
  区与手写比值 1.000）。

数值正确性在 correctness_cuda.py（vs fixtures FP32 真值）。
产出 results/benchmark-results-cuda.json。
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiment_utils import OperatorTimer, write_results_json

CHAPTER = Path(__file__).resolve().parent.parent.parent
SCHEMA_VERSION = "kernel-chapter15.cuda-latency.v2"

CH04_ANCHORS = {
    "fused_fp32_rows2048_ms": 0.0237,
    "fused_fp16_rows2048_ms": 0.0269,
    "fused_fp32_rows32768_ms": 0.5873,
    "fused_fp16_rows32768_ms": 0.2946,
    "sep_fp16_rows2048_ms": 0.0362,
    "sep_fp32_rows2048_ms": 0.0314,
    "sep_fp16_rows32768_ms": 0.4155,
    "sep_fp32_rows32768_ms": 0.7703,
    "copy_peak_gbps": 918.8,
}
CH14_ANCHORS = {
    "triton_fused_fp32_rows32768_ms": 0.5878,
    "triton_fused_fp16_rows32768_ms": 0.2929,
    "triton_launch_floor_eager_us": 31.0,
    "handwritten_ctypes_floor_us": 23.0,
}


def add_rmsnorm(r, h, w, eps: float = 1e-6):
    t = r + h
    ft = t.float()
    var = ft.pow(2).mean(-1, keepdim=True)
    y = w * (ft * torch.rsqrt(var + eps)).to(t.dtype)
    return t, y


def make_inputs(rows: int, hidden: int, dtype, seed_base: int = 20260415):
    gen = torch.Generator(device="cpu").manual_seed(seed_base + rows)
    r = torch.randn(rows, hidden, generator=gen, dtype=torch.float32).to(
        device="cuda", dtype=dtype
    )
    h = torch.randn(rows, hidden, generator=gen, dtype=torch.float32).to(
        device="cuda", dtype=dtype
    )
    w = torch.randn(hidden, generator=gen, dtype=torch.float32).to(
        device="cuda", dtype=dtype
    )
    return r, h, w


def counters_summary() -> dict:
    counters = torch._dynamo.utils.counters
    out: dict[str, dict[str, int]] = {}
    for key, sub in counters.items():
        try:
            out[str(key)] = {str(k): int(v) for k, v in sub.items()}
        except TypeError:
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,128,2048,32768")
    parser.add_argument(
        "--modes", default="eager,inductor,reduce-overhead",
        help="max-autotune-no-cudagraphs 可加（耗时失控默认不跑，取舍已记录）",
    )
    parser.add_argument("--with-max-autotune", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")
    if torch.__version__ != "2.12.1+cu130":
        raise SystemExit(
            f"torch 版本门禁失败：{torch.__version__} != 2.12.1+cu130"
            "（ep24 升 2.14 降级中——同代性声明要求，等待降级完成后重跑）"
        )

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_version_gate": "2.12.1+cu130 PASS（同代性声明：CPU 列 2.6/CUDA 列 2.12 各自全量内对照，跨代结论不做）",
        "driver_device": {
            "name": properties.name,
            "cc": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
        },
        "triton": _triton_version(),
        "anchors": {"ch04": CH04_ANCHORS, "ch14": CH14_ANCHORS,
                    "ch10_graph": {"intercept_us": 2.4, "slope_us_per_node": 0.84,
                                   "small_shape_graph_floor_ms": 0.0143}},
        "rows_sequence": None,
        "entries": [],
    }
    rows_list = [int(v) for v in args.rows.split(",")]
    modes = list(args.modes.split(","))
    if args.with_max_autotune:
        modes.append("max-autotune-no-cudagraphs")
    payload["modes"] = modes

    for dtype in (torch.float16, torch.float32):
        dtype_key = str(dtype).replace("torch.", "")
        anchor_key = "fp16" if dtype == torch.float16 else "fp32"
        # eager 基线：每 (rows, dtype) 测一次（mode 无关，供全部 mode 引用）
        eager_by_rows: dict[int, dict] = {}
        for rows in rows_list:
            r, h, w = make_inputs(rows, 1024, dtype)
            torch.cuda.synchronize()
            warmup, repeats = (3, 10) if rows <= 2048 else (3, 10)
            eager_latency = OperatorTimer(device, warmup, repeats).measure(
                lambda: add_rmsnorm(r, h, w)
            )
            eager_by_rows[rows] = eager_latency
            del r, h, w
        for mode in modes:
            torch._dynamo.reset()
            if mode == "eager":
                fn = add_rmsnorm
            elif mode == "inductor":
                fn = torch.compile(add_rmsnorm)
            else:
                fn = torch.compile(add_rmsnorm, mode=mode)
            for rows in rows_list:
                r, h, w = make_inputs(rows, 1024, dtype)
                torch.cuda.synchronize()
                started = time.perf_counter()
                t_c, y_c = fn(r, h, w)
                torch.cuda.synchronize()
                first_call_ms = (time.perf_counter() - started) * 1000.0
                graphs = counters_summary().get("stats", {}).get("unique_graphs", 0)
                warmup, repeats = (3, 10) if rows <= 2048 else (3, 10)
                latency = OperatorTimer(device, warmup, repeats).measure(
                    lambda: fn(r, h, w)
                )
                eager_latency = eager_by_rows[rows]
                bytes_moved = 4 * rows * 1024 * (2 if dtype == torch.float16 else 4)
                ach = (
                    bytes_moved / (latency["p50_ms"] * 1e-3) / 1e9 / CH04_ANCHORS["copy_peak_gbps"]
                    if rows == 32768
                    else None
                )
                anchor_name = f"fused_{anchor_key}_rows{rows}_ms"
                anchor = CH04_ANCHORS.get(anchor_name)
                entry = {
                    "rows": rows,
                    "dtype": dtype_key,
                    "mode": mode,
                    "first_call_ms": round(first_call_ms, 1),
                    "unique_graphs_cumulative": graphs,
                    "latency": latency,
                    "eager": eager_latency,
                    "speedup_p50_vs_eager": round(
                        eager_latency["p50_ms"] / latency["p50_ms"], 4
                    ),
                    "vs_ch04_fused_anchor_over": (
                        round(latency["p50_ms"] / anchor, 4) if anchor else None
                    ),
                    "dram_effective_gbps": (
                        round(bytes_moved / (latency["p50_ms"] * 1e-3) / 1e9, 1)
                        if rows == 32768
                        else None
                    ),
                    "dram_ach_pct": round(ach * 100, 2) if ach else None,
                }
                if rows == 32768 and dtype == torch.float16:
                    entry["vs_ch14_triton_fused_over"] = round(
                        latency["p50_ms"] / CH14_ANCHORS["triton_fused_fp16_rows32768_ms"], 4
                    )
                if rows == 32768 and dtype == torch.float32:
                    entry["vs_ch14_triton_fused_over"] = round(
                        latency["p50_ms"] / CH14_ANCHORS["triton_fused_fp32_rows32768_ms"], 4
                    )
                payload["entries"].append(entry)
                print(
                    f"[cuda-bench] {rows}x1024 {dtype_key} {mode}: "
                    f"first_call {first_call_ms:.0f} ms, p50 {latency['p50_ms']} ms, "
                    f"speedup {entry['speedup_p50_vs_eager']}"
                    + (f", ach {entry['dram_ach_pct']}%" if ach else ""),
                    flush=True,
                )
                del r, h, w, t_c, y_c
    payload["rows_sequence"] = rows_list
    write_results_json(CHAPTER / "results" / "benchmark-results-cuda.json", payload)


def _triton_version() -> str | None:
    try:
        import triton

        return triton.__version__
    except ImportError:
        return None


if __name__ == "__main__":
    main()
