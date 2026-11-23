#!/usr/bin/env python3
"""第 01 期三后端 RMSNorm Baseline 性能测量。

口径（全篇统一）：
- 被测对象：RMSNorm 组合表达式（第 03 期起被自定义 Kernel 替换的对照基线）；
- 形状：rows ∈ {1, 4, 128, 512, 2048} × H = 1024（合同 benchmark_shapes）；
- 计时：OperatorTimer（Warm-up + 事件/同步边界，报告 P50/P95）；
- 有效带宽 = 理论字节数 / P50；理论字节 = (rows*H 读 x + rows*H 写 y + H 读 w) × dtype 字节数；
- 达成率 = 有效带宽 / 平台规格峰值带宽（cuda: 1008 GB/s, musa: 768 GB/s，来自大纲 §3.3 规格表）；
- 可选 --with-model：附测手写 Qwen3 FP16 的 Prefill / Decode 单步时间与峰值显存。

用法示例：
    python benchmark.py --backend cuda --with-model --model-dir models/qwen3-0.6b
    python benchmark.py --backend musa
    python benchmark.py --backend cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiment_utils import (
    OperatorTimer,
    backend_sync,
    device_environment_record,
    measure_with_graph,
    resolve_device,
    write_results_json,
)
from rmsnorm_reference import RMS_NORM_EPS, composite_rmsnorm_torch

HIDDEN_SIZE = 1024
ROW_COUNTS = [1, 4, 128, 512, 2048]
PEAK_BANDWIDTH_BYTES_PER_SEC = {"cuda": 1008e9, "musa": 768e9, "cpu": None}
DEFAULT_PROMPT = "用一句话解释 RMSNorm 的作用。"


def theoretical_bytes(rows: int, hidden_size: int, itemsize: int) -> int:
    return (rows * hidden_size * 2 + hidden_size) * itemsize


def rmsnorm_chain(arguments, device: torch.device, dtype: torch.dtype) -> dict:
    itemsize = torch.tensor([], dtype=dtype).element_size()
    weight = torch.randn(HIDDEN_SIZE, device=device, dtype=dtype)
    results = []
    for rows in ROW_COUNTS:
        seed = 20260101 + rows * 100000 + HIDDEN_SIZE  # 与 fixtures 同一 seed 规则
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(rows, HIDDEN_SIZE, generator=generator).to(device=device, dtype=dtype)

        repeats = 200 if rows <= 128 else (100 if rows == 512 else 50)
        timer = OperatorTimer(device, warmup=10, repeats=repeats)
        operation = lambda: composite_rmsnorm_torch(x, weight, RMS_NORM_EPS)  # noqa: E731
        timing = timer.measure(operation)
        timing["mode"] = "eager"

        total_bytes = theoretical_bytes(rows, HIDDEN_SIZE, itemsize)
        entry = {
            "rows": rows,
            "hidden_size": HIDDEN_SIZE,
            "dtype": str(dtype).replace("torch.", ""),
            "theoretical_bytes": total_bytes,
        }
        entry.update(timing)

        # Graph Replay 口径：eager 链路在小算子时被 CPU 发射开销主导
        # （launch-bound），Graph 回放测得的才是纯 GPU 执行时间。
        # 达成率取两种口径中更优者并显式标注 mode——MUSA 实测 Graph 回放
        # 可能慢于 eager（平台差异，第 10 期深挖），不能默认选 Graph。
        graph_timing = measure_with_graph(device, warmup=3, repeats=repeats, operation=operation)
        entry["graph_replay"] = graph_timing
        effective_source = timing
        if graph_timing and graph_timing["p50_ms"] < timing["p50_ms"]:
            effective_source = graph_timing

        if PEAK_BANDWIDTH_BYTES_PER_SEC[device.type] is not None:
            effective = total_bytes / (effective_source["p50_ms"] / 1000.0)
            entry["effective_bandwidth_gb_s"] = round(effective / 1e9, 2)
            entry["peak_bandwidth_gb_s_spec"] = PEAK_BANDWIDTH_BYTES_PER_SEC[device.type] / 1e9
            entry["achievement_ratio"] = round(
                effective / PEAK_BANDWIDTH_BYTES_PER_SEC[device.type], 4
            )
            entry["bandwidth_mode"] = effective_source["mode"]
        results.append(entry)
        print(
            f"[rmsnorm/{arguments.backend}] rows={rows:<5} "
            f"p50={timing['p50_ms']:.3f}ms p95={timing['p95_ms']:.3f}ms "
            + (
                f"有效带宽={entry['effective_bandwidth_gb_s']}GB/s 达成率={entry['achievement_ratio']:.2%}"
                if "achievement_ratio" in entry
                else "(CPU 不计带宽)"
            )
        )
    return {"dtype": str(dtype).replace("torch.", ""), "entries": results}


def model_chain(arguments, device: torch.device) -> dict | None:
    if not arguments.with_model:
        return None
    from qwen3_model import load_handwritten_model, resolve_model_directory
    from qwen3_tokenizer import Qwen3Tokenizer

    model_directory = resolve_model_directory(arguments.model_dir)
    tokenizer = Qwen3Tokenizer(model_directory)
    model = load_handwritten_model(model_directory, device, dtype=torch.float16).eval()

    prompt_ids = tokenizer.encode_chat_prompt(DEFAULT_PROMPT)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Prefill：一次整段前向。
    timer = OperatorTimer(device, warmup=3, repeats=20)
    prefill = timer.measure(lambda: model(input_ids))

    # Decode：Greedy 32 步的逐步平均（每步重算整段序列，第 01 期无 KV Cache 边界）。
    decode_steps = []
    with torch.no_grad():
        generated = input_ids
        for _ in range(arguments.max_new_tokens):
            timer_step = OperatorTimer(device, warmup=0, repeats=1)
            timing = timer_step.measure(lambda: model(generated))
            decode_steps.append(timing["p50_ms"])
            next_token_id = model(generated)[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token_id), dim=1)

    peak_memory_bytes = None
    memory_api = getattr(torch, device.type, None)
    if memory_api is not None and hasattr(memory_api, "max_memory_allocated"):
        peak_memory_bytes = memory_api.max_memory_allocated()

    result = {
        "model_dtype": "float16",
        "prompt_token_count": len(prompt_ids),
        "max_new_tokens": arguments.max_new_tokens,
        "prefill": prefill,
        "decode_avg_ms": round(sum(decode_steps) / len(decode_steps), 4),
        "decode_p50_of_steps_ms": round(sorted(decode_steps)[len(decode_steps) // 2], 4),
        "peak_memory_allocated_bytes": peak_memory_bytes,
        "note": "第 01 期无 KV Cache，每步重算整段序列；数值仅作三后端同口径对照",
    }
    print(
        f"[model/{arguments.backend}] prefill p50={prefill['p50_ms']:.1f}ms "
        f"decode 平均={result['decode_avg_ms']:.1f}ms/步 "
        f"峰值显存={None if peak_memory_bytes is None else round(peak_memory_bytes / 2**30, 2)}GiB"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cpu", "cuda", "musa"])
    parser.add_argument("--dtype", default=None, choices=["fp32", "fp16"], help="默认：CPU 用 fp32，GPU 用 fp16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--model-dir", default="models/qwen3-0.6b")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    arguments = parser.parse_args()

    if arguments.dtype is None:
        arguments.dtype = "fp32" if arguments.backend == "cpu" else "fp16"
    dtype = torch.float16 if arguments.dtype == "fp16" else torch.float32

    device = resolve_device(arguments.backend)
    if device.type != "cpu":
        memory_api = getattr(torch, device.type)
        memory_api.empty_cache()
        if hasattr(memory_api, "reset_peak_memory_stats"):
            memory_api.reset_peak_memory_stats()

    results = {
        "schema_version": "kernel-chapter01.benchmark.v1",
        "backend": arguments.backend,
        "rmsnorm_baseline": rmsnorm_chain(arguments, device, dtype),
        "model": model_chain(arguments, device),
        "environment": device_environment_record(device),
    }
    output = Path("results") / f"benchmark-results-{arguments.backend}.json"
    write_results_json(output, results)
    backend_sync(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
