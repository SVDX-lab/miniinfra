"""第 12 期容量压力主实验：swap vs recompute vs 保守准入 vs 无压力。

固定负载、固定到达轨迹与固定 GPU Block Pool，只比较容量不足时的处理方式：

- swap：乐观准入 + swap 抢占（KV 换出到 Pinned CPU，换回续算）；
- recompute：同一池与准入，抢占时丢弃 KV，恢复时重新 Prefill；
- conservative：悲观准入（预留 prompt+max_new），绝不抢占；
- relaxed：大池乐观准入，全程无抢占（上界与负对照）。
"""

import argparse
import statistics

import torch

from engine import run_engine
from experiment_utils import (
    collect_environment,
    save_results,
    seed_everything,
    synthesize_workload,
)
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3Config,
    load_handwritten_model,
    resolve_model_directory,
)
from scheduler import make_request_specs

PROMPT_LENGTHS = (1536, 768, 1280, 1024, 1792, 896, 1152, 640)
MODES = ("swap", "recompute", "conservative", "relaxed")


def parse_args():
    parser = argparse.ArgumentParser(description="抢占式换出容量压力主实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--pool-blocks", default="230,320",
                        help="逗号分隔可扫描多个池大小")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-running-requests", type=int, default=6)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--modes", default="swap,recompute,conservative,relaxed")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def itl_overlaps(request_metrics, resource_events):
    """只观察事件发生时确实驻留的共存请求，不把受害者暂停算作拷贝停顿。"""
    gaps = []
    for item in request_metrics:
        times = item["token_times_ms"]
        for left, right in zip(times, times[1:]):
            gaps.append((left, right, item["request_id"]))
    overlaps = []
    for event in resource_events:
        if event["type"] not in ("swap_out", "swap_in"):
            continue
        if event.get("wall_ms", 0.0) <= 0.0:
            continue
        end = event["clock_ms"]
        start = end - event["wall_ms"]
        owner = event["request_id"]
        coresident_ids = set(event.get("coresident_request_ids", ()))
        values = [
            right - left
            for left, right, request_id in gaps
            if request_id != owner
            and request_id in coresident_ids
            and left < end and right > start
        ]
        overlaps.append({
            "type": event["type"],
            "request_id": owner,
            "clock_ms": event["clock_ms"],
            "wall_ms": event["wall_ms"],
            "event_aligned_itl_max_ms": max(values) if values else 0.0,
            "event_aligned_itl_values_ms": values,
            "coresident_request_count": len(coresident_ids),
        })
    return overlaps


def summarize(result):
    metrics = dict(result["metrics"])
    overlaps = itl_overlaps(result["request_metrics"], result["resource_events"])
    event_aligned = [
        item["event_aligned_itl_max_ms"] for item in overlaps
        if item["event_aligned_itl_values_ms"]
    ]
    metrics["event_aligned_itl_max_ms"] = (
        max(event_aligned) if event_aligned else 0.0
    )
    metrics["event_aligned_itl_mean_ms"] = (
        sum(event_aligned) / len(event_aligned) if event_aligned else 0.0
    )
    metrics["event_aligned_copy_windows"] = overlaps
    return metrics


def print_mode_summary(label, metrics):
    print(
        "%-14s makespan %8.1f ms | 输出 %6.1f tok/s | ITL p50 %5.1f p95 %6.1f "
        "max %7.1f | 抢占 %d | 暂停 p95 %7.1f max %7.1f ms" % (
            label,
            metrics["makespan_ms"],
            metrics["output_token_throughput_per_second"],
            metrics["itl_ms_p50"], metrics["itl_ms_p95"], metrics["itl_ms_max"],
            metrics["preemption_count"],
            metrics["pause_ms_p95"], metrics["pause_ms_max"],
        )
    )


def run_mode(model, specs, mode, pool_blocks, common):
    if mode == "conservative":
        return run_engine(
            model, specs, preempt_mode="swap",
            admission_mode="conservative", pool_blocks=pool_blocks, **common,
        )
    if mode == "relaxed":
        return run_engine(
            model, specs, preempt_mode="swap", **common,
        )
    return run_engine(
        model, specs, preempt_mode=mode, pool_blocks=pool_blocks, **common,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("主实验需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    seed_everything(args.seed)
    model_directory = resolve_model_directory(args.model, args.revision)
    config = Qwen3Config.from_model_directory(model_directory)
    sequences, arrivals = synthesize_workload(
        config.vocab_size, PROMPT_LENGTHS, args.max_new_tokens, args.seed,
    )
    specs = make_request_specs(sequences, args.max_new_tokens, arrivals)
    pool_sizes = [int(value) for value in args.pool_blocks.split(",")]
    modes = [mode.strip() for mode in args.modes.split(",")]
    for mode in modes:
        if mode not in MODES:
            raise ValueError("未知模式: " + mode)
    largest = max(
        (length + args.max_new_tokens + args.block_size - 1) // args.block_size
        for length in PROMPT_LENGTHS
    )
    for pool_blocks in pool_sizes:
        if pool_blocks < largest:
            raise ValueError(
                "池 %d 块小于单请求最坏 %d 块，负载无法运行"
                % (pool_blocks, largest)
            )

    model = load_handwritten_model(model_directory, device, dtype)
    common = dict(
        max_running_requests=args.max_running_requests,
        eos_token_id=-1,
        device=device,
        token_budget=args.token_budget,
        block_size=args.block_size,
        stop_on_eos=False,
    )

    # warm-up：小负载触发 CUDA 初始化与 kernel 编译路径。
    if args.warmup > 0:
        warmup_lengths = (128, 128)
        warmup_sequences, warmup_arrivals = synthesize_workload(
            config.vocab_size, warmup_lengths, 8, args.seed,
        )
        warmup_specs = make_request_specs(
            warmup_sequences, 8, warmup_arrivals
        )
        for _ in range(args.warmup):
            run_engine(
                model, warmup_specs, preempt_mode="swap",
                pool_blocks=64, **common,
            )

    report = {
        "environment": collect_environment(device),
        "workload": {
            "prompt_lengths": list(PROMPT_LENGTHS),
            "max_new_tokens": args.max_new_tokens,
            "arrivals_ms": arrivals,
            "seed": args.seed,
            "block_size": args.block_size,
            "token_budget": args.token_budget,
            "max_running_requests": args.max_running_requests,
            "dtype": args.dtype,
            "largest_single_request_blocks": largest,
            "total_prompt_blocks": sum(
                (length + args.block_size - 1) // args.block_size
                for length in PROMPT_LENGTHS
            ),
            "warmup_runs": args.warmup,
            "formal_repeats": args.repeats,
            "run_order": modes,
        },
        "runs": [],
    }

    for pool_blocks in pool_sizes:
        for mode in modes:
            display_pool = "large" if mode == "relaxed" else str(pool_blocks)
            per_repeat = []
            for repeat in range(args.repeats):
                result = run_mode(model, specs, mode, pool_blocks, common)
                metrics = summarize(result)
                metrics["repeat"] = repeat
                per_repeat.append(metrics)
                print_mode_summary(
                    "pool=%s %s#%d" % (display_pool, mode, repeat), metrics,
                )
            def mean(key):
                return sum(item[key] for item in per_repeat) / len(per_repeat)

            def stddev(key):
                return statistics.pstdev(item[key] for item in per_repeat)

            aggregated = {
                "requested_pool_blocks": pool_blocks,
                "pool_blocks": per_repeat[0]["pool_blocks"],
                "mode": mode,
                "repeats": args.repeats,
                "makespan_ms": mean("makespan_ms"),
                "makespan_ms_stddev": stddev("makespan_ms"),
                "output_token_throughput_per_second": mean(
                    "output_token_throughput_per_second"
                ),
                "output_token_throughput_stddev": stddev(
                    "output_token_throughput_per_second"
                ),
                "itl_ms_p50": mean("itl_ms_p50"),
                "itl_ms_p95": mean("itl_ms_p95"),
                "itl_ms_max": mean("itl_ms_max"),
                "ttft_ms_p50": mean("ttft_ms_p50"),
                "ttft_ms_p95": mean("ttft_ms_p95"),
                "end_to_end_ms_p95": mean("end_to_end_ms_p95"),
                "preemption_count": mean("preemption_count"),
                "pause_ms_p95": mean("pause_ms_p95"),
                "pause_ms_max": mean("pause_ms_max"),
                "swap_out_wall_ms_total": mean("swap_out_wall_ms_total"),
                "swap_in_wall_ms_total": mean("swap_in_wall_ms_total"),
                "swap_out_gb_per_second": mean("swap_out_gb_per_second"),
                "swap_in_gb_per_second": mean("swap_in_gb_per_second"),
                "swap_bytes_total": mean("swap_out_bytes_total")
                + mean("swap_in_bytes_total"),
                "swap_logical_bytes_total": mean("swap_out_logical_bytes_total")
                + mean("swap_in_logical_bytes_total"),
                "swap_tail_fragment_bytes_total": mean(
                    "swap_out_tail_fragment_bytes_total"
                ) + mean("swap_in_tail_fragment_bytes_total"),
                "dropped_kv_bytes_total": mean("dropped_kv_bytes_total"),
                "recompute_redo_tokens_total": mean("recompute_redo_tokens_total"),
                "recompute_prefill_wall_ms_total": mean(
                    "recompute_prefill_wall_ms_total"
                ),
                "event_aligned_itl_max_ms": mean("event_aligned_itl_max_ms"),
                "event_aligned_itl_mean_ms": mean("event_aligned_itl_mean_ms"),
                "pool_cache_bytes": per_repeat[0]["pool_cache_bytes"],
                "peak_memory_bytes": mean("peak_memory_bytes"),
                "logical_concurrency_peak": mean("logical_concurrency_peak"),
                "cpu_pool_peak_bytes": (
                    per_repeat[0]["cpu_pool"]["peak_live_bytes"]
                    if "cpu_pool" in per_repeat[0] else 0
                ),
                "per_repeat": per_repeat,
            }
            report["runs"].append(aggregated)
            print_mode_summary(
                "pool=%s %s 均值" % (display_pool, mode), aggregated
            )

    if args.output:
        save_results(args.output, report)


if __name__ == "__main__":
    main()
