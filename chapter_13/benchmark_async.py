"""第 13 期容量压力主实验：同步传输 vs 异步传输。"""

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
MODES = ("sync", "async", "relaxed")


def parse_args():
    parser = argparse.ArgumentParser(description="同步/异步 KV 传输容量压力实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--pool-blocks", default="230,320")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-running-requests", type=int, default=6)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--modes", default="sync,async,relaxed")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def print_summary(label, metrics):
    print(
        "%s: makespan %.1f ms, %.1f tok/s, ITL p95 %.1f ms, preempt %.0f, "
        "transfer device/exposed %.1f/%.1f ms"
        % (
            label, metrics["makespan_ms"],
            metrics["output_token_throughput_per_second"],
            metrics["itl_ms_p95"], metrics["preemption_count"],
            metrics.get("transfer_device_ms_total", 0.0),
            metrics.get("transfer_exposed_wait_ms_total", 0.0),
        )
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("主实验需要 NVIDIA GPU")
    device = torch.device("cuda")
    seed_everything(args.seed)
    directory = resolve_model_directory(args.model, args.revision)
    config = Qwen3Config.from_model_directory(directory)
    sequences, arrivals = synthesize_workload(
        config.vocab_size, PROMPT_LENGTHS, args.max_new_tokens, args.seed
    )
    specs = make_request_specs(sequences, args.max_new_tokens, arrivals)
    pool_sizes = [int(item) for item in args.pool_blocks.split(",")]
    modes = [item.strip() for item in args.modes.split(",")]
    if any(mode not in MODES for mode in modes):
        raise ValueError("modes 必须来自 %s" % (MODES,))
    largest = max(
        (length + args.max_new_tokens + args.block_size - 1) // args.block_size
        for length in PROMPT_LENGTHS
    )
    if any(size < largest for size in pool_sizes):
        raise ValueError("GPU 池小于单请求最坏 Block 数 %d" % largest)

    model = load_handwritten_model(directory, device, getattr(torch, args.dtype))
    common = dict(
        max_running_requests=args.max_running_requests,
        eos_token_id=-1,
        device=device,
        token_budget=args.token_budget,
        block_size=args.block_size,
        stop_on_eos=False,
    )
    if args.warmup:
        warm_sequences, warm_arrivals = synthesize_workload(
            config.vocab_size, (128, 128), 8, args.seed
        )
        warm_specs = make_request_specs(warm_sequences, 8, warm_arrivals)
        for _ in range(args.warmup):
            run_engine(
                model, warm_specs, transfer_mode="async", pool_blocks=64,
                **common,
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
            "warmup_runs": args.warmup,
            "formal_repeats": args.repeats,
            "run_order": modes,
        },
        "runs": [],
    }
    metric_keys = (
        "makespan_ms",
        "output_token_throughput_per_second",
        "itl_ms_p50",
        "itl_ms_p95",
        "itl_ms_max",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "end_to_end_ms_p95",
        "preemption_count",
        "pause_ms_p95",
        "pause_ms_max",
        "transfer_device_ms_total",
        "transfer_submit_wall_ms_total",
        "transfer_exposed_wait_ms_total",
        "transfer_non_exposed_ms_total",
        "d2h_device_ms_total",
        "h2d_device_ms_total",
        "d2h_exposed_wait_ms_total",
        "h2d_exposed_wait_ms_total",
        "peak_memory_bytes",
    )
    for pool_blocks in pool_sizes:
        for mode in modes:
            rows = []
            for repeat in range(args.repeats):
                result = run_engine(
                    model,
                    specs,
                    transfer_mode="sync" if mode == "relaxed" else mode,
                    pool_blocks=None if mode == "relaxed" else pool_blocks,
                    **common,
                )
                row = dict(result["metrics"])
                row["repeat"] = repeat
                rows.append(row)
                print_summary(
                    "pool=%s %s #%d"
                    % ("large" if mode == "relaxed" else pool_blocks, mode, repeat),
                    row,
                )
            aggregated = {
                "requested_pool_blocks": pool_blocks,
                "actual_pool_blocks": rows[0]["pool_blocks"],
                "mode": mode,
                "repeats": args.repeats,
                "per_repeat": rows,
            }
            for key in metric_keys:
                values = [row[key] for row in rows]
                aggregated[key] = statistics.mean(values)
                aggregated[key + "_stddev"] = statistics.pstdev(values)
            aggregated["swap_bytes_total"] = (
                statistics.mean(row["swap_out_bytes_total"] for row in rows)
                + statistics.mean(row["swap_in_bytes_total"] for row in rows)
            )
            report["runs"].append(aggregated)
            print_summary(
                "pool=%s %s mean"
                % ("large" if mode == "relaxed" else pool_blocks, mode),
                aggregated,
            )
    if args.output:
        save_results(args.output, report)


if __name__ == "__main__":
    main()
