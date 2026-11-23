"""受害请求上下文扫描：隔离 swap 与 recompute 的恢复工作量。

每个长度使用两个同长请求和固定的一块增长余量，使两者完成 Prefill 后第一次
跨块 Decode 必然抢占最后接纳的请求。先运行的请求完成并释放 Block 后，受害者
分别通过 H2D 换入或 Chunked Prefill 重算恢复。实验同时保留 D2H，因为系统级
swap 成本必须包含换出和换入。
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


def parse_args():
    parser = argparse.ArgumentParser(description="swap/recompute 恢复成本上下文扫描")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--lengths", default="256,512,1024,2048,4096")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=17)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--output", default="chapter_12/benchmark-recovery-results.json"
    )
    return parser.parse_args()


def mean(items, key):
    return sum(item[key] for item in items) / len(items)


def stddev(items, key):
    return statistics.pstdev(item[key] for item in items)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("恢复成本扫描需要可用的 NVIDIA GPU")
    if args.max_new_tokens <= args.block_size:
        raise ValueError("max_new_tokens 必须大于 block_size，才能稳定触发跨块增长")
    lengths = [int(item) for item in args.lengths.split(",")]
    if any(length < 1 or length % args.block_size for length in lengths):
        raise ValueError("扫描长度必须为正数且是 block_size 的整数倍")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    seed_everything(args.seed)
    model_directory = resolve_model_directory(args.model, args.revision)
    config = Qwen3Config.from_model_directory(model_directory)
    model = load_handwritten_model(model_directory, device, dtype)
    common = {
        "max_running_requests": 2,
        "eos_token_id": -1,
        "device": device,
        "token_budget": args.token_budget,
        "block_size": args.block_size,
        "stop_on_eos": False,
    }

    if args.warmup:
        sequences, arrivals = synthesize_workload(
            config.vocab_size, (128, 128), args.block_size + 1, args.seed,
            arrivals=[0.0, 0.0],
        )
        specs = make_request_specs(sequences, args.block_size + 1, arrivals)
        for _ in range(args.warmup):
            run_engine(model, specs, preempt_mode="swap", pool_blocks=17, **common)

    report = {
        "environment": collect_environment(device),
        "configuration": {
            "lengths": lengths,
            "dtype": args.dtype,
            "block_size": args.block_size,
            "max_new_tokens": args.max_new_tokens,
            "token_budget": args.token_budget,
            "warmup_runs": args.warmup,
            "formal_repeats": args.repeats,
            "seed": args.seed,
        },
        "runs": [],
    }

    for length in lengths:
        sequences, arrivals = synthesize_workload(
            config.vocab_size, (length, length), args.max_new_tokens, args.seed,
            arrivals=[0.0, 0.0],
        )
        specs = make_request_specs(sequences, args.max_new_tokens, arrivals)
        prompt_blocks = length // args.block_size
        pool_blocks = 2 * prompt_blocks + 1
        for mode in ("swap", "recompute"):
            raw = []
            for repeat in range(args.repeats):
                result = run_engine(
                    model, specs, preempt_mode=mode, pool_blocks=pool_blocks,
                    **common,
                )
                metrics = result["metrics"]
                if metrics["preemption_count"] != 1 or metrics["resume_count"] != 1:
                    raise RuntimeError(
                        "受控扫描预期恰好一次抢占和恢复，实际为 %d/%d"
                        % (metrics["preemption_count"], metrics["resume_count"])
                    )
                raw.append({
                    "repeat": repeat,
                    "makespan_ms": metrics["makespan_ms"],
                    "swap_out_wall_ms": metrics["swap_out_wall_ms_total"],
                    "swap_in_wall_ms": metrics["swap_in_wall_ms_total"],
                    "recompute_prefill_wall_ms": metrics[
                        "recompute_prefill_wall_ms_total"
                    ],
                    "recompute_redo_tokens": metrics[
                        "recompute_redo_tokens_total"
                    ],
                    "physical_bytes": metrics["swap_out_bytes_total"],
                    "logical_bytes": metrics["swap_out_logical_bytes_total"],
                    "tail_fragment_bytes": metrics[
                        "swap_out_tail_fragment_bytes_total"
                    ],
                })
            entry = {
                "context_tokens": length,
                "mode": mode,
                "pool_blocks": pool_blocks,
                "repeats": args.repeats,
                "makespan_ms_mean": mean(raw, "makespan_ms"),
                "makespan_ms_stddev": stddev(raw, "makespan_ms"),
                "swap_out_wall_ms_mean": mean(raw, "swap_out_wall_ms"),
                "swap_in_wall_ms_mean": mean(raw, "swap_in_wall_ms"),
                "recompute_prefill_wall_ms_mean": mean(
                    raw, "recompute_prefill_wall_ms"
                ),
                "recompute_redo_tokens": raw[0]["recompute_redo_tokens"],
                "physical_bytes": raw[0]["physical_bytes"],
                "logical_bytes": raw[0]["logical_bytes"],
                "tail_fragment_bytes": raw[0]["tail_fragment_bytes"],
                "raw": raw,
            }
            report["runs"].append(entry)
            print(
                "L=%4d %-9s swap D2H/H2D=%7.2f/%7.2f ms "
                "recompute=%8.2f ms makespan=%8.1f ms"
                % (
                    length, mode, entry["swap_out_wall_ms_mean"],
                    entry["swap_in_wall_ms_mean"],
                    entry["recompute_prefill_wall_ms_mean"],
                    entry["makespan_ms_mean"],
                )
            )

    save_results(args.output, report)


if __name__ == "__main__":
    main()
