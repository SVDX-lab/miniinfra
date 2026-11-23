"""同步 KV 搬运对固定 Decode batch 的增量停顿微基准。

主压力实验中的 ITL 同时受暂停、驻留集合和 batch 大小影响，不能直接归因于 PCIe。
本脚本固定两个 running 请求及其 slot，不执行抢占调度，只在相邻 Decode 之间对
第二个请求的全部物理 KV Block 做一次同步 D2H+H2D round-trip。与紧邻的无拷贝
Decode 对照后，报告 event-aligned 间隔增量。

该实验隔离的是同步传输停顿，不代表真实换出带来的容量收益或暂停时长。
"""

import argparse
import statistics
import time

import torch

from engine import _timed_decode, _timed_prefill, synchronize
from experiment_utils import (
    collect_environment,
    save_results,
    seed_everything,
    synthesize_workload,
)
from paged_cache import CPUPinnedPool, PagedKVCache
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3Config,
    load_handwritten_model,
    resolve_model_directory,
)
from scheduler import PrefillPlan, RequestState, make_request_specs


def parse_args():
    parser = argparse.ArgumentParser(description="固定 Decode batch 的同步拷贝停顿")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--output", default="chapter_12/benchmark-copy-stall-results.json"
    )
    return parser.parse_args()


def summarize(values):
    return {
        "mean_ms": sum(values) / len(values),
        "stddev_ms": statistics.pstdev(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("同步拷贝停顿微基准需要可用的 NVIDIA GPU")
    if args.context_tokens < 1:
        raise ValueError("context_tokens 必须大于 0")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    seed_everything(args.seed)
    model_directory = resolve_model_directory(args.model, args.revision)
    config = Qwen3Config.from_model_directory(model_directory)
    model = load_handwritten_model(model_directory, device, dtype)
    total_decode_steps = 3 * (args.warmup + args.repeats) + 2
    sequences, _ = synthesize_workload(
        config.vocab_size,
        (args.context_tokens, args.context_tokens),
        total_decode_steps,
        args.seed,
        arrivals=[0.0, 0.0],
    )
    specs = make_request_specs(sequences, total_decode_steps, [0.0, 0.0])
    states = [RequestState(spec) for spec in specs]
    required_per_request = (
        args.context_tokens + total_decode_steps + args.block_size - 1
    ) // args.block_size
    cache = PagedKVCache(
        config, args.block_size, 2 * required_per_request, device, dtype
    )
    for slot, state in enumerate(states):
        state.status = "prefilling"
        state.slot_index = slot
        cache.begin_request(state.spec.request_id)
        cache.reserve_blocks(state.spec.request_id, args.context_tokens)
    plans = [PrefillPlan(state, 0, args.context_tokens) for state in states]
    first_tokens, _, _, _, _, _ = _timed_prefill(
        model, plans, cache, 0, device, False
    )
    for state in states:
        state.prefill_cursor = args.context_tokens
        state.cache_length = args.context_tokens
        state.generated.append(first_tokens[state.spec.request_id])
        state.status = "running"

    victim_id = states[-1].spec.request_id
    cpu_pool = CPUPinnedPool(
        config, args.block_size, required_per_request, dtype, pin_memory=True
    )

    def decode_once():
        result = _timed_decode(model, states, cache, 2, 0, device, False)
        tokens, _, total_ms, _, _, _, _ = result
        for state, token in zip(states, tokens):
            state.generated.append(token)
            state.cache_length += 1
        return total_ms

    def roundtrip_once():
        victim_blocks = list(cache.block_tables[victim_id])
        synchronize(device)
        start = time.perf_counter()
        for cpu_id, gpu_id in enumerate(victim_blocks):
            cpu_pool.blocks[cpu_id].copy_(cache.blocks[gpu_id])
        for cpu_id, gpu_id in enumerate(victim_blocks):
            cache.blocks[gpu_id].copy_(cpu_pool.blocks[cpu_id])
        synchronize(device)
        return (time.perf_counter() - start) * 1000, len(victim_blocks)

    for _ in range(args.warmup):
        decode_once()
        roundtrip_once()
        decode_once()
        decode_once()

    rows = []
    for repeat in range(args.repeats):
        control_before_ms = decode_once()
        copy_ms, copy_blocks = roundtrip_once()
        copy_decode_ms = decode_once()
        control_after_ms = decode_once()
        control_decode_ms = (control_before_ms + control_after_ms) / 2
        event_interval_ms = copy_ms + copy_decode_ms
        rows.append({
            "repeat": repeat,
            "control_before_ms": control_before_ms,
            "control_after_ms": control_after_ms,
            "control_decode_ms": control_decode_ms,
            "copy_roundtrip_ms": copy_ms,
            "copy_blocks": copy_blocks,
            "physical_bytes_each_direction": copy_blocks * cache.bytes_per_block,
            "copy_followed_by_decode_ms": copy_decode_ms,
            "event_aligned_interval_ms": event_interval_ms,
            "estimated_increment_ms": event_interval_ms - control_decode_ms,
        })

    report = {
        "environment": collect_environment(device),
        "configuration": {
            "initial_context_tokens": args.context_tokens,
            "block_size": args.block_size,
            "copy_blocks_min": min(row["copy_blocks"] for row in rows),
            "copy_blocks_max": max(row["copy_blocks"] for row in rows),
            "physical_bytes_each_direction_min": min(
                row["physical_bytes_each_direction"] for row in rows
            ),
            "physical_bytes_each_direction_max": max(
                row["physical_bytes_each_direction"] for row in rows
            ),
            "active_requests": 2,
            "fixed_slots": [0, 1],
            "dtype": args.dtype,
            "warmup_runs": args.warmup,
            "formal_repeats": args.repeats,
        },
        "summary": {
            "control_decode": summarize(
                [row["control_decode_ms"] for row in rows]
            ),
            "copy_roundtrip": summarize(
                [row["copy_roundtrip_ms"] for row in rows]
            ),
            "event_aligned_interval": summarize(
                [row["event_aligned_interval_ms"] for row in rows]
            ),
            "estimated_increment": summarize(
                [row["estimated_increment_ms"] for row in rows]
            ),
        },
        "raw": rows,
    }
    save_results(args.output, report)
    print(
        "固定 batch Decode %.2f ms；D2H+H2D %.2f ms；"
        "event-aligned 增量 %.2f ms"
        % (
            report["summary"]["control_decode"]["mean_ms"],
            report["summary"]["copy_roundtrip"]["mean_ms"],
            report["summary"]["estimated_increment"]["mean_ms"],
        )
    )


if __name__ == "__main__":
    main()
