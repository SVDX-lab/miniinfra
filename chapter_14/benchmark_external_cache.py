"""外部 Load 与重新 Prefill 的长度扫描微基准。"""

import argparse
import statistics

import torch

from compare_external_cache import synthetic_tokens
from engine import run_request
from experiment_utils import (
    add_model_arguments,
    cache_service,
    environment_record,
    load_model,
    parse_dtype,
    write_json,
)


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description="第 14 期 Load vs Prefill 基准")
    add_model_arguments(parser)
    parser.add_argument("--prefix-lengths", default="256,512,1024,2048")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--external-chunk-size", type=int, default=256)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--capacity-mib", type=float, default=1024.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    lengths = [int(value) for value in args.prefix_lengths.split(",")]
    if any(value < args.external_chunk_size for value in lengths):
        raise ValueError("prefix-lengths 必须不小于 external-chunk-size")

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    _, config, model = load_model(args.model, args.revision, device, dtype)
    maximum = max(lengths) + 1
    all_tokens = synthetic_tokens(maximum, config.vocab_size)
    rows = []
    with cache_service(args.capacity_mib) as (client, _):
        for prefix_length in lengths:
            # 每个长度从冷服务开始，Store 成本与淘汰状态不继承上一组。
            client.clear()
            prompt = all_tokens[:prefix_length + 1]
            common = dict(
                model=model, token_ids=prompt, max_new_tokens=1,
                eos_token_id=-1, device=device, model_id=args.model,
                revision=args.revision, block_size=args.block_size,
                external_chunk_size=args.external_chunk_size,
                token_budget=args.token_budget, stop_on_eos=False,
            )
            # 显式 populate，不把首次 Store 成本混入 warm Load。
            populate = run_request(
                mode="external", external_client=client, **common
            )
            for _ in range(args.warmup):
                run_request(mode="recompute", **common)
                run_request(mode="external", external_client=client, **common)
            samples = []
            for repeat in range(args.repeats):
                baseline = run_request(mode="recompute", **common)
                external = run_request(
                    mode="external", external_client=client, **common
                )
                if baseline["new_token_ids"] != external["new_token_ids"]:
                    raise RuntimeError("长度 %d 的输出 Token 不一致" % prefix_length)
                samples.append({
                    "repeat": repeat,
                    "recompute_ttft_ms": baseline["metrics"]["service_ttft_ms"],
                    "recompute_prefill_model_ms": baseline["metrics"]["prefill_model_ms"],
                    "external_ttft_ms": external["metrics"]["service_ttft_ms"],
                    "lookup_ms": external["metrics"]["lookup_ms"],
                    "load_ms": external["metrics"]["load_ms"],
                    "network_load_ms": external["metrics"]["network_load_ms"],
                    "import_h2d_ms": external["metrics"]["import_h2d_ms"],
                    "external_suffix_prefill_ms": external["metrics"]["prefill_model_ms"],
                    "loaded_bytes": external["metrics"]["loaded_bytes"],
                    "hit_tokens": external["metrics"]["hit_tokens"],
                })
            recompute = [item["recompute_ttft_ms"] for item in samples]
            external = [item["external_ttft_ms"] for item in samples]
            row = {
                "prefix_length": prefix_length,
                "prompt_length": prefix_length + 1,
                "populate_store_ms": populate["metrics"]["store_ms"],
                "populate_stored_bytes": populate["metrics"]["stored_bytes"],
                "recompute_ttft_ms": summarize(recompute),
                "external_ttft_ms": summarize(external),
                "median_speedup": statistics.median(recompute) / statistics.median(external),
                "samples": samples,
            }
            rows.append(row)
            print(
                "prefix=%d recompute=%.2fms external=%.2fms speedup=%.2fx"
                % (
                    prefix_length,
                    row["recompute_ttft_ms"]["median"],
                    row["external_ttft_ms"]["median"],
                    row["median_speedup"],
                )
            )
        stats = client.stats()
    result = {
        "environment": environment_record(args.model, args.revision, device, dtype),
        "config": vars(args),
        "rows": rows,
        "server_stats": stats,
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
