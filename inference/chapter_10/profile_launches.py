"""用 PyTorch Profiler 观察 Eager 提交与 CUDA Graph Replay 的启动事件。"""

import argparse

import torch

from experiment_utils import (
    environment_snapshot,
    load_model,
    set_seed,
    synthetic_prompts,
    write_json,
)
from qwen3_model import DEFAULT_MODEL_ID
from static_decode import StaticDecodeRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--capacity", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--context-bucket", type=int, default=160)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("eager", "flash"), default="flash")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="profile-launches-results.json")
    return parser.parse_args()


def profile_one_step(runner, prompts, mode):
    runner.prepare_prompts(prompts)
    torch.cuda.synchronize(runner.device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
    ) as profiler:
        runner.step(mode)
        torch.cuda.synchronize(runner.device)

    averages = profiler.key_averages()
    cuda_runtime = []
    for event in averages:
        lowered = event.key.lower()
        if "cuda" in lowered and (
            "launch" in lowered or "graph" in lowered or "memcpy" in lowered
        ):
            cuda_runtime.append({
                "name": event.key,
                "count": event.count,
                "self_cpu_time_us": event.self_cpu_time_total,
                "self_device_time_us": getattr(
                    event, "self_device_time_total", 0.0
                ),
            })
    top_cpu = sorted(
        (
            {
                "name": event.key,
                "count": event.count,
                "self_cpu_time_us": event.self_cpu_time_total,
            }
            for event in averages
        ),
        key=lambda item: item["self_cpu_time_us"],
        reverse=True,
    )[:20]
    raw_cuda_events = sum(
        1 for event in profiler.events() if str(event.device_type).endswith("CUDA")
    )
    return {
        "mode": mode,
        "cuda_runtime_events": cuda_runtime,
        "raw_cuda_activity_event_count": raw_cuda_events,
        "top_cpu_events": top_cpu,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Profiler 实验需要 NVIDIA CUDA GPU")
    if args.prompt_length >= args.context_bucket:
        raise ValueError("context_bucket 必须大于 prompt_length")
    set_seed(args.seed)
    model_directory, model = load_model(
        args.model, "cuda", args.dtype, args.attention_backend
    )
    runner = StaticDecodeRunner(
        model,
        args.capacity,
        args.context_bucket,
        args.block_size,
    )
    runner.capture()
    prompts = synthetic_prompts(
        args.capacity, args.prompt_length, model.config.vocab_size, args.seed
    )
    results = [
        profile_one_step(runner, prompts, mode)
        for mode in StaticDecodeRunner.MODES
    ]
    payload = {
        "environment": environment_snapshot(
            model_directory, args.dtype, args.attention_backend
        ),
        "config": vars(args),
        "results": results,
        "note": (
            "Profiler 本身会扰动延迟；本实验只用于观察提交事件数量，"
            "性能数值以 benchmark_cuda_graph.py 为准。"
        ),
    }
    write_json(args.output, payload)
    for result in results:
        print(result["mode"])
        for event in result["cuda_runtime_events"]:
            print("  %s count=%d" % (event["name"], event["count"]))
    print("结果已写入", args.output)


if __name__ == "__main__":
    main()
