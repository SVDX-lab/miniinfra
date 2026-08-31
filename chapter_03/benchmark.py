"""固定 Batch 的吞吐、Padding、槽位利用率与显存实验。"""

import argparse
import json
from pathlib import Path

import torch

from batch_generation import generate_fixed_batch, generate_serial, percentile
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="固定批处理受控实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16]
    )
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--suite", choices=("throughput", "padding", "slots", "all"),
        default="all",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("serial", "batch"),
        default=["serial", "batch"],
        help="吞吐实验运行的路径；大 Batch 容量测试可只选 batch",
    )
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args()


def make_sequence(length, vocab_size, salt):
    values = torch.arange(length, dtype=torch.long)
    return ((values * (7919 + salt) + 17 + salt) % (vocab_size - 1) + 1).tolist()


def make_sequences(lengths, vocab_size):
    return [
        make_sequence(length, vocab_size, 23 * (index + 1))
        for index, length in enumerate(lengths)
    ]


def mean(values):
    return sum(values) / len(values)


def aggregate_throughput(mode, batch_size, runs):
    metrics = [run["metrics"] for run in runs]
    row = {
        "mode": mode,
        "batch_size": batch_size,
        "end_to_end_ms_mean": mean([item["end_to_end_ms"] for item in metrics]),
        "request_throughput_per_second_mean": mean(
            [item["request_throughput_per_second"] for item in metrics]
        ),
        "output_token_throughput_per_second_mean": mean(
            [item["output_token_throughput_per_second"] for item in metrics]
        ),
        "peak_memory_mib_max": max(
            item["peak_memory_bytes"] for item in metrics
        ) / 1024**2,
    }
    if mode == "batch":
        decode_samples = [
            value for item in metrics for value in item["decode_times_ms"]
        ]
        row.update(
            {
                "prefill_ms_mean": mean([item["prefill_ms"] for item in metrics]),
                "decode_iteration_ms_mean": mean(decode_samples),
                "decode_iteration_ms_p50": percentile(decode_samples, 50),
                "decode_iteration_ms_p95": percentile(decode_samples, 95),
                "cache_mib": metrics[0]["cache_bytes"] / 1024**2,
            }
        )
    return row


def run_throughput(args, model, device):
    rows = []
    for batch_size in args.batch_sizes:
        sequences = make_sequences(
            [args.prompt_length] * batch_size, model.config.vocab_size
        )
        for mode in args.modes:
            function = generate_serial if mode == "serial" else generate_fixed_batch
            for _ in range(args.warmup):
                function(
                    model, sequences, args.max_new_tokens, -1, device,
                    stop_on_eos=False,
                )
            runs = [
                function(
                    model, sequences, args.max_new_tokens, -1, device,
                    stop_on_eos=False,
                )
                for _ in range(args.repeats)
            ]
            row = aggregate_throughput(mode, batch_size, runs)
            rows.append(row)
            print(
                "throughput,%s,batch=%d,e2e=%.3fms,requests=%.3f/s,tokens=%.3f/s,peak=%.3fMiB"
                % (
                    mode,
                    batch_size,
                    row["end_to_end_ms_mean"],
                    row["request_throughput_per_second_mean"],
                    row["output_token_throughput_per_second_mean"],
                    row["peak_memory_mib_max"],
                )
            )
    return rows


def aggregate_batch_case(name, lengths, budgets, runs):
    metrics = [run["metrics"] for run in runs]
    decode_samples = [
        value for item in metrics for value in item["decode_times_ms"]
    ]
    return {
        "case": name,
        "prompt_lengths": lengths,
        "output_budgets": budgets,
        "padding_efficiency": metrics[0]["padding_efficiency"],
        "effective_prompt_token_throughput_per_second_mean": mean(
            [
                item["effective_prompt_token_throughput_per_second"]
                for item in metrics
            ]
        ),
        "padded_prompt_token_throughput_per_second_mean": mean(
            [
                item["padded_prompt_token_throughput_per_second"]
                for item in metrics
            ]
        ),
        "slot_utilization": metrics[0]["slot_utilization"],
        "active_counts": metrics[0]["active_counts"],
        "completion_steps": metrics[0]["completion_steps"],
        "prefill_ms_mean": mean([item["prefill_ms"] for item in metrics]),
        "decode_iteration_ms_mean": mean(decode_samples),
        "end_to_end_ms_mean": mean([item["end_to_end_ms"] for item in metrics]),
        "output_token_throughput_per_second_mean": mean(
            [item["output_token_throughput_per_second"] for item in metrics]
        ),
        "peak_memory_mib_max": max(
            item["peak_memory_bytes"] for item in metrics
        ) / 1024**2,
        "cache_mib": metrics[0]["cache_bytes"] / 1024**2,
    }


def run_batch_case(args, model, device, name, lengths, budgets):
    sequences = make_sequences(lengths, model.config.vocab_size)
    for _ in range(args.warmup):
        generate_fixed_batch(
            model, sequences, budgets, -1, device, stop_on_eos=False
        )
    runs = [
        generate_fixed_batch(
            model, sequences, budgets, -1, device, stop_on_eos=False
        )
        for _ in range(args.repeats)
    ]
    row = aggregate_batch_case(name, lengths, budgets, runs)
    print(
        "%s,padding_efficiency=%.2f%%,slot_utilization=%.2f%%,"
        "effective_prompt_tokens=%.3f/s,e2e=%.3fms,tokens=%.3f/s,peak=%.3fMiB"
        % (
            name,
            row["padding_efficiency"] * 100,
            row["slot_utilization"] * 100,
            row["effective_prompt_token_throughput_per_second_mean"],
            row["end_to_end_ms_mean"],
            row["output_token_throughput_per_second_mean"],
            row["peak_memory_mib_max"],
        )
    )
    return row


def run_padding(args, model, device):
    batch_size = 4
    output_budgets = [args.max_new_tokens] * batch_size
    homogeneous = [args.prompt_length] * batch_size
    mixed = [
        max(1, args.prompt_length // 8),
        max(1, args.prompt_length // 4),
        max(1, args.prompt_length // 2),
        args.prompt_length,
    ]
    return [
        run_batch_case(
            args, model, device, "padding_homogeneous", homogeneous, output_budgets
        ),
        run_batch_case(
            args, model, device, "padding_mixed", mixed, output_budgets
        ),
    ]


def run_slots(args, model, device):
    budgets = [8, 16, 32, 64]
    lengths = [args.prompt_length] * len(budgets)
    return [
        run_batch_case(args, model, device, "slots_mixed_budgets", lengths, budgets)
    ]


def main():
    args = parse_args()
    if (
        any(value < 1 for value in args.batch_sizes)
        or args.prompt_length < 1
        or args.max_new_tokens < 2
        or args.warmup < 0
        or args.repeats < 1
    ):
        raise ValueError("Batch、长度和 repeats 必须为正，warmup 不能小于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")
    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    model = load_handwritten_model(model_directory, device)
    results = {}
    if args.suite in ("throughput", "all"):
        results["throughput"] = run_throughput(args, model, device)
    if args.suite in ("padding", "all"):
        results["padding"] = run_padding(args, model, device)
    if args.suite in ("slots", "all"):
        results["slots"] = run_slots(args, model, device)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        environment = {
            "gpu": torch.cuda.get_device_name(device),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "model": args.model,
            "revision": args.revision,
            "dtype": str(next(model.parameters()).dtype),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "prompt_length": args.prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "input": "synthetic fixed token IDs",
            "tokenizer_included": False,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"environment": environment, "results": results},
                file, ensure_ascii=False, indent=2,
            )
        print("JSON 结果已写入:", output_path)


if __name__ == "__main__":
    main()
