"""使用固定 Token 长度测量 no-cache 与 KV Cache 的延迟和显存。"""

import argparse
import json
from pathlib import Path

import torch

from generation import generate, percentile, synchronize
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="KV Cache 固定长度受控实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[16, 64, 256, 512, 1024],
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", help="可选：将完整结果写入 JSON 文件")
    return parser.parse_args()


def make_fixed_input(prompt_length, vocab_size, device):
    """构造内容确定、长度精确的合成 Token 序列。"""

    token_ids = torch.arange(prompt_length, dtype=torch.long, device=device)
    token_ids = (token_ids * 7919 + 17) % vocab_size
    return token_ids.unsqueeze(0)


def average(values):
    return sum(values) / len(values)


def aggregate(mode, prompt_length, runs):
    metrics = [run["metrics"] for run in runs]
    decode_samples = [
        value for metric in metrics for value in metric["decode_times_ms"]
    ]
    return {
        "mode": mode,
        "prompt_tokens": prompt_length,
        "generated_tokens": len(runs[0]["new_token_ids"]),
        "prefill_ms_mean": average([item["prefill_ms"] for item in metrics]),
        "decode_ms_mean": average(decode_samples) if decode_samples else 0.0,
        "decode_ms_p50": percentile(decode_samples, 50),
        "decode_ms_p95": percentile(decode_samples, 95),
        "end_to_end_ms_mean": average(
            [item["end_to_end_ms"] for item in metrics]
        ),
        "memory_after_prefill_mib_max": max(
            item["memory_after_prefill_bytes"] for item in metrics
        )
        / 1024**2,
        "peak_memory_mib_max": max(item["peak_memory_bytes"] for item in metrics)
        / 1024**2,
        "cache_mib": metrics[0]["cache_bytes"] / 1024**2,
        "decode_samples_ms": decode_samples,
    }


def print_table(rows):
    print(
        "mode,prompt_tokens,generated_tokens,prefill_ms_mean,decode_ms_mean,"
        "decode_ms_p50,decode_ms_p95,end_to_end_ms_mean,"
        "memory_after_prefill_mib_max,peak_memory_mib_max,cache_mib"
    )
    for row in rows:
        print(
            "{mode},{prompt_tokens},{generated_tokens},{prefill_ms_mean:.3f},"
            "{decode_ms_mean:.3f},{decode_ms_p50:.3f},{decode_ms_p95:.3f},"
            "{end_to_end_ms_mean:.3f},{memory_after_prefill_mib_max:.3f},"
            "{peak_memory_mib_max:.3f},{cache_mib:.3f}".format(**row)
        )


def main():
    args = parse_args()
    if any(length < 1 for length in args.prompt_lengths):
        raise ValueError("所有 Prompt 长度必须大于 0")
    if args.max_new_tokens < 2:
        raise ValueError("--max-new-tokens 至少为 2，才能测量 Decode")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup 不能小于 0，repeats 必须大于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    model = load_handwritten_model(model_directory, device)
    rows = []

    for prompt_length in args.prompt_lengths:
        input_ids = make_fixed_input(prompt_length, model.config.vocab_size, device)
        for mode in ("no-cache", "kv-cache"):
            for _ in range(args.warmup):
                generate(
                    mode,
                    model,
                    input_ids,
                    args.max_new_tokens,
                    eos_token_id=-1,
                    device=device,
                    stop_on_eos=False,
                )
            synchronize(device)

            runs = []
            for _ in range(args.repeats):
                runs.append(
                    generate(
                        mode,
                        model,
                        input_ids,
                        args.max_new_tokens,
                        eos_token_id=-1,
                        device=device,
                        stop_on_eos=False,
                    )
                )
            rows.append(aggregate(mode, prompt_length, runs))

    print_table(rows)
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
            "max_new_tokens": args.max_new_tokens,
            "input": "synthetic fixed token IDs",
            "tokenizer_included": False,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"environment": environment, "results": rows},
                file,
                ensure_ascii=False,
                indent=2,
            )
        print("JSON 结果已写入:", output_path)


if __name__ == "__main__":
    main()
