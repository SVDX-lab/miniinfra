"""Eager 与手写 Triton FlashAttention 的独立进程长序列微基准。"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from flash_attention import flash_attention_forward
from qwen3_model import eager_attention_forward


def parse_args():
    parser = argparse.ArgumentParser(description="FlashAttention 长序列微基准")
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 4096, 8192])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", help="可选 JSON 输出路径")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("eager", "flash"), help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def execute(backend, length, warmup, repeats):
    torch.manual_seed(9)
    device = torch.device("cuda")
    shape = (1, 16, length, 128)
    query = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.25
    key = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.25
    value = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.25
    valid = torch.ones((1, length), dtype=torch.bool, device=device)
    operation = eager_attention_forward if backend == "eager" else flash_attention_forward
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                output = operation(query, key, value, valid, valid, 0)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_bytes = torch.cuda.memory_allocated(device)
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            for start, end in zip(starts, ends):
                start.record()
                output = operation(query, key, value, valid, valid, 0)
                end.record()
            torch.cuda.synchronize()
            samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
            peak_bytes = torch.cuda.max_memory_allocated(device)
            del output
        ordered = sorted(samples)
        return {
            "backend": backend,
            "length": length,
            "status": "ok",
            "latency_ms_mean": sum(samples) / len(samples),
            "latency_ms_p50": ordered[len(ordered) // 2],
            "tokens_per_second": length / (sum(samples) / len(samples) / 1000),
            "incremental_peak_memory_bytes": max(0, peak_bytes - baseline_bytes),
        }
    except torch.OutOfMemoryError as error:
        return {
            "backend": backend,
            "length": length,
            "status": "oom",
            "error": str(error).splitlines()[0],
        }


def system_environment():
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = "unknown"
    return {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_gib": properties.total_memory / 1024**3,
        "compute_capability": "%d.%d" % (properties.major, properties.minor),
        "operating_system": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "triton": __import__("triton").__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_driver": driver,
    }
def worker(args):
    if not torch.cuda.is_available():
        raise RuntimeError("微基准需要 NVIDIA GPU")
    print(json.dumps(execute(args.backend, args.length, args.warmup, args.repeats)))


def parent(args):
    rows = []
    script = str(Path(__file__).resolve())
    for length in args.lengths:
        for backend in ("eager", "flash"):
            command = [
                sys.executable, script, "--worker", "--backend", backend,
                "--length", str(length), "--warmup", str(args.warmup),
                "--repeats", str(args.repeats),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                row = {
                    "backend": backend,
                    "length": length,
                    "status": "error",
                    "error": completed.stderr.strip().splitlines()[-1],
                }
            else:
                row = json.loads(completed.stdout.strip().splitlines()[-1])
            rows.append(row)
            if row["status"] == "ok":
                print(
                    "%s L=%d: %.3f ms, %.1f token/s, peak +%.1f MiB"
                    % (
                        backend, length, row["latency_ms_mean"],
                        row["tokens_per_second"],
                        row["incremental_peak_memory_bytes"] / 1024**2,
                    )
                )
            else:
                print("%s L=%d: %s" % (backend, length, row["status"]))
    comparisons = []
    for length in args.lengths:
        eager = next(row for row in rows if row["length"] == length and row["backend"] == "eager")
        flash = next(row for row in rows if row["length"] == length and row["backend"] == "flash")
        comparison = {"length": length}
        if eager["status"] == flash["status"] == "ok":
            comparison["speedup"] = eager["latency_ms_mean"] / flash["latency_ms_mean"]
            comparison["peak_memory_reduction"] = 1 - (
                flash["incremental_peak_memory_bytes"]
                / eager["incremental_peak_memory_bytes"]
            ) if eager["incremental_peak_memory_bytes"] else None
            print("L=%d speedup: %.2fx" % (length, comparison["speedup"]))
        comparisons.append(comparison)
    payload = {
        "environment": system_environment(),
        "configuration": {
            "batch": 1, "heads": 16, "head_dim": 128, "dtype": "bfloat16",
            "causal": True, "warmup": args.warmup, "repeats": args.repeats,
        },
        "results": rows,
        "comparisons": comparisons,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    if args.worker:
        worker(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
