"""受控测量逐块 KV 传输与独立 GPU 计算的潜在重叠。"""

import argparse
import statistics

import torch

from experiment_utils import collect_environment, save_results, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="KV 传输/计算重叠微基准")
    parser.add_argument("--blocks", default="16,64,128")
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--compute-repeats", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def mean(values):
    return statistics.mean(values)


def summarize(rows):
    keys = rows[0]
    return {
        key + "_mean": mean([row[key] for row in rows])
        for key in keys
    }


def run_case(block_count, direction, matrix_size, compute_repeats):
    device = torch.device("cuda")
    # Qwen3-0.6B，Block Size 16：每块 28×K/V×8×16×128×2B = 1.75 MiB。
    gpu_blocks = torch.randn(
        (block_count, 28, 2, 8, 16, 128),
        dtype=torch.bfloat16,
        device=device,
    )
    cpu_blocks = torch.empty(
        gpu_blocks.shape, dtype=gpu_blocks.dtype, device="cpu", pin_memory=True
    )
    if direction == "h2d":
        cpu_blocks.copy_(gpu_blocks)
    left = torch.randn(
        (matrix_size, matrix_size), dtype=torch.bfloat16, device=device
    )
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    default = torch.cuda.current_stream(device)
    copy_stream = torch.cuda.Stream(device=device)

    def copies(stream):
        with torch.cuda.stream(stream):
            for index in range(block_count):
                if direction == "d2h":
                    cpu_blocks[index].copy_(gpu_blocks[index], non_blocking=True)
                else:
                    gpu_blocks[index].copy_(cpu_blocks[index], non_blocking=True)

    def compute():
        for _ in range(compute_repeats):
            torch.mm(left, right, out=output)

    torch.cuda.synchronize(device)

    copy_start = torch.cuda.Event(enable_timing=True)
    copy_end = torch.cuda.Event(enable_timing=True)
    copy_start.record(copy_stream)
    copies(copy_stream)
    copy_end.record(copy_stream)
    copy_end.synchronize()
    copy_alone_ms = copy_start.elapsed_time(copy_end)

    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)
    compute_start.record(default)
    compute()
    compute_end.record(default)
    compute_end.synchronize()
    compute_alone_ms = compute_start.elapsed_time(compute_end)

    serial_start = torch.cuda.Event(enable_timing=True)
    serial_end = torch.cuda.Event(enable_timing=True)
    serial_start.record(default)
    copies(default)
    compute()
    serial_end.record(default)
    serial_end.synchronize()
    serial_ms = serial_start.elapsed_time(serial_end)

    start = torch.cuda.Event(enable_timing=True)
    copy_concurrent_start = torch.cuda.Event(enable_timing=True)
    copy_concurrent_end = torch.cuda.Event(enable_timing=True)
    compute_concurrent_start = torch.cuda.Event(enable_timing=True)
    compute_concurrent_end = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(default)
    copy_stream.wait_event(start)
    copy_concurrent_start.record(copy_stream)
    copies(copy_stream)
    copy_concurrent_end.record(copy_stream)
    compute_concurrent_start.record(default)
    compute()
    compute_concurrent_end.record(default)
    default.wait_event(copy_concurrent_end)
    end.record(default)
    end.synchronize()
    concurrent_ms = start.elapsed_time(end)
    copy_during_ms = copy_concurrent_start.elapsed_time(copy_concurrent_end)
    compute_during_ms = compute_concurrent_start.elapsed_time(compute_concurrent_end)
    overlap_numerator = copy_alone_ms + compute_alone_ms - concurrent_ms
    overlap_efficiency = overlap_numerator / min(copy_alone_ms, compute_alone_ms)
    return {
        "copy_alone_ms": copy_alone_ms,
        "compute_alone_ms": compute_alone_ms,
        "serial_ms": serial_ms,
        "concurrent_ms": concurrent_ms,
        "copy_during_concurrency_ms": copy_during_ms,
        "compute_during_concurrency_ms": compute_during_ms,
        "overlap_efficiency": overlap_efficiency,
        "concurrent_vs_serial_speedup": serial_ms / concurrent_ms,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("重叠微基准需要 NVIDIA GPU")
    seed_everything(13)
    block_counts = [int(item) for item in args.blocks.split(",")]
    report = {
        "environment": collect_environment(torch.device("cuda")),
        "config": vars(args),
        "cases": [],
    }
    for block_count in block_counts:
        for direction in ("d2h", "h2d"):
            for _ in range(args.warmup):
                run_case(
                    block_count, direction, args.matrix_size,
                    args.compute_repeats,
                )
            rows = [
                run_case(
                    block_count, direction, args.matrix_size,
                    args.compute_repeats,
                )
                for _ in range(args.repeats)
            ]
            entry = {
                "blocks": block_count,
                "direction": direction,
                "mib": block_count * 1.75,
                **summarize(rows),
                "raw": rows,
            }
            report["cases"].append(entry)
            print(
                "%s %3d blocks: serial %.2f ms, concurrent %.2f ms, "
                "speedup %.3fx, overlap efficiency %.1f%%"
                % (
                    direction.upper(), block_count,
                    entry["serial_ms_mean"], entry["concurrent_ms_mean"],
                    entry["concurrent_vs_serial_speedup_mean"],
                    entry["overlap_efficiency_mean"] * 100,
                )
            )
    if args.output:
        save_results(args.output, report)


if __name__ == "__main__":
    main()
