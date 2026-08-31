"""第 12 期 PCIe 传输微基准。

测量内容：
- 块数扫描：逐块同步拷贝的时间线性度与每块固定开销；
- pinned 与 pageable 的 D2H/H2D 实测带宽；
- 一次 contiguous 大拷贝作为带宽上界参考；
- Pinned 池的预分配成本。
"""

import argparse
import statistics
import time

import torch

from experiment_utils import collect_environment, save_results
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3Config,
    resolve_model_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GPU/CPU KV 块传输微基准")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-blocks", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--output", default="chapter_12/benchmark-transfer-results.json"
    )
    return parser.parse_args()


def block_shape(config, block_size):
    return (
        config.num_hidden_layers, 2, config.num_key_value_heads,
        block_size, config.head_dim,
    )


def timed_loop_copy(dst_blocks, src_blocks, count, device, repeats):
    """逐块同步拷贝 count 个块，CUDA event 计时，返回每次毫秒列表。"""
    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        for index in range(count):
            dst_blocks[index].copy_(src_blocks[index])
        end.record()
        torch.cuda.synchronize(device)
        timings.append(start.elapsed_time(end))
    return timings


def timed_flat_copy(dst, src, elements, device, repeats):
    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        dst[:elements].copy_(src[:elements])
        end.record()
        torch.cuda.synchronize(device)
        timings.append(start.elapsed_time(end))
    return timings


def stats(values):
    ordered = sorted(values)
    return {
        "mean_ms": sum(ordered) / len(ordered),
        "stddev_ms": statistics.pstdev(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("传输微基准需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    model_directory = resolve_model_directory(args.model, args.revision)
    config = Qwen3Config.from_model_directory(model_directory)
    shape = (args.max_blocks,) + block_shape(config, args.block_size)
    bytes_per_block = (
        torch.empty(shape[1:], dtype=dtype).numel()
        * torch.empty((), dtype=dtype).element_size()
    )

    print("GPU:", torch.cuda.get_device_name(device))
    print("单块字节数: %.2f MiB（%d 层 × %d KV 头 × head_dim %d × block %d）"
          % (bytes_per_block / 2**20, config.num_hidden_layers,
             config.num_key_value_heads, config.head_dim, args.block_size))

    gpu_blocks = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    pinned_blocks = torch.empty(shape, dtype=dtype, pin_memory=True)
    pageable_blocks = torch.empty(shape, dtype=dtype)

    for _ in range(args.warmup):
        pinned_blocks[0].copy_(gpu_blocks[0])
        gpu_blocks[0].copy_(pinned_blocks[0])
    torch.cuda.synchronize(device)

    report = {
        "environment": collect_environment(device),
        "block_size": args.block_size,
        "max_blocks": args.max_blocks,
        "bytes_per_block": bytes_per_block,
        "dtype": args.dtype,
        "warmup_runs": args.warmup,
        "formal_repeats": args.repeats,
        "block_sweep": [],
        "flat_copy": {},
        "pageable": {},
        "pinned_allocation": {},
    }

    print("\n=== 逐块同步拷贝（pinned） ===")
    print("%8s %12s %12s %12s %12s" % (
        "blocks", "MiB", "D2H ms", "H2D ms", "GB/s(D2H)"))
    sweep_counts = [
        count for count in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        if count <= args.max_blocks
    ]
    for count in sweep_counts:
        d2h = stats(timed_loop_copy(
            pinned_blocks, gpu_blocks, count, device, args.repeats
        ))
        h2d = stats(timed_loop_copy(
            gpu_blocks, pinned_blocks, count, device, args.repeats
        ))
        moved = count * bytes_per_block
        d2h_gbps = moved / (d2h["mean_ms"] / 1000) / 1e9
        h2d_gbps = moved / (h2d["mean_ms"] / 1000) / 1e9
        report["block_sweep"].append({
            "blocks": count, "bytes": moved,
            "d2h": d2h, "h2d": h2d,
            "d2h_gb_per_second": d2h_gbps,
            "h2d_gb_per_second": h2d_gbps,
        })
        print("%8d %12.2f %12.3f %12.3f %12.2f" % (
            count, moved / 2**20, d2h["mean_ms"], h2d["mean_ms"], d2h_gbps))

    print("\n=== contiguous 大拷贝参考（pinned，%d 块） ===" % args.max_blocks)
    gpu_flat = gpu_blocks.reshape(-1)
    pinned_flat = pinned_blocks.reshape(-1)
    total_elements = gpu_flat.numel()
    d2h = stats(timed_flat_copy(
        pinned_flat, gpu_flat, total_elements, device, args.repeats
    ))
    h2d = stats(timed_flat_copy(
        gpu_flat, pinned_flat, total_elements, device, args.repeats
    ))
    total_bytes = args.max_blocks * bytes_per_block
    report["flat_copy"] = {
        "blocks": args.max_blocks, "bytes": total_bytes,
        "d2h": d2h, "h2d": h2d,
        "d2h_gb_per_second": total_bytes / (d2h["mean_ms"] / 1000) / 1e9,
        "h2d_gb_per_second": total_bytes / (h2d["mean_ms"] / 1000) / 1e9,
    }
    print("D2H %.3f ms（%.2f GB/s），H2D %.3f ms（%.2f GB/s）" % (
        d2h["mean_ms"], report["flat_copy"]["d2h_gb_per_second"],
        h2d["mean_ms"], report["flat_copy"]["h2d_gb_per_second"],
    ))

    count = min(64, args.max_blocks)
    print("\n=== pageable 对照（%d 块） ===" % count)
    d2h = stats(timed_loop_copy(
        pageable_blocks, gpu_blocks, count, device, args.repeats
    ))
    h2d = stats(timed_loop_copy(
        gpu_blocks, pageable_blocks, count, device, args.repeats
    ))
    moved = count * bytes_per_block
    report["pageable"] = {
        "blocks": count, "bytes": moved, "d2h": d2h, "h2d": h2d,
        "d2h_gb_per_second": moved / (d2h["mean_ms"] / 1000) / 1e9,
        "h2d_gb_per_second": moved / (h2d["mean_ms"] / 1000) / 1e9,
    }
    print("D2H %.3f ms（%.2f GB/s），H2D %.3f ms（%.2f GB/s）" % (
        d2h["mean_ms"], report["pageable"]["d2h_gb_per_second"],
        h2d["mean_ms"], report["pageable"]["h2d_gb_per_second"],
    ))

    print("\n=== pinned 池预分配成本 ===")
    timings = []
    for _ in range(3):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        probe = torch.empty(shape, dtype=dtype, pin_memory=True)
        elapsed = (time.perf_counter() - start) * 1000
        del probe
        timings.append(elapsed)
    report["pinned_allocation"] = {
        "bytes": total_bytes, "wall_ms": stats(timings),
    }
    print("分配 %.1f MiB pinned：%.1f ms（均值）" % (
        total_bytes / 2**20, report["pinned_allocation"]["wall_ms"]["mean_ms"],
    ))

    save_results(args.output, report)


if __name__ == "__main__":
    main()
