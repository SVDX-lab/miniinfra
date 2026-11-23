import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Work:
    flops: int
    logical_bytes: int

    @property
    def intensity(self):
        return self.flops / self.logical_bytes


def add_work(count, element_bytes):
    return Work(count, 3 * count * element_bytes)


def matmul_work(rows, shared, columns, element_bytes):
    return Work(
        2 * rows * shared * columns,
        (rows * shared + shared * columns + rows * columns) * element_bytes,
    )


def roof(work, peak_gflops, bandwidth_gbs):
    return min(peak_gflops, bandwidth_gbs * work.intensity)


def verify_small(device):
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device=device)
    b = torch.tensor([[2, 0], [1, 3]], dtype=a.dtype, device=device)
    total, product = a + b, a @ b
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    options = dict(dtype=a.dtype, device=device)
    torch.testing.assert_close(total, torch.tensor([[3, 2], [4, 7]], **options))
    torch.testing.assert_close(product, torch.tensor([[4, 6], [10, 12]], **options))
    print(f"small FP32 add={total.to('cpu').tolist()}, matmul={product.to('cpu').tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Compare logical work and an idealized Roofline")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu_smoke else "cuda:0")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. Use --cpu-smoke for the arithmetic check.")
        props = torch.cuda.get_device_properties(device)
        print(f"device={props.name}, SMs={props.multi_processor_count}, "
              f"compute capability={props.major}.{props.minor}")
        print(f"torch={torch.__version__}, torch CUDA build={torch.version.cuda}")
    else:
        print("CPU smoke: logical traffic and hypothetical ceilings only")

    verify_small(device)
    peak_gflops, bandwidth_gbs = 1000, 100
    print(f"hypothetical peak={peak_gflops} GFLOP/s, bandwidth={bandwidth_gbs} GB/s, "
          f"balance={peak_gflops / bandwidth_gbs:.0f} FLOP/byte")
    for name, work in (
        ("add N=4096", add_work(4096, 4)),
        ("matmul 256x256x256", matmul_work(256, 256, 256, 4)),
    ):
        print(f"{name}: FLOP={work.flops}, bytes={work.logical_bytes}, "
              f"I={work.intensity:.6f}, roof={roof(work, peak_gflops, bandwidth_gbs):.3f} GFLOP/s")


if __name__ == "__main__":
    main()
