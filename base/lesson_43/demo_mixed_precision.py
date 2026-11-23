import torch


def main():
    grad = torch.tensor(2.**-28, dtype=torch.float32)
    scale = 2.**16
    direct = grad.to(torch.float16)
    bf16 = grad.to(torch.bfloat16)
    scaled = (grad * scale).to(torch.float16)
    recovered = scaled.float() / scale
    assert direct.item() == 0 and bf16.item() == grad.item() == recovered.item()
    print(f"small gradient: FP32={grad.item():.10e}, FP16={direct.item():.1f}, BF16={bf16.item():.10e}")
    print(f"scale={int(scale)}: stored FP16={scaled.item():.10e}, FP32 unscale={recovered.item():.10e}")

    master = torch.tensor(1., dtype=torch.float32)
    rounded = master.to(torch.bfloat16)
    for _ in range(4):
        master = master - 2.**-10
        rounded = rounded - 2.**-10
    forward_copy = master.to(torch.bfloat16)
    assert master.item() == forward_copy.item() == 0.99609375 and rounded.item() == 1.
    print(f"four updates: FP32 master={master.item():.8f}, BF16 copy={forward_copy.item():.8f}, "
          f"BF16 stored updates={rounded.item():.8f}")

    low_sum = torch.tensor(1., dtype=torch.bfloat16)
    high_sum = low_sum.float()
    increment = torch.tensor(2.**-10, dtype=torch.bfloat16)
    for _ in range(8):
        low_sum += increment
        high_sum += increment.float()
    assert low_sum.item() == 1. and high_sum.item() == 1.0078125
    print(f"eight additions: BF16 accumulator={low_sum.item():.8f}, "
          f"FP32 accumulator={high_sum.item():.8f}")


if __name__ == "__main__":
    main()
