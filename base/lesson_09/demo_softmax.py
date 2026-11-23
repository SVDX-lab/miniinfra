import math

import torch


def softmax_last(logits, temperature=1.0):
    if logits.dtype != torch.float64 or logits.device.type != "cpu":
        raise ValueError("this demo expects a CPU float64 tensor")
    if logits.ndim == 0 or logits.shape[-1] == 0:
        raise ValueError("the last axis must contain at least one score")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not torch.isfinite(logits).all():
        raise ValueError("this demo expects finite scores")

    maximum = logits.amax(dim=-1, keepdim=True)
    shifted = logits - maximum
    if not torch.isfinite(shifted).all():
        raise ValueError("the score range must fit in float64")
    weights = torch.exp(shifted / temperature)
    return weights / weights.sum(dim=-1, keepdim=True)


def format_row(values):
    return "[" + ", ".join(f"{value:.6f}" for value in values.tolist()) + "]"


def main():
    logits = torch.tensor([math.log(4), math.log(2), 0], dtype=torch.float64, device="cpu")
    print(f"logits {tuple(logits.shape)}: {format_row(logits)}")
    for temperature in (0.5, 1.0, 2.0):
        probs = softmax_last(logits, temperature)
        print(f"T={temperature:.1f}: {format_row(probs)}, "
              f"sum={probs.sum().item():.6f}, argmax={probs.argmax().item()}")

    expected = torch.tensor(
        [4 / 7, 2 / 7, 1 / 7], dtype=logits.dtype, device=logits.device
    )
    probs = softmax_last(logits)
    torch.testing.assert_close(probs, expected, rtol=0, atol=1e-14)

    large_logits = logits + 1000
    naive_weights = torch.exp(large_logits)
    naive = naive_weights / naive_weights.sum()
    stable = softmax_last(large_logits)
    torch.testing.assert_close(stable, probs, rtol=0, atol=1e-12)
    print(f"shift +1000, naive: {format_row(naive)}")
    print(f"shift +1000, stable: {format_row(stable)}")
    print("stable matches original within atol=1e-12")


if __name__ == "__main__":
    main()
