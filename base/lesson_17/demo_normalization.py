import math

import torch


def normalize(x, epsilon):
    if x.ndim < 1 or x.shape[-1] == 0 or not x.is_floating_point():
        raise ValueError("expected a floating tensor with a nonempty channel axis")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    variance = centered.square().mean(dim=-1, keepdim=True)
    mean_square = x.square().mean(dim=-1, keepdim=True)
    layer = centered / (variance + epsilon).sqrt()
    rms = x / (mean_square + epsilon).sqrt()
    return mean, variance, mean_square, layer, rms


def rows(tensor):
    return [[round(value, 6) for value in row] for row in tensor.tolist()]


def main():
    options = dict(dtype=torch.float64, device="cpu")
    epsilon = 1e-6
    x = torch.tensor([[1, 3], [3, 5], [0, 0]], **options)
    mean, variance, mean_square, layer, rms = normalize(x, epsilon)
    print(f"X: {rows(x)}, shape={tuple(x.shape)}, {x.dtype}, {x.device}")
    print(f"mean: {rows(mean)}")
    print(f"population variance: {rows(variance)}")
    print(f"mean square: {rows(mean_square)}")
    print(f"LayerNorm core: {rows(layer)}")
    print(f"RMSNorm core: {rows(rms)}")
    torch.testing.assert_close(layer[0], layer[1], atol=1e-12, rtol=0)
    expected = torch.tensor([1, 3], **options) / math.sqrt(5 + epsilon)
    torch.testing.assert_close(rms[0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(rms[-1], torch.zeros(2, **options), atol=0, rtol=0)
    print("shift and zero-row checks: pass")


if __name__ == "__main__":
    main()
