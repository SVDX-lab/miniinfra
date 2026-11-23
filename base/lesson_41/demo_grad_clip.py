import math

import torch


@torch.no_grad()
def clip_global(parameters, threshold):
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0., 1.
    anchor = gradients[0]
    if any(not g.is_floating_point() or g.device != anchor.device or g.is_sparse for g in gradients):
        raise ValueError("expected dense floating gradients on one device")
    square_sum = torch.stack([g.to(torch.float64).square().sum() for g in gradients]).sum()
    norm = square_sum.sqrt().item()
    if not math.isfinite(norm):
        raise ValueError("gradient norm is not finite")
    factor = min(1., threshold / norm) if norm else 1.
    if factor < 1:
        for gradient in gradients:
            gradient.mul_(factor)
    return norm, factor


def main():
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    y = torch.zeros((), dtype=x.dtype, requires_grad=True)
    loss = x @ x.new_tensor([3., 4.]) + 12 * y
    loss.backward()
    print(f"before: x.grad={x.grad.tolist()}, y.grad={y.grad.item():.1f}")
    norm, scale = clip_global((x, y), 6.5)
    assert (norm, scale) == (13., 0.5)
    print(f"norm={norm:.1f}, threshold=6.5, scale={scale:.2f}")
    after, _ = clip_global((x, y), 100.)
    assert after == 6.5
    print(f"after: x.grad={x.grad.tolist()}, y.grad={y.grad.item():.1f}, norm={after:.1f}")
    with torch.no_grad():
        x.sub_(0.25 * x.grad)
        y.sub_(0.25 * y.grad)
    print(f"one SGD update: x={x.tolist()}, y={y.item():.2f}")


if __name__ == "__main__":
    main()
