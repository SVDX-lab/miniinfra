import math

import torch


def step(weight, inputs, targets, learning_rate):
    if weight.ndim != 0 or inputs.ndim != 1 or inputs.shape != targets.shape or not inputs.numel():
        raise ValueError("expected a scalar weight and two matching nonempty batch vectors")
    if not weight.is_floating_point() or any(
        value.dtype != weight.dtype or value.device != weight.device for value in (inputs, targets)
    ):
        raise ValueError("expected one floating dtype and device")
    if not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning rate must be finite and nonnegative")
    error = weight * inputs - targets
    loss = error.square().mean() / 2
    gradient = (error * inputs).mean()
    return weight - learning_rate * gradient, loss, gradient


def main():
    x = torch.tensor([1., 2.], dtype=torch.float64)
    y = 2 * x
    for rate, expected in ((0.2, 1.75), (0.8, 4.0), (0.9, 5.90625)):
        weight = torch.zeros((), dtype=x.dtype)
        path = [weight.item()]
        for _ in range(3):
            weight, loss, grad = step(weight, x, y, rate)
            if len(path) == 1:
                assert loss.item() == 5 and grad.item() == -5
            path.append(round(weight.item(), 6))
        assert math.isclose(weight.item(), expected, abs_tol=1e-12, rel_tol=0)
        print(f"lr={rate:.1f}: w={path}")
    print("initial batch loss=5, gradient=-5; optimum w=2")


if __name__ == "__main__":
    main()
