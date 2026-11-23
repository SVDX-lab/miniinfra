import torch


def finite_difference(function, x, step):
    if step <= 0:
        raise ValueError("the finite-difference step must be positive")
    with torch.no_grad():
        return (function(x + step) - function(x - step)) / (2 * step)


def main():
    x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    u = x.square()
    y = u * x
    print(f"x/u: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
    print(f"x: leaf={x.is_leaf}, requires_grad={x.requires_grad}, grad={x.grad}")
    print(f"u: leaf={u.is_leaf}, requires_grad={u.requires_grad}, retains_grad={u.retains_grad}")
    u.retain_grad()
    y.backward()
    print(f"y={y.item():.1f}; x.grad={x.grad.item():.1f}; retained u.grad={u.grad.item():.1f}")

    step = 1e-4
    numerical = finite_difference(lambda value: value.square() * value, x, step)
    error = (numerical - x.grad).abs().item()
    assert x.grad.item() == 12 and u.grad.item() == 2 and error < 1e-7
    print(f"central difference (h={step:g}): {numerical.item():.8f}; abs error < 1e-7")
    with torch.no_grad():
        value = x.square()
    assert not value.requires_grad and value.grad_fn is None and x.requires_grad
    print(f"no_grad: output.requires_grad={value.requires_grad}, x.requires_grad={x.requires_grad}")


if __name__ == "__main__":
    main()
