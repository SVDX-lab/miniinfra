import torch


def forward(x):
    u = x.square()
    v = 2 * x
    w = u * v
    return w + u, u, v


def manual_adjoint(x, u, v):
    loss_bar = 1
    w_bar = loss_bar
    u_bar = loss_bar + w_bar * v
    v_bar = w_bar * u
    via_u = u_bar * 2 * x
    via_v = v_bar * 2
    return u_bar, v_bar, via_u, via_v


def main():
    x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    loss, u, v = forward(x)
    print(f"forward: x={x.item():.0f}, u={u.item():.0f}, v={v.item():.0f}, loss={loss.item():.0f}")
    du, dv, via_u, via_v = manual_adjoint(x.item(), u.item(), v.item())
    print(f"reverse: dL/du={du:.0f}, dL/dv={dv:.0f}, paths to x={via_u:.0f}+{via_v:.0f}")
    expected = via_u + via_v

    loss.backward()
    assert x.grad is not None and x.grad.item() == expected
    print(f"first backward: x.grad={x.grad.item():.0f}")
    forward(x)[0].backward()
    assert x.grad.item() == 2 * expected
    print(f"fresh graph, same leaf: x.grad={x.grad.item():.0f}")
    x.grad = None
    forward(x)[0].backward()
    assert x.grad.item() == expected
    print(f"reset grad, fresh graph: x.grad={x.grad.item():.0f}")


if __name__ == "__main__":
    main()
