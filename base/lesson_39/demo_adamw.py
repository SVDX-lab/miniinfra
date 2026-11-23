import math

import torch


def update(theta, gradient, m, v, step, *, decoupled):
    lr, decay, eps = 0.1, 0.2, 1e-8
    beta1, beta2 = 0.5, 0.75
    g = gradient if decoupled else gradient + decay * theta
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** step)
    v_hat = v / (1 - beta2 ** step)
    base = theta * (1 - lr * decay) if decoupled else theta
    theta = base - lr * m_hat / (math.sqrt(v_hat) + eps)
    return theta, m, v, m_hat, v_hat


def main():
    for label, optimizer_type, decoupled in (
        ("AdamW", torch.optim.AdamW, True), ("Adam + L2", torch.optim.Adam, False)
    ):
        parameter = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        optimizer = optimizer_type([parameter],
                                   lr=0.1, betas=(0.5, 0.75), eps=1e-8,
                                   weight_decay=0.2, foreach=False)
        theta, m, v = 1., 0., 0.
        for step, raw in enumerate((2., -1.), start=1):
            theta, m, v, m_hat, v_hat = update(theta, raw, m, v, step, decoupled=decoupled)
            parameter.grad = torch.tensor(raw, dtype=parameter.dtype)
            optimizer.step()
            assert math.isclose(theta, parameter.item(), abs_tol=1e-12, rel_tol=0)
            print(f"{label} step={step}: m={m:.6f}, v={v:.6f}, "
                  f"m_hat={m_hat:.6f}, v_hat={v_hat:.6f}, theta={theta:.6f}")
    print("manual scalar updates match PyTorch")


if __name__ == "__main__":
    main()
