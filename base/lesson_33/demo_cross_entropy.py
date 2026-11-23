import math

import torch


def loss_and_gradient(logits, target):
    if (logits.ndim != 1 or not logits.is_floating_point()
            or not 0 <= target < logits.numel() or not torch.isfinite(logits).all()):
        raise ValueError("expected finite floating logits [vocabulary] and valid target")
    shifted = logits - logits.max()
    exp = shifted.exp()
    total = exp.sum()
    probs = exp / total
    loss = total.log() - shifted[target]
    gradient = probs.clone()
    gradient[target] -= 1
    return probs, loss, gradient


def main():
    logits = torch.tensor([1, 2, 3], dtype=torch.float64).log()
    target = 1
    probs, loss, grad = loss_and_gradient(logits, target)
    print(f"probabilities: {[round(p, 6) for p in probs.tolist()]}")
    print(f"target={target}, loss={loss.item():.6f}")
    print(f"gradient: {[round(g, 6) for g in grad.tolist()]}")
    large_logits = logits + 1000
    assert torch.isinf(large_logits.exp()).all()
    _, stable, stable_grad = loss_and_gradient(large_logits, target)
    print(f"shift=1000: naive exp overflows, stable loss={stable.item():.6f}")
    torch.testing.assert_close(loss, torch.tensor(math.log(3), dtype=logits.dtype))
    torch.testing.assert_close(stable, loss, atol=1e-12, rtol=0)
    torch.testing.assert_close(stable_grad, grad, atol=1e-12, rtol=0)
    expected_grad = torch.tensor([1/6, -2/3, 1/2], dtype=logits.dtype)
    torch.testing.assert_close(grad, expected_grad, atol=1e-12, rtol=0)
    print("hand gradient and shift checks: pass")


if __name__ == "__main__":
    main()
