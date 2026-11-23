import torch


def reduce_losses(losses, valid):
    if (losses.ndim != 2 or not losses.is_floating_point() or valid.shape != losses.shape
            or valid.dtype != torch.bool or valid.device != losses.device):
        raise ValueError("expected loss and boolean mask [batch, token] on one device")
    counts = valid.sum(dim=1)
    kept = counts > 0
    if not kept.any():
        raise ValueError("batch has no supervised tokens")
    sums = losses.masked_fill(~valid, 0).sum(dim=1)
    by_sample = (sums[kept] / counts[kept]).mean()
    by_token = sums.sum() / counts.sum()
    return by_sample, by_token, sums, counts


def main():
    scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    raw = torch.tensor([[4., 100., 100.], [1., 1., 1.]], dtype=scale.dtype)
    valid = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    a, b, sums, counts = reduce_losses(scale * raw, valid)
    da = torch.autograd.grad(a, scale, retain_graph=True)[0]
    db = torch.autograd.grad(b, scale)[0]
    print(f"sums={sums.tolist()}, counts={counts.tolist()}, shape={tuple(raw.shape)}, dtype={raw.dtype}")
    print(f"sample mean={a.item():.2f}, dL/dscale={da.item():.2f}")
    print(f"token mean={b.item():.2f}, dL/dscale={db.item():.2f}")
    pooled = ((sums / counts) * counts).sum() / counts.sum()
    assert (a.item(), b.item(), da.item(), db.item()) == (2.5, 1.75, 2.5, 1.75)
    assert pooled.item() == b.item()
    print(f"merge two microbatches by token count={pooled.item():.2f}")


if __name__ == "__main__":
    main()
