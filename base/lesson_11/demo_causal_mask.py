import torch


def row_softmax(scores):
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    exp = shifted.exp()
    return exp / exp.sum(dim=-1, keepdim=True)


def causal_weights(scores):
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or scores.numel() == 0:
        raise ValueError("scores must be a nonempty square matrix")
    if not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError("scores must be finite floating-point values")

    positions = torch.arange(scores.shape[0], device=scores.device)
    future = positions[None, :] > positions[:, None]
    mask = torch.zeros_like(scores).masked_fill(future, float("-inf"))
    return mask, row_softmax(scores + mask)


def main():
    options = {"dtype": torch.float64, "device": "cpu"}
    scores = torch.zeros(4, 4, **options)
    values = torch.tensor([[2, 0], [0, 4], [4, 2], [10, 6]], **options)
    changed = values.clone()
    changed[3] += 8

    mask, weights = causal_weights(scores)
    before, after = weights @ values, weights @ changed
    unmasked = row_softmax(scores)
    open_before, open_after = unmasked @ values, unmasked @ changed

    expected = torch.tensor([[2, 0], [1, 2], [2, 2], [4, 3]], **options)
    torch.testing.assert_close(before, expected, rtol=0, atol=1e-14)
    torch.testing.assert_close(before[:3], after[:3], rtol=0, atol=0)
    assert ((open_after[:3] - open_before[:3]) != 0).all().item()
    assert (weights[mask.isneginf()] == 0).all().item()

    print("scores=(4, 4) V=(4, 2), dtype=float64, device=cpu")
    for index, row in enumerate(mask.tolist()):
        print(f"mask[{index}]=" + ", ".join(f"{value:g}" for value in row))
    for index, row in enumerate(weights.tolist()):
        print(f"weights[{index}]=" + ", ".join(f"{value:.6f}" for value in row))
    print(f"masked before={before.tolist()}")
    print(f"masked after ={after.tolist()}")
    print(f"unmasked row 0: {open_before[0].tolist()} -> {open_after[0].tolist()}")
    print("prefix unchanged with mask: passed")


if __name__ == "__main__":
    main()
