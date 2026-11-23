import math

import torch


def row_softmax(scores):
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    exp = shifted.exp()
    return exp / exp.sum(dim=-1, keepdim=True)


def attention(q, k, v):
    if any(x.ndim != 2 or x.numel() == 0 for x in (q, k, v)):
        raise ValueError("Q, K and V must be nonempty rank-2 tensors")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Q and K widths, and K and V row counts, must match")
    if not q.is_floating_point() or any(
        x.dtype != q.dtype or x.device != q.device for x in (k, v)
    ):
        raise ValueError("Q, K and V must share floating dtype and device")

    scaled_scores = (q @ k.T) / math.sqrt(q.shape[1])
    if not torch.isfinite(scaled_scores).all() or not torch.isfinite(v).all():
        raise ValueError("Scaled scores and V must be finite")
    weights = row_softmax(scaled_scores)
    return scaled_scores, weights, weights @ v


def main():
    options = {"dtype": torch.float64, "device": "cpu"}
    q = torch.tensor([[2, 0, 0, 0], [0, 2, 0, 0]], **options)
    k = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], **options)
    v = torch.tensor([[2, 0], [0, 4], [1, 1]], **options)
    scores, weights, output = attention(q, k, v)

    e = math.e
    expected = torch.tensor([[2 * e + 1, 5], [3, 4 * e + 1]], **options)
    torch.testing.assert_close(output, expected / (e + 2), rtol=0, atol=1e-14)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, **options))

    print("dtype=float64, device=cpu")
    print(f"Q={tuple(q.shape)} K={tuple(k.shape)} V={tuple(v.shape)}")
    print(f"scaled scores={scores.tolist()}")
    for name, tensor in (("weights", weights), ("output", output)):
        for row, values in enumerate(tensor.tolist()):
            print(f"{name}[{row}]=" + ", ".join(f"{x:.6f}" for x in values))
    print("hand calculation: passed")


if __name__ == "__main__":
    main()
