import torch


def partition(x, w, parts):
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0]:
        raise ValueError("expected [M,K] multiplied by [K,N]")
    if x.dtype != w.dtype or x.device != w.device:
        raise ValueError("expected one dtype and device")
    if parts < 1 or not (x.shape[1] and w.shape[1]) or x.shape[1] % parts or w.shape[1] % parts:
        raise ValueError("positive K and N must each be divisible by parts")
    columns = [x @ block for block in w.chunk(parts, dim=1)]
    partials = [a @ b for a, b in zip(x.chunk(parts, dim=1), w.chunk(parts, dim=0), strict=True)]
    return columns, partials


def main():
    x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.float64)
    w = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]], dtype=x.dtype)
    columns, partials = partition(x, w, 2)
    via_concat = torch.cat(columns, dim=1)
    via_sum = torch.stack(partials).sum(dim=0)
    whole = x @ w
    assert torch.equal(whole, via_concat) and torch.equal(whole, via_sum)
    assert whole.tolist() == [[4., 6., 3., 7.], [6., 4., 7., 3.]]
    print(f"X={tuple(x.shape)}, W={tuple(w.shape)}, dtype={x.dtype}, device={x.device}")
    print(f"column outputs={[c.tolist() for c in columns]}")
    print(f"row partials={[p.tolist() for p in partials]}")
    print(f"concat == sum == full: {whole.tolist()}")


if __name__ == "__main__":
    main()
