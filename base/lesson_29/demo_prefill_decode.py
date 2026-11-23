import math

import torch


def attention(q, k, v, *, first_query):
    if (q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or k.shape[0] != v.shape[0]
            or q.shape[1] != k.shape[1] or q.shape[1] == 0
            or not 0 <= first_query or first_query + q.shape[0] > k.shape[0]):
        raise ValueError("expected matching Q/K channels and valid aligned sequence positions")
    if not q.is_floating_point() or any(t.dtype != q.dtype or t.device != q.device for t in (k, v)):
        raise ValueError("Q/K/V must have the same floating dtype and device")
    q_pos = first_query + torch.arange(q.shape[0], device=q.device)
    k_pos = torch.arange(k.shape[0], device=q.device)
    scores = q @ k.T / math.sqrt(q.shape[1])
    weights = scores.masked_fill(k_pos[None, :] > q_pos[:, None], -torch.inf).softmax(-1)
    return weights @ v, scores.shape


def main():
    q = torch.zeros((5, 1), dtype=torch.float64)
    k = torch.ones_like(q)
    v = torch.tensor([[2], [4], [6], [8], [10]], dtype=q.dtype)
    full, _ = attention(q, k, v, first_query=0)
    prefill, grid = attention(q[:3], k[:3], v[:3], first_query=0)
    print(f"prefill: scores={tuple(grid)}, output={prefill.flatten().tolist()}")
    cached_k, cached_v = k[:3], v[:3]
    parts = [prefill]
    for pos in (3, 4):
        cached_k = torch.cat((cached_k, k[pos:pos + 1]))
        cached_v = torch.cat((cached_v, v[pos:pos + 1]))
        value, grid = attention(q[pos:pos + 1], cached_k, cached_v, first_query=pos)
        parts.append(value)
        print(f"decode pos={pos}: scores={tuple(grid)}, output={value.flatten().tolist()}")
    split = torch.cat(parts)
    expected = torch.tensor([[2], [3], [4], [5], [6]], dtype=q.dtype)
    torch.testing.assert_close(split, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(split, full, rtol=0, atol=1e-12)
    print("cached path / full causal path: match (atol=1e-12)")


if __name__ == "__main__":
    main()
