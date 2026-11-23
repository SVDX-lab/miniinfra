import math

import torch


def split_heads(projected, num_heads):
    if projected.ndim != 3 or min(projected.shape) == 0:
        raise ValueError("expected nonempty (batch, seq, channels)")
    batch, seq, channels = projected.shape
    if num_heads <= 0 or channels % num_heads != 0:
        raise ValueError("channels must be divisible by a positive num_heads")
    head_dim = channels // num_heads
    return projected.reshape(batch, seq, num_heads, head_dim).transpose(1, 2)


def attention_heads(q, k, v, *, causal=False):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or min(q.shape) == 0:
        raise ValueError("Q, K and V must share nonempty (batch, heads, seq, head_dim)")
    if not all(t.device.type == "cpu" and t.dtype == torch.float64 for t in (q, k, v)):
        raise ValueError("this demonstration expects CPU float64")
    head_dim = q.shape[-1]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
    if causal:
        seq = q.shape[-2]
        future = torch.ones((seq, seq), dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(future, -math.inf)
    weights = torch.softmax(scores, dim=-1)
    return weights, weights @ v


def combine_heads(head_output):
    batch, heads, seq, head_dim = head_output.shape
    return head_output.transpose(1, 2).reshape(batch, seq, heads * head_dim)


def main():
    def tensor(values):
        return torch.tensor(values, dtype=torch.float64, device="cpu")

    x = tensor([[[1, 0, 0, 0], [0, 1, 0, 0]]])
    r = math.sqrt(2)
    c = math.log(3)
    w_q = tensor([[r, r, 0, 0], [0, 0, 0, 0], [r, r, 0, 0], [0, 0, 0, 0]])
    w_k = tensor([[c, 0, 0, 0], [0, 0, 0, 0], [0, c, 0, 0], [0, 0, 0, 0]])
    w_v = tensor([[4, 0, 0, 0], [0, 4, 0, 0], [8, 0, 0, 0], [0, 8, 0, 0]])
    w_o = tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1]])
    projections = tuple(x @ w.T for w in (w_q, w_k, w_v))
    q, k, v = (split_heads(projected, 2) for projected in projections)
    weights, heads = attention_heads(q, k, v)
    joined = combine_heads(heads)
    output = joined @ w_o.T
    causal_weights, causal_heads = attention_heads(q, k, v, causal=True)
    causal_output = combine_heads(causal_heads) @ w_o.T

    expected = tensor([[[5, 7, 3, 6], [5, 7, 3, 6]]])
    expected_causal = tensor([[[12, 0, 4, 0], [5, 7, 3, 6]]])
    torch.testing.assert_close(output, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(causal_output, expected_causal, rtol=0, atol=1e-12)

    def show(values):
        return [round(value, 4) for value in values.tolist()]

    print(f"X / packed Q: {tuple(x.shape)} / {tuple(projections[0].shape)}")
    print(f"split Q,K,V: {tuple(q.shape)}")
    print(f"weights / per-head output: {tuple(weights.shape)} / {tuple(heads.shape)}")
    for h in range(q.shape[1]):
        print(f"head {h}, query 0: weights={show(weights[0,h,0])}, out={show(heads[0,h,0])}")
    print(f"concatenated: {tuple(joined.shape)}, row 0={show(joined[0,0])}")
    print(f"projected: {tuple(output.shape)}, row 0={show(output[0,0])}")
    print(f"causal head 1, query 0: weights={show(causal_weights[0,1,0])}")
    print(f"causal projected rows: {show(causal_output[0,0])}, {show(causal_output[0,1])}")
    print("hand calculation: pass (float64, CPU, atol=1e-12)")


if __name__ == "__main__":
    main()
