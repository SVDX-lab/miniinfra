import math

import torch


def grouped_attention(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Q, K and V must be 4-D, with matching K/V shapes")
    batch, query_heads, query_length, width = q.shape
    kv_batch, kv_heads, key_length, kv_width = k.shape
    if batch != kv_batch or width != kv_width or not width or not key_length:
        raise ValueError("Batch/head width must match, with positive width and key length")
    if not query_heads or not kv_heads or query_heads % kv_heads:
        raise ValueError("Positive query head count must be divisible by KV head count")
    if k.dtype != q.dtype or v.dtype != q.dtype or not q.is_floating_point():
        raise ValueError("Q, K and V must share a floating dtype")
    if k.device != q.device or v.device != q.device:
        raise ValueError("Q, K and V must share a device")

    group_size = query_heads // kv_heads
    grouped_q = q.reshape(batch, kv_heads, group_size, query_length, width)
    keys = k.unsqueeze(2).transpose(-1, -2)
    scores = (grouped_q @ keys) / math.sqrt(width)
    weights = scores.softmax(dim=-1)
    context = weights @ v.unsqueeze(2)
    weights = weights.reshape(batch, query_heads, query_length, key_length)
    return weights, context.reshape(q.shape)


def rounded_rows(tensor):
    return [[round(value, 4) for value in row] for row in tensor[0, :, 0].tolist()]


def main():
    q = torch.tensor([[[[1, 0]], [[0, 1]], [[-1, 0]], [[0, -1]]]],
                     dtype=torch.float64, device="cpu")
    scale = math.sqrt(2) * math.log(3)
    k = q.new_tensor([[[[0, 0], [scale, 0]], [[0, 0], [0, scale]]]])
    v = q.new_tensor([[[[0, 2], [4, 6]], [[10, 20], [14, 24]]]])
    weights, context = grouped_attention(q, k, v)
    print(f"Q: {tuple(q.shape)}, K/V: {tuple(k.shape)}, {q.dtype}, {q.device}")
    print(f"weights: {tuple(weights.shape)} {rounded_rows(weights)}")
    print(f"context: {tuple(context.shape)} {rounded_rows(context)}")

    for name, kv_heads in (("MHA", 4), ("GQA", 2), ("MQA", 1)):
        group_size = q.shape[1] // kv_heads
        head_map = [head // group_size for head in range(q.shape[1])]
        kv_shape = (q.shape[0], kv_heads, k.shape[2], k.shape[3])
        elements = 2 * math.prod(kv_shape)
        print(
            f"{name}: map={head_map}, K/V each={kv_shape}, "
            f"bytes={elements * k.element_size()}"
        )

    # Explicit repetition is only a numerical reference for this tiny input.
    repeated_k = k.repeat_interleave(2, dim=1)
    repeated_v = v.repeat_interleave(2, dim=1)
    reference_scores = q @ repeated_k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    reference = reference_scores.softmax(-1)
    torch.testing.assert_close(weights, reference, rtol=0, atol=1e-12)
    torch.testing.assert_close(context, reference @ repeated_v, rtol=0, atol=1e-12)
    print("repeat reference: matches within atol=1e-12")


if __name__ == "__main__":
    main()
