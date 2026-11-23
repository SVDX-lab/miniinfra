import math

import torch


def attention(q, k, v, bias=None):
    if not all(t.ndim == 2 and min(t.shape) > 0 for t in (q, k, v)):
        raise ValueError("expected nonempty 2-D Q, K and V")
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Q and K need equal widths, K and V need equal row counts")
    if not all(t.device.type == "cpu" and t.dtype == torch.float64 for t in (q, k, v)):
        raise ValueError("this demonstration expects CPU float64")
    scores = (q @ k.T) / math.sqrt(q.shape[1])
    if bias is not None:
        if (bias.shape, bias.dtype, bias.device) != (scores.shape, scores.dtype, scores.device):
            raise ValueError("bias must match score shape, dtype and device")
        scores = scores + bias
    return torch.softmax(scores, dim=-1) @ v


def relative_scores(positions, log_ratio):
    offsets = positions[None, :] - positions[:, None]
    return offsets.to(torch.float64) * log_ratio


def sinusoidal(positions, width):
    if width <= 0 or width % 2:
        raise ValueError("this demonstration expects a positive even width")
    pair_indices = torch.arange(width // 2, dtype=torch.float64, device=positions.device)
    rates = 10000 ** (-2 * pair_indices / width)
    angles = positions[:, None] * rates
    return torch.stack((angles.sin(), angles.cos()), dim=-1).reshape(len(positions), width)


def main():
    def tensor(values):
        return torch.tensor(values, dtype=torch.float64, device="cpu")

    log3 = math.log(3)
    q = tensor([[1], [0]])
    k = tensor([[0], [log3]])
    v = tensor([[2], [10]])
    swap = torch.tensor([1, 0], dtype=torch.int64, device="cpu")
    original = attention(q, k, v)
    kv_swapped = attention(q[:1], k[swap], v[swap])
    all_swapped = attention(q[swap], k[swap], v[swap])

    causal = tensor([[0, -math.inf], [0, 0]])
    causal_original = attention(q, k, v, causal)
    causal_swapped = attention(q[swap], k[swap], v[swap], causal)

    absolute = tensor([[0], [log3]])
    absolute_original = attention(q[:1], k + absolute, v)
    absolute_swapped = attention(q[:1], k[swap] + absolute, v[swap])

    positions = torch.arange(2, dtype=torch.int64, device="cpu")
    relative = relative_scores(positions, log3)
    translated = relative_scores(positions + 10, log3)
    relative_output = attention(q, k, v, relative)
    translated_output = attention(q, k, v, translated)
    sine_table = sinusoidal(positions, 4)

    def close(actual, expected):
        torch.testing.assert_close(actual, tensor(expected), rtol=0, atol=1e-12)

    close(original, [[8], [6]])
    close(kv_swapped, [[8]])
    close(all_swapped, [[6], [8]])
    close(causal_original, [[2], [6]])
    close(causal_swapped, [[10], [8]])
    close(absolute_original, [[9.2]])
    close(absolute_swapped, [[6]])
    close(relative_output, [[9.2], [8]])
    torch.testing.assert_close(relative_output, translated_output, rtol=0, atol=1e-12)

    def show(column):
        return [round(value, 6) for value in column.flatten().tolist()]

    print(f"Q / K / V: {tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}")
    print(f"unmasked output: {show(original)}")
    print(f"fixed query 0, swap K/V: {show(kv_swapped)}")
    print(f"swap Q/K/V: {show(all_swapped)}")
    print(f"fixed causal mask: {show(causal_original)} -> {show(causal_swapped)}")
    print(f"fixed absolute slots, query 0: {show(absolute_original)} -> {show(absolute_swapped)}")
    print(f"relative, positions [0,1] and [10,11]: {show(relative_output)} -> {show(translated_output)}")
    for position, row in enumerate(sine_table):
        print(f"sinusoidal p={position}: {show(row)}")
    print("hand calculation: pass (float64, CPU, atol=1e-12)")


if __name__ == "__main__":
    main()
