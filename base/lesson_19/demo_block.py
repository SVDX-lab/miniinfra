from dataclasses import dataclass
import math

import torch


@dataclass
class BlockWeights:
    input_scale: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    post_scale: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


def rms_norm(x, scale, epsilon):
    mean_square = x.square().mean(dim=-1, keepdim=True)
    return x / (mean_square + epsilon).sqrt() * scale


def project_heads(x, weight, head_dim):
    batch, sequence, _ = x.shape
    return (x @ weight.T).reshape(batch, sequence, -1, head_dim).transpose(1, 2)


def rotate_heads(x, angles):
    first, second = x.chunk(2, dim=-1)
    cosine, sine = angles.cos(), angles.sin()
    return torch.cat((first * cosine - second * sine,
                      first * sine + second * cosine), dim=-1)


def attention(x, weights, positions, epsilon):
    batch, sequence, _ = x.shape
    head_dim = weights.q_scale.numel()
    q = rms_norm(project_heads(x, weights.q, head_dim), weights.q_scale, epsilon)
    k = rms_norm(project_heads(x, weights.k, head_dim), weights.k_scale, epsilon)
    v = project_heads(x, weights.v, head_dim)

    # One illustrative frequency, shared by the two channels in each Head.
    angles = (positions.to(x.dtype) * (math.pi / 2)).reshape(1, 1, sequence, 1)
    q, k = rotate_heads(q, angles), rotate_heads(k, angles)
    kv_heads = k.shape[1]
    group_size = q.shape[1] // kv_heads
    q_grouped = q.reshape(batch, kv_heads, group_size, sequence, head_dim)
    scores = (q_grouped @ k.unsqueeze(2).transpose(-1, -2)) / math.sqrt(head_dim)
    future = positions[:, None] < positions[None, :]
    probabilities = scores.masked_fill(future, -torch.inf).softmax(dim=-1)
    grouped_context = probabilities @ v.unsqueeze(2)
    context = grouped_context.reshape(batch, -1, sequence, head_dim)
    merged = context.transpose(1, 2).reshape(batch, sequence, -1)
    return merged @ weights.o.T, q, k, probabilities


def block(x, weights, positions, epsilon):
    if x.ndim != 3 or x.shape[-1] != 2 or 0 in x.shape[:2]:
        raise ValueError("expected a nonempty [B,S,2] tensor")
    if positions.shape != (x.shape[1],) or positions.device != x.device:
        raise ValueError("expected positions [S] on the input device")
    expected_positions = torch.arange(x.shape[1], device=x.device)
    if positions.dtype != torch.int64 or not torch.equal(positions, expected_positions):
        raise ValueError("positions must be int64 and consecutive from zero")
    if not x.is_floating_point() or any(
        weight.dtype != x.dtype or weight.device != x.device
        for weight in vars(weights).values()
    ):
        raise ValueError("expected one floating dtype and one device")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    expected_shapes = ((2,), (4, 2), (2, 2), (2, 2), (2, 4),
                       (2,), (2,), (2,), (1, 2), (1, 2), (2, 1))
    if tuple(weight.shape for weight in vars(weights).values()) != expected_shapes:
        raise ValueError("weights must match the teaching block shapes")

    attention_input = rms_norm(x, weights.input_scale, epsilon)
    delta, q, k, probabilities = attention(attention_input, weights, positions, epsilon)
    residual = x + delta
    ffn_input = rms_norm(residual, weights.post_scale, epsilon)
    gate = ffn_input @ weights.gate.T
    up = ffn_input @ weights.up.T
    ffn = (gate * gate.sigmoid() * up) @ weights.down.T
    return residual + ffn, q, k, probabilities, delta, residual, ffn


def display(tensor):
    return [round(value, 6) for value in tensor.flatten().tolist()]


def main():
    options = dict(dtype=torch.float64, device="cpu")

    def tensor(values):
        return torch.tensor(values, **options)

    unit = tensor([1, 1])
    eye = torch.eye(2, **options)
    weights = BlockWeights(
        input_scale=unit, q=eye.repeat(2, 1), k=eye, v=eye,
        o=torch.cat((eye, eye), dim=1) / 2,
        q_scale=unit, k_scale=unit, post_scale=unit,
        gate=tensor([[1, 0]]), up=tensor([[2, 0]]), down=tensor([[0], [1]]),
    )
    x = tensor([[[2, 0], [2, 0]]])
    positions = torch.arange(2, dtype=torch.int64, device="cpu")
    epsilon = 1e-6
    out, q, k, weights_attn, delta, residual, ffn = block(x, weights, positions, epsilon)
    print(f"X / Q / K: {tuple(x.shape)} / {tuple(q.shape)} / {tuple(k.shape)}")
    print(f"Q head 0 by position: {display(q[0, 0])}")
    print(f"attention rows, Q head 0: {display(weights_attn[0, 0, 0])}")
    print(f"attention update: {display(delta)}")
    print(f"after first addition: {display(residual)}")
    print(f"FFN update: {display(ffn)}")
    print(f"block output: {display(out)}, {out.dtype}, {out.device}")

    a = 2 / math.sqrt(2 + epsilon)
    c = a / math.sqrt(a * a / 2 + epsilon)
    expected_q = tensor([[c, 0], [0, c]])
    first_probability = 1 / (1 + math.exp(c * c / math.sqrt(2)))
    expected_probs = tensor([[1, 0], [first_probability, 1 - first_probability]])
    torch.testing.assert_close(q[0, 0], expected_q, atol=1e-12, rtol=0)
    torch.testing.assert_close(weights_attn[0, 0, 0], expected_probs, atol=1e-12, rtol=0)
    r = 2 + a
    z = r / math.sqrt(r * r / 2 + epsilon)
    expected = tensor([[[r, 2 * z * z / (1 + math.exp(-z))]] * 2])
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=0)
    changed = tensor([[[2, 0], [0, 2]]])
    changed_out, *_ = block(changed, weights, positions, epsilon)
    torch.testing.assert_close(out[:, 0], changed_out[:, 0], atol=1e-12, rtol=0)
    assert not torch.allclose(out[:, 1], changed_out[:, 1])
    print("hand calculation and causal prefix: pass")


if __name__ == "__main__":
    main()
