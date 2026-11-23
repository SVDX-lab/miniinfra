import math

import torch


def apply_rope(x, positions, frequencies):
    if x.ndim != 3 or x.shape[-1] == 0 or x.shape[-1] % 2:
        raise ValueError("x must have shape [heads, positions, even head_dim]")
    if positions.shape != (x.shape[1],) or frequencies.shape != (x.shape[2] // 2,):
        raise ValueError("positions and frequencies must match the last two axes")
    if x.dtype != torch.float64 or positions.dtype != x.dtype or frequencies.dtype != x.dtype:
        raise ValueError("x, positions and frequencies must be float64")
    if x.device.type != "cpu" or positions.device != x.device or frequencies.device != x.device:
        raise ValueError("all inputs must be on CPU")

    angles = positions[:, None] * frequencies[None, :]
    cosine, sine = angles.cos(), angles.sin()
    first, second = x.chunk(2, dim=-1)
    return torch.cat(
        (first * cosine - second * sine, first * sine + second * cosine),
        dim=-1,
    )


def hand_values(tensor):
    # Remove trig round-off and signed zero for this integer-valued example.
    return tensor.round().to(torch.int64).tolist()


def main():
    q = torch.tensor([[[1, 0]], [[3, 4]]], dtype=torch.float64, device="cpu")
    k = q.new_tensor([[[0, 1]], [[4, 0]]])
    v = q.new_tensor([[[2, 5]], [[7, 8]]])
    frequency = q.new_tensor([math.pi / 2])
    q_pos, k_pos = q.new_tensor([1]), q.new_tensor([2])

    q_rot = apply_rope(q, q_pos, frequency)
    k_rot = apply_rope(k, k_pos, frequency)
    dots = (q_rot * k_rot).sum(dim=-1)
    relative = (q * apply_rope(k, k_pos - q_pos, frequency)).sum(dim=-1)
    q_shift = apply_rope(q, q_pos + 2, frequency)
    k_shift = apply_rope(k, k_pos + 2, frequency)
    shifted = (q_shift * k_shift).sum(dim=-1)
    k_only_shift = apply_rope(k, k_pos + 1, frequency)
    different = (q_rot * k_only_shift).sum(dim=-1)
    torch.testing.assert_close(dots, relative, rtol=0, atol=1e-12)
    torch.testing.assert_close(dots, shifted, rtol=0, atol=1e-12)
    torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1), rtol=0, atol=1e-12)

    print(f"Q/K/V: {tuple(q.shape)}, {q.dtype}, {q.device}")
    print(f"Q at 1: {hand_values(q_rot)}")
    print(f"K at 2: {hand_values(k_rot)}")
    print(f"V unchanged: {hand_values(v)}")
    print(f"dots: {hand_values(dots)}")
    print(f"relative 2-1: {hand_values(relative)}")
    print(f"shift both +2: {hand_values(shifted)}")
    print(f"shift K only +1: {hand_values(different)}")

    vector = q.new_tensor([[[1, 3, 2, 4]]])
    two_planes = apply_rope(vector, q_pos, q.new_tensor([math.pi / 2, math.pi]))
    print(f"two planes: {hand_values(vector)} -> {hand_values(two_planes)}")


if __name__ == "__main__":
    main()
