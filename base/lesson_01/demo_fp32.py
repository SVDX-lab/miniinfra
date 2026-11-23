"""在 CPU 上观察 IEEE 754 binary32 的字段、舍入和特殊值。"""

import torch


def fp32(value: float | int) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32, device="cpu")


def split_bits(value: torch.Tensor) -> tuple[int, int, int]:
    if value.shape != () or value.dtype != torch.float32 or not value.is_cpu:
        raise ValueError("expected a scalar CPU tensor with dtype torch.float32")

    word = value.reshape(1).view(torch.int32).item() & 0xFFFFFFFF
    sign = word >> 31
    exponent = (word >> 23) & 0xFF
    fraction = word & 0x7FFFFF
    return sign, exponent, fraction


def main() -> None:
    value = fp32(-6.5)
    sign, exponent, fraction = split_bits(value)
    print(f"-6.5 bits = {sign:b} | {exponent:08b} | {fraction:023b}")
    print(f"exponent = {exponent} - 127 = {exponent - 127}")
    significand = 1 + fraction / 2**23
    decoded = (-1)**sign * significand * 2**(exponent - 127)
    print(f"significand = {significand}, decoded = {decoded}")

    base = fp32(2**24)
    midpoint = base + fp32(1)
    neighbor = base + fp32(2)
    print(f"2^24 = {base.item():.0f}")
    print(f"2^24 + 1 -> {midpoint.item():.0f}")
    print(f"2^24 + 2 -> {neighbor.item():.0f}")

    zero = fp32(0)
    infinity = fp32(1) / zero
    nan = zero / zero
    sign, exponent, fraction = split_bits(infinity)
    print(f"1 / +0 -> {infinity.item()}, fields = {sign}, {exponent}, {fraction}")
    _, exponent, fraction = split_bits(nan)
    print(
        f"0 / 0 -> {nan.item()}, exponent = {exponent}, "
        f"fraction != 0: {fraction != 0}"
    )
    print(f"NaN == NaN: {(nan == nan).item()}, isnan: {torch.isnan(nan).item()}")


if __name__ == "__main__":
    main()
