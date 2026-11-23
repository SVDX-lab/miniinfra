import torch


def main():
    # Three exact powers or sums of powers of two in FP32.
    inputs = torch.tensor(
        [2**16, 2**-26, 1 + 3 * 2**-10],
        dtype=torch.float32,
        device="cpu",
    )
    print(
        f"input: shape={tuple(inputs.shape)}, "
        f"dtype={inputs.dtype}, device={inputs.device}"
    )
    print("format  bytes  2^16     2^-26              1+3*2^-10")

    formats = (
        ("FP32", torch.float32),
        ("FP16", torch.float16),
        ("BF16", torch.bfloat16),
    )
    for name, dtype in formats:
        stored = inputs.to(dtype=dtype)
        large, small, near_one = stored.tolist()
        print(
            f"{name:<6}  {stored.element_size():>5}  "
            f"{large:<7g}  {small:<17.10g}  {near_one:.10f}"
        )


if __name__ == "__main__":
    main()
