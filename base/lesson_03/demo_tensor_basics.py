import torch


def describe(name, tensor):
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, device={tensor.device}"
    )


def main():
    x = torch.arange(24, dtype=torch.float32, device="cpu").reshape(2, 3, 4)
    describe("x", x)
    describe("x_bf16", x.to(dtype=torch.bfloat16))
    print(f"x[1, 2] = {x[1, 2].tolist()}")

    by_head = x.reshape(2, 3, 2, 2)
    head_first = by_head.transpose(1, 2)
    describe("by_head", by_head)
    describe("head_first", head_first)
    print(f"head_first[1, :, 2] = {head_first[1, :, 2].tolist()}")

    offset = torch.tensor([10, 20, 30, 40], dtype=x.dtype, device=x.device)
    y = x + offset
    totals = y.sum(dim=-1)
    describe("offset", offset)
    describe("y", y)
    print(f"y[1, 2] = {y[1, 2].tolist()}")
    describe("totals", totals)
    print(f"totals = {totals.tolist()}")


if __name__ == "__main__":
    main()
