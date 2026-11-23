import torch


def bytes_of(tensors):
    return sum(value.numel() * value.element_size() for value in tensors)


def main():
    parameters = [torch.empty((2, 3), dtype=torch.float32), torch.empty(4, dtype=torch.float32)]
    names = ("parameters", "gradients", "first moment", "second moment")
    groups = [parameters] + [[torch.zeros_like(value) for value in parameters] for _ in range(3)]
    count = sum(value.numel() for value in parameters)
    print(f"shapes={[tuple(value.shape) for value in parameters]}, dtype=float32, N={count}")
    totals = []
    for name, tensors in zip(names, groups, strict=True):
        size = bytes_of(tensors)
        totals.append(size)
        print(f"{name}: {size} bytes")
    print(f"four-group total={sum(totals)} bytes; bytes/parameter={sum(totals) // count}")
    assert totals == [40] * 4
    qwen = 596_049_920
    size = 16 * qwen
    print(f"Qwen3 count only: N={qwen}, total={size} bytes, {size / 2**30:.3f} GiB")


if __name__ == "__main__":
    main()
