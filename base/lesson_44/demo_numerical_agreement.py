import torch


def same_float32_bits(left, right):
    if left.dtype != torch.float32 or right.dtype != torch.float32 or left.shape != right.shape:
        raise ValueError("bit comparison here requires matching FP32 tensors")
    return torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))


def main():
    a, b, c = torch.tensor([2**24, 1, -(2**24)], dtype=torch.float32)
    lhs, rhs = (a + b) + c, a + (b + c)
    print(f"FP32 reduction: left={lhs.item():.1f}, right={rhs.item():.1f}, real sum=1")
    print(f"FP64 reference: {(a.double() + b.double() + c.double()).item():.1f}")

    left = torch.tensor([1., 1.], dtype=torch.float32)
    right = torch.tensor([1., 1. + 2**-20], dtype=left.dtype)
    close = torch.allclose(left, right, atol=1e-6, rtol=0)
    bits = same_float32_bits(left, right)
    print(f"close outputs: atol=1e-6, rtol=0, close={close}, bits_equal={bits}")
    print(f"greedy IDs: left={int(left.argmax())}, right={int(right.argmax())}")
    zeros = (left.new_tensor([0.]), left.new_tensor([-0.]))
    print(f"signed zero: value_equal={torch.equal(*zeros)}, bits_equal={same_float32_bits(*zeros)}")
    assert (lhs.item(), rhs.item(), close, bits) == (0., 1., True, False)


if __name__ == "__main__":
    main()
