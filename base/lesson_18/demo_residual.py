import math

import torch


def branches(x, weight, epsilon):
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("expected a nonempty channel axis")
    width = x.shape[-1]
    if weight.shape != (width, width):
        raise ValueError("branch must preserve the channel width")
    if not x.is_floating_point() or weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("expected one floating dtype and one device")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    def norm(values):
        return values / (values.square().mean(dim=-1, keepdim=True) + epsilon).sqrt()

    pre_delta = norm(x) @ weight.T
    pre = x + pre_delta
    post_sum = x + x @ weight.T
    post = norm(post_sum)
    return pre_delta, pre, post_sum, post


def display(row):
    return [round(value, 6) for value in row.tolist()]


def main():
    options = dict(dtype=torch.float64, device="cpu")
    epsilon = 1e-6
    x = torch.tensor([3, 4], **options)
    weight = torch.tensor([[2, 0], [0, 0]], **options)
    pre_delta, pre, post_sum, post = branches(x, weight, epsilon)
    print(f"X: {display(x)}, shape={tuple(x.shape)}, {x.dtype}, {x.device}")
    print(f"pre branch update: {display(pre_delta)}")
    print(f"pre output: {display(pre)}")
    print(f"post sum before norm: {display(post_sum)}")
    print(f"post output: {display(post)}")
    _, pre_zero, _, post_zero = branches(x, torch.zeros_like(weight), epsilon)
    print(f"zero branch pre: {display(pre_zero)}; post: {display(post_zero)}")
    expected_pre = torch.tensor([3 + 6 / math.sqrt(12.5 + epsilon), 4], **options)
    expected_post = torch.tensor([9, 4], **options) / math.sqrt(48.5 + epsilon)
    torch.testing.assert_close(pre, expected_pre, atol=1e-12, rtol=0)
    torch.testing.assert_close(post, expected_post, atol=1e-12, rtol=0)
    torch.testing.assert_close(pre_zero, x, atol=0, rtol=0)
    print("layout and identity checks: pass")


if __name__ == "__main__":
    main()
