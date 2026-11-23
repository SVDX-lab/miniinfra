import torch


def row_column_product(left, right):
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Both inputs must be matrices (2-D tensors)")
    rows, shared = left.shape
    right_rows, columns = right.shape
    if shared != right_rows:
        raise ValueError("Left columns must match right rows")
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise ValueError("Both inputs must have dtype torch.float32")
    if left.device.type != "cpu" or right.device.type != "cpu":
        raise ValueError("Both inputs must be on CPU")

    result = torch.zeros((rows, columns), dtype=left.dtype, device=left.device)
    for row in range(rows):
        for column in range(columns):
            for feature in range(shared):
                result[row, column] += left[row, feature] * right[feature, column]
    return result


def main():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cpu")
    weight = x.new_tensor([[1, 0, -1], [2, 1, 0], [0, 1, 1], [-1, 0, 2]])
    bias = x.new_tensor([10, -1, 2, 0])
    right = weight.transpose(0, 1)
    print(f"x: {tuple(x.shape)}, {x.dtype}, {x.device}")
    print(f"weight: {tuple(weight.shape)}, right: {tuple(right.shape)}")
    print(f"bias: {tuple(bias.shape)}, values={bias.tolist()}")

    raw = row_column_product(x, right)
    if not torch.equal(raw, x @ right):
        raise AssertionError("Row-column result differs from @ for these inputs")
    print(f"x @ weight.T: {raw.tolist()} (@ agrees)")

    y = raw + bias
    print(f"y: {tuple(y.shape)}, values={y.tolist()}")
    batch_x = x.reshape(2, 1, 3)
    batch_y = batch_x @ right + bias
    print(f"batched: {tuple(batch_x.shape)} -> {tuple(batch_y.shape)}")
    print(f"batched values: {batch_y.tolist()}")


if __name__ == "__main__":
    main()
