import torch


def embedding_lookup(table, ids):
    if table.ndim != 2 or table.dtype != torch.float32 or not table.is_cpu:
        raise ValueError("table must be a 2-D CPU float32 tensor")
    if ids.dtype != torch.int64 or ids.device != table.device:
        raise ValueError("ids must be an int64 tensor on the table's device")
    if ((ids < 0) | (ids >= table.shape[0])).any():
        raise ValueError("Token ID is outside the table")
    return table[ids]


def lm_head(hidden, weight):
    if weight.ndim != 2 or weight.dtype != torch.float32 or not weight.is_cpu:
        raise ValueError("weight must be a 2-D CPU float32 tensor")
    if hidden.ndim == 0 or hidden.shape[-1] != weight.shape[1]:
        raise ValueError("hidden features must match the projection weight")
    if hidden.dtype != weight.dtype or hidden.device != weight.device:
        raise ValueError("hidden and weight must have the same dtype and device")
    return hidden @ weight.T


def main():
    table = torch.tensor([[1, 0], [0, 2], [1, 1]], dtype=torch.float32, device="cpu")
    ids = torch.tensor([[2, 0]], dtype=torch.int64, device="cpu")
    print(f"table {tuple(table.shape)}: {table.tolist()}")
    print(f"ids {tuple(ids.shape)}: {ids.tolist()}")

    looked_up = embedding_lookup(table, ids)
    one_hot = torch.eye(table.shape[0], dtype=table.dtype, device=table.device)[ids]
    by_product = one_hot @ table
    assert torch.equal(looked_up, by_product)
    print(f"one-hot {tuple(one_hot.shape)}: {one_hot.tolist()}")
    print(f"lookup {tuple(looked_up.shape)}: {looked_up.tolist()}")
    print(f"one-hot @ table: {by_product.tolist()}")

    hidden = torch.tensor([[[2, 1], [-1, 2]]], dtype=table.dtype, device=table.device)
    untied_weight = torch.tensor(
        [[1, 1], [1, 0], [0, 1]], dtype=table.dtype, device=table.device
    )
    tied_weight = table
    untied = lm_head(hidden, untied_weight)
    tied = lm_head(hidden, tied_weight)

    print(f"hidden {tuple(hidden.shape)}: {hidden.tolist()}")
    print(f"untied logits {tuple(untied.shape)}: {untied.tolist()}")
    print(f"tied logits {tuple(tied.shape)}: {tied.tolist()}")
    print(f"same tensor: {tied_weight is table}")
    print(f"weight elements: untied={table.numel() + untied_weight.numel()}, "
          f"tied={table.numel()}")


if __name__ == "__main__":
    main()
