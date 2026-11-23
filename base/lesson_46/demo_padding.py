import torch


def pad_rows(rows, pad_id):
    if not rows or any(not row for row in rows):
        raise ValueError("need at least one row and at least one token per row")
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad_id, dtype=torch.int64)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for index, row in enumerate(rows):
        ids[index, :len(row)] = torch.tensor(row, dtype=ids.dtype)
        valid[index, :len(row)] = True
    return ids, valid


def attend(values, valid):
    if (values.ndim != 2 or values.shape != valid.shape or valid.dtype != torch.bool
            or not values.is_floating_point() or values.device != valid.device):
        raise ValueError("expected matching values and boolean mask [batch, sequence]")
    if values.shape[1] == 0 or not valid[:, 0].all():
        raise ValueError("need a valid first token for each sequence")
    length = values.shape[1]
    causal = torch.ones((length, length), dtype=torch.bool, device=values.device).tril()
    allowed = causal[None] & valid[:, None, :]
    scores = values.new_zeros(allowed.shape).masked_fill(~allowed, -torch.inf)
    weighted = scores.softmax(dim=-1) @ values.unsqueeze(-1)
    return weighted.squeeze(-1).masked_fill(~valid, 0), allowed


def main():
    ids, valid = pad_rows([[1, 2, 3], [4, 5]], pad_id=99)
    values = ids.double()
    out, allowed = attend(values, valid)
    target_valid = valid[:, :-1] & valid[:, 1:]
    print(f"ids={ids.tolist()}, dtype={ids.dtype}, shape={tuple(ids.shape)}")
    print(f"attention_mask={valid.int().tolist()}")
    print(f"allowed shape={tuple(allowed.shape)}, shorter row={allowed[1].int().tolist()}")
    print(f"outputs={out.tolist()}; supervised pairs={target_valid.int().tolist()}")
    expected = torch.tensor([[1, 1.5, 2], [4, 4.5, 0]], dtype=out.dtype)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    changed = attend(values.masked_fill(~valid, -999), valid)[0]
    torch.testing.assert_close(changed, expected, rtol=0, atol=0)
    print("changing the pad value leaves the valid outputs unchanged")


if __name__ == "__main__":
    main()
