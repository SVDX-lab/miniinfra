import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class TokenRows(Dataset):
    rows = ([11, 12], [21, 22, 23, 24], [31], [41, 42, 43], [51, 52])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return index, self.rows[index]


class ShortFirst(Sampler):
    def __init__(self, dataset):
        self.order = sorted(range(len(dataset)), key=lambda i: len(dataset[i][1]))

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


def collate(items):
    indices = [index for index, _ in items]
    lengths = [len(row) for _, row in items]
    ids = torch.zeros((len(items), max(lengths)), dtype=torch.int64)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, ((_, row), length) in enumerate(zip(items, lengths, strict=True)):
        ids[i, :length] = torch.tensor(row, dtype=ids.dtype)
        mask[i, :length] = True
    return indices, lengths, ids, mask


def main():
    dataset = TokenRows()
    sampler = ShortFirst(dataset)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, collate_fn=collate,
                        num_workers=0, drop_last=False)
    seen = []
    print(f"dataset size={len(dataset)}, sampler order={list(sampler)}")
    for step, (indices, lengths, ids, valid) in enumerate(loader):
        seen.extend(indices)
        assert valid.sum(dim=1).tolist() == lengths
        print(f"step={step}: indices={indices}, shape={tuple(ids.shape)}, lengths={lengths}")
        print(f"  ids={ids.tolist()}, mask={valid.int().tolist()}")
    assert seen == [2, 0, 4, 3, 1]
    print("seen once; ids=int64, mask=bool, device=cpu")


if __name__ == "__main__":
    main()
