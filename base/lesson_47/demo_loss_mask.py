import math

import torch
from torch.nn import functional as F


IGNORE = -100


def targets_for(ids, valid, assistant, *, assistant_only):
    if (ids.ndim != 2 or ids.shape[1] < 2 or ids.dtype != torch.int64
            or ids.shape != valid.shape or ids.shape != assistant.shape
            or valid.dtype != torch.bool or assistant.dtype != torch.bool
            or ids.device != valid.device or ids.device != assistant.device):
        raise ValueError("expected aligned token IDs and boolean masks [batch, sequence]")
    chosen = valid[:, :-1] & valid[:, 1:]
    if assistant_only:
        chosen = chosen & assistant[:, 1:]
    return ids[:, 1:].masked_fill(~chosen, IGNORE)


def main():
    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 0]], dtype=torch.int64)
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    assistant = torch.tensor([[0, 0, 1, 1, 0, 1, 0]], dtype=torch.bool)
    logits = torch.zeros((1, 7, 6), dtype=torch.float64)
    logits[0, 1, 2] = math.log(5)
    for name, only in (("full", False), ("assistant-only", True)):
        targets = targets_for(ids, valid, assistant, assistant_only=only)
        losses = F.cross_entropy(logits[:, :-1].transpose(1, 2), targets,
                                 ignore_index=IGNORE, reduction="none")
        active = targets != IGNORE
        values = [round(x, 6) for x in losses[active].tolist()]
        assert active.sum().item() == (3 if only else 5)
        assert torch.count_nonzero(losses[~active]) == 0
        print(f"{name}: labels={targets[0].tolist()}, valid={active.sum().item()}")
        print(f"{name}: active losses={values}")
    print("shape: logits=(1, 7, 6) float64; labels=(1, 6) int64, cpu")


if __name__ == "__main__":
    main()
