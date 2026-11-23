from math import prod

import torch


def per_rank_bytes(elements, ranks, stage):
    if elements <= 0 or ranks <= 0 or elements % ranks or not 0 <= stage <= 3:
        raise ValueError("need positive divisible N, positive D and a stage in 0..3")
    # Order: low-precision parameters, low-precision gradients, master, m, v.
    widths = (2, 2, 4, 4, 4)
    starts = (3, 2, 1, 1, 1)
    return [width * (elements // ranks if stage >= start else elements)
            for width, start in zip(widths, starts, strict=True)]


def main():
    shape, ranks = (3, 4), 3
    count = prod(shape)
    full = torch.arange(count, dtype=torch.bfloat16).reshape(shape)
    shards = full.flatten().chunk(ranks)
    assert torch.equal(torch.cat(shards).reshape(shape), full)
    print(f"shape={shape}, dtype={full.dtype}, device={full.device}, N={count}, D={ranks}")
    print(f"stage-3 parameter shards={[part.tolist() for part in shards]}")
    totals = []
    for stage in range(4):
        groups = per_rank_bytes(count, ranks, stage)
        totals.append(sum(groups))
        print(f"stage={stage}: params/grad/master/m/v={groups} bytes, total={sum(groups)} bytes")
    assert totals == [192, 96, 80, 64]
    print("reassembled parameter equals original")


if __name__ == "__main__":
    main()
