from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def transfer(chunk, rank, size):
    if chunk.ndim != 1 or not chunk.is_contiguous() or chunk.dtype != torch.float32:
        raise ValueError("expected contiguous FP32 ring chunk")
    incoming = torch.empty_like(chunk)
    next_rank, previous = (rank + 1) % size, (rank - 1) % size
    works = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, chunk, next_rank),
        dist.P2POp(dist.irecv, incoming, previous),
    ])
    for work in works:
        work.wait()
    return incoming


def ring(chunks, rank, size):
    for step in range(size - 1):
        sent = (rank - step) % size
        received = (rank - step - 1) % size
        chunks[received].add_(transfer(chunks[sent], rank, size))
    owner = (rank + 1) % size
    owned = chunks[owner].clone()

    for step in range(size - 1):
        sent = (rank + 1 - step) % size
        received = (rank - step) % size
        chunks[received].copy_(transfer(chunks[sent], rank, size))
    return owner, owned, torch.cat(chunks)


def worker(rank, root):
    size = 3
    dist.init_process_group("gloo", init_method=(root / "store").as_uri(),
                            rank=rank, world_size=size, timeout=timedelta(seconds=20))
    try:
        assert dist.get_rank() == rank and dist.get_process_group_ranks(dist.group.WORLD) == [0, 1, 2]
        x = torch.arange(1, 7, dtype=torch.float32) + 10 * rank
        expected = 3 * torch.arange(11, 17, dtype=torch.float32)
        chunks = [chunk.clone() for chunk in x.chunk(size)]
        assert all(c.shape == (2,) and c.device.type == "cpu" for c in chunks)
        owner, owned, result = ring(chunks, rank, size)
        assert torch.equal(owned, expected.chunk(size)[owner])
        assert torch.equal(result, expected)
        report = root / "rank0.txt"
        if rank == 0:
            message = 2 * x.element_size()
            amount = 2 * (size - 1) * message
            report.write_text(
                f"ranks={size}, local shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}\n"
                f"rank0 owns chunk {owner} after reduce-scatter: {owned.tolist()}\n"
                f"all ranks reconstruct {result.tolist()}\n"
                f"per rank: rounds=2+2, message={message} bytes, sent={amount}, received={amount}\n"
            )
        dist.barrier()
        assert report.is_file()
    finally:
        dist.destroy_process_group()
    assert not dist.is_initialized()


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        mp.spawn(worker, args=(root,), nprocs=3, join=True)
        print((root / "rank0.txt").read_text(), end="")
    print("rank-0 report checked; groups destroyed")


if __name__ == "__main__":
    main()
