from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def exercise(rank):
    cell = torch.tensor([rank + 1.], dtype=torch.float32)
    assert cell.shape == (1,) and cell.device.type == "cpu"
    merged = cell.clone()
    dist.all_reduce(merged)
    assert merged.tolist() == [3.]

    ordered = [torch.empty_like(cell) for _ in range(2)]
    dist.all_gather(ordered, cell)
    assert [tensor.item() for tensor in ordered] == [1., 2.]

    inputs = torch.tensor([rank + 1., 10. * (rank + 1)], dtype=cell.dtype)
    shard = torch.empty_like(cell)
    assert inputs.numel() == 2 * shard.numel()
    dist.reduce_scatter_tensor(shard, inputs)
    assert shard.item() == (3. if rank == 0 else 30.)

    shared = torch.full_like(cell, 42. if rank == 0 else 0.)
    dist.broadcast(shared, src=0)
    assert shared.item() == 42.

    message = torch.full_like(cell, 7. if rank == 0 else 0.)
    if rank == 0:
        dist.send(message, dst=1)
    else:
        dist.recv(message, src=0)
    assert message.item() == 7.


def worker(rank, root):
    dist.init_process_group("gloo", init_method=(root / "store").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        assert dist.get_rank() == rank and dist.get_process_group_ranks(dist.group.WORLD) == [0, 1]
        exercise(rank)
        report = root / "rank0.txt"
        if rank == 0:
            report.write_text(
                "AllReduce: both [3.0]\nAllGather: both [1.0, 2.0]\n"
                "ReduceScatter: rank0 [3.0], rank1 [30.0]\n"
                "Broadcast: both [42.0]\nP2P: rank1 received [7.0] from rank0\n"
            )
        dist.barrier()
        assert report.is_file()
    finally:
        dist.destroy_process_group()
    assert not dist.is_initialized()


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        mp.spawn(worker, args=(root,), nprocs=2, join=True)
        print((root / "rank0.txt").read_text(), end="")
    print("Gloo ranks=[0, 1]; float32 CPU; rank-0 report checked; groups destroyed")


if __name__ == "__main__":
    main()
