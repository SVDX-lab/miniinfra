import argparse
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, size, use_cuda, root):
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        assert torch.cuda.current_device() == rank
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(
        backend, init_method=(root / "rendezvous").as_uri(), rank=rank,
        world_size=size, timeout=timedelta(seconds=25),
    )
    try:
        assert dist.get_rank() == rank and dist.get_world_size() == size
        members = dist.get_process_group_ranks(dist.group.WORLD)
        assert members == list(range(size))
        value = torch.tensor([rank + 1.], dtype=torch.float32, device=device)
        assert value.shape == (1,)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if use_cuda:
            torch.cuda.synchronize(device)
        expected = sum(range(1, size + 1))
        assert value.item() == expected
        report = root / "rank0.txt"
        if rank == 0:
            report.write_text(f"backend={backend}, members={members}, bindings={device.type}\n"
                              f"all_reduce: shape=(1,), dtype=float32, sum={expected:.1f}\n")
        dist.barrier(device_ids=[rank] if use_cuda else None)
        assert report.is_file()
    finally:
        dist.destroy_process_group()
    assert not dist.is_initialized()


def main():
    parser = argparse.ArgumentParser(description="Run the minimal rank-bound Process Group example")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--ranks", type=int)
    args = parser.parse_args()
    size = args.ranks if args.ranks is not None else (1 if args.cuda else 2)
    if not 1 <= size <= 4:
        parser.error("--ranks must be between one and four")
    if args.cuda and (not dist.is_nccl_available() or torch.cuda.device_count() < size):
        parser.error("the requested ranks need one visible CUDA device each and NCCL")
    with TemporaryDirectory() as workspace:
        root = Path(workspace).resolve()
        mp.spawn(worker, args=(size, args.cuda, root), nprocs=size, join=True)
        print((root / "rank0.txt").read_text(), end="")
        print("rank-0 report visible after barrier; all groups destroyed")


if __name__ == "__main__":
    main()
