import argparse
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException


def worker(rank, root, inject):
    dist.init_process_group("gloo", init_method=(root / "store").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        assert dist.get_rank() == rank and dist.get_process_group_ranks(dist.group.WORLD) == [0, 1]
        try:
            if inject and rank == 1:
                raise ValueError("injected input validation failure")
            failed = 0
        except ValueError:
            failed = 1

        local = torch.tensor([failed], dtype=torch.int64)
        statuses = [torch.empty_like(local) for _ in range(2)]
        dist.all_gather(statuses, local)
        errors = [member for member, status in enumerate(statuses) if status.item()]
        if errors:
            raise RuntimeError(f"preflight failed on rank {errors[0]}")

        report = root / "rank0.txt"
        if rank == 0:
            with report.open("x") as output:
                output.write("rank 0 published its report\n")
        dist.barrier()
        assert report.read_text() == "rank 0 published its report\n"
    finally:
        dist.destroy_process_group()
    assert not dist.is_initialized()


def main():
    parser = argparse.ArgumentParser(description="Check rank ownership, safe preflight and teardown")
    parser.add_argument("--inject-failure", action="store_true")
    args = parser.parse_args()
    with TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        try:
            mp.spawn(worker, args=(root, args.inject_failure), nprocs=2, join=True)
        except ProcessRaisedException as error:
            if not args.inject_failure or "preflight failed on rank 1" not in str(error):
                raise
            assert not (root / "rank0.txt").exists()
            print("rank 1 validation failed; ranks agreed to skip publication", flush=True)
            return 1
        print((root / "rank0.txt").read_text(), end="")
    print("both ranks read after barrier; groups destroyed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
