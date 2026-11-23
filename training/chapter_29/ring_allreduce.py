"""Chapter 29 teaching module: ring AllReduce spelled out on point-to-point ops.

A ring AllReduce of a tensor with ``D`` elements across ``W`` ranks runs
``2 * (W - 1)`` steps and moves ``2 * (W - 1) / W * D`` elements per link:

- Reduce-Scatter phase (``W - 1`` steps): on step ``s`` rank ``r`` sends chunk
  ``(r - s) mod W`` and receives chunk ``(r - s - 1) mod W``, adding the
  received chunk into its local copy. Afterwards rank ``r`` owns the fully
  reduced chunk ``(r + 1) mod W``.
- AllGather phase (``W - 1`` steps): the same ring walk, but each received
  chunk REPLACES the local chunk instead of being added.

The implementation runs on CPU tensors with the Gloo backend, which is what
the smoke test uses to verify it against ``dist.all_reduce`` bit-for-bit on
integers. Gradient AllReduce in the real training path uses NCCL's tuned
ring/tree algorithms; this module exists to make the data volume and the
Reduce/Broadcast walk visible, not to replace NCCL.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def allreduce_volume(num_elements: int, element_size: int, world_size: int) -> dict[str, float]:
    """Per-link data volume and step count of a ring AllReduce."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    data_bytes = num_elements * element_size
    return {
        "tensor_bytes": float(data_bytes),
        "steps": float(2 * (world_size - 1)),
        "bytes_per_link": float(2 * (world_size - 1) / world_size * data_bytes),
        "bytes_per_link_ratio": float(2 * (world_size - 1) / world_size),
    }


def ring_all_reduce(tensor: torch.Tensor, rank: int, world_size: int, op: int = dist.ReduceOp.SUM) -> torch.Tensor:
    """Ring AllReduce on a 1-D CPU tensor; returns a new reduced tensor.

    Chunks must divide evenly; the teaching path does not pad. Additive
    reduction only (the operations the training path needs).
    """

    if tensor.dim() != 1:
        raise ValueError("ring_all_reduce expects a 1-D tensor")
    if tensor.numel() % world_size != 0:
        raise ValueError("numel must divide evenly across the ring")
    if op != dist.ReduceOp.SUM:
        raise ValueError("teaching ring supports SUM only")

    work = tensor.clone()
    chunks = list(work.chunk(world_size))
    send_buffer = torch.empty_like(chunks[0])

    # Phase 1: Reduce-Scatter. After W-1 steps rank r owns reduced chunk (r+1) mod W.
    for step in range(world_size - 1):
        send_chunk = (rank - step) % world_size
        recv_chunk = (rank - step - 1) % world_size
        send_req = dist.isend(chunks[send_chunk], dst=(rank + 1) % world_size)
        dist.recv(send_buffer, src=(rank - 1) % world_size)
        chunks[recv_chunk].add_(send_buffer)
        send_req.wait()

    # Phase 2: AllGather. Received fully reduced chunks replace local ones.
    for step in range(world_size - 1):
        send_chunk = (rank + 1 - step) % world_size
        recv_chunk = (rank - step) % world_size
        send_req = dist.isend(chunks[send_chunk], dst=(rank + 1) % world_size)
        dist.recv(send_buffer, src=(rank - 1) % world_size)
        chunks[recv_chunk].copy_(send_buffer)
        send_req.wait()

    return work
