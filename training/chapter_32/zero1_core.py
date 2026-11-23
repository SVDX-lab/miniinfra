"""Chapter 32 core: ZeRO-1 — optimizer states sharded across data-parallel ranks.

Every rank still holds the FULL FP32 parameter copy and the FULL gradient of
plain DDP. What changes is optimizer state ownership: the AdamW first and
second moment vectors are allocated only for this rank's shard of every
parameter (1/world_size of the state), and the parameter update runs only on
that shard. Because AdamW is elementwise, updating shard ``r`` with shard
``r`` of the gradient is BITWISE identical to updating the full tensor with
torch's AdamW — the state update never mixes elements.

After the update the owner ranks exchange fresh parameter shards with one
AllGather, so every replica starts the next step with identical parameters.

The update mirrors ``torch.optim.AdamW(foreach=False, fused=False,
amsgrad=False)`` operation for operation so the parity check against the
unsharded optimizer can be as strong as possible:

    param.mul_(1 - lr * weight_decay)            # decoupled weight decay
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = (exp_avg_sq.sqrt() / sqrt(1 - beta2**t)).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-lr / (1 - beta1**t))
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist


def shard_boundaries(numel: int, world_size: int, rank: int) -> tuple[int, int]:
    """Contiguous [start, end) of this rank's shard of a flat parameter."""

    base, remainder = divmod(numel, world_size)
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


class ShardedStateAdamW:
    """AdamW with 1/world_size of the state, updating only the local shard.

    The parameter tensors themselves stay full and replicated (that is ZeRO-1,
    not ZeRO-3): a rank writes ONLY its own shard during ``step`` and then
    ``all_gather_parameters`` refreshes every shard from its owner.
    """

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        world_size: int,
        rank: int,
        lr: float = 2e-5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.parameters = list(parameters)
        self.world_size = world_size
        self.rank = rank
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.exp_avg: list[torch.Tensor] = []
        self.exp_avg_sq: list[torch.Tensor] = []
        self.bounds: list[tuple[int, int]] = []
        for parameter in self.parameters:
            start, end = shard_boundaries(parameter.numel(), world_size, rank)
            self.bounds.append((start, end))
            self.exp_avg.append(torch.zeros(end - start, dtype=parameter.dtype, device=parameter.device))
            self.exp_avg_sq.append(torch.zeros(end - start, dtype=parameter.dtype, device=parameter.device))

    def state_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.exp_avg + self.exp_avg_sq)

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.beta1, self.beta2
        bias_correction1 = 1 - beta1 ** self.step_count
        bias_correction2 = 1 - beta2 ** self.step_count
        bias_correction2_sqrt = bias_correction2 ** 0.5
        step_size = self.lr / bias_correction1
        for parameter, (start, end), exp_avg, exp_avg_sq in zip(self.parameters, self.bounds, self.exp_avg, self.exp_avg_sq):
            if parameter.grad is None:
                continue
            flat_parameter = parameter.view(-1)
            grad_shard = parameter.grad.view(-1)[start:end]
            parameter_shard = flat_parameter[start:end]
            if self.weight_decay != 0:
                parameter_shard.mul_(1 - self.lr * self.weight_decay)
            exp_avg.lerp_(grad_shard, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_shard, grad_shard, value=1 - beta2)
            denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            parameter_shard.addcdiv_(exp_avg, denom, value=-step_size)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)


def all_gather_parameters(
    parameters: list[torch.nn.Parameter],
    world_size: int,
    rank: int,
    chunk_bytes: int = 128 * 1024 * 1024,
) -> tuple[float, int]:
    """Refresh every parameter from the owner ranks via chunked AllGathers.

    Chunks keep the transient gather buffers small (about ``2 * chunk_bytes``
    per rank); a single giant gather would hand the state saving straight
    back to the allocator. Returns (milliseconds, bytes sent per rank). After
    this call every rank's parameters are bitwise identical: each shard was
    written only by its owner.
    """

    if world_size == 1:
        return 0.0, 0
    on_cuda = parameters[0].is_cuda
    start = time.perf_counter()
    cuda_events = []

    pending: list[tuple[torch.nn.Parameter, int, int]] = []  # (parameter, start, end)
    pending_bytes = 0
    total_bytes = 0

    def padded_length(numel: int) -> int:
        return (numel + world_size - 1) // world_size

    def flush() -> None:
        nonlocal pending, pending_bytes, total_bytes
        if not pending:
            return
        # Pack with PADDED per-parameter lengths so every rank's buffer has the
        # same size even when a tensor's element count is not divisible by W.
        pieces = []
        for parameter, shard_start, shard_end in pending:
            piece = torch.zeros(padded_length(parameter.numel()), dtype=parameter.dtype, device=parameter.device)
            piece[: shard_end - shard_start] = parameter.detach().view(-1)[shard_start:shard_end]
            pieces.append(piece)
        local = torch.cat(pieces)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        total_bytes += local.numel() * local.element_size()
        if on_cuda:
            gather_start = torch.cuda.Event(enable_timing=True)
            gather_end = torch.cuda.Event(enable_timing=True)
            gather_start.record()
        dist.all_gather(gathered, local)
        if on_cuda:
            gather_end.record()
            cuda_events.append((gather_start, gather_end))
        src_offsets = [0] * world_size
        with torch.no_grad():
            for parameter, _shard_start, _shard_end in pending:
                numel = parameter.numel()
                flat_parameter = parameter.view(-1)
                for source_rank in range(world_size):
                    owner_start, owner_end = shard_boundaries(numel, world_size, source_rank)
                    length = owner_end - owner_start
                    flat_parameter[owner_start:owner_end] = gathered[source_rank][src_offsets[source_rank] : src_offsets[source_rank] + length]
                    src_offsets[source_rank] += padded_length(numel)
        pending = []
        pending_bytes = 0
        del local, gathered

    for parameter in parameters:
        shard_start, shard_end = shard_boundaries(parameter.numel(), world_size, rank)
        piece_bytes = (shard_end - shard_start) * parameter.element_size()
        if pending and pending_bytes + piece_bytes > chunk_bytes:
            flush()
        pending.append((parameter, shard_start, shard_end))
        pending_bytes += piece_bytes
    flush()

    if on_cuda:
        torch.cuda.synchronize()
        elapsed_ms = sum(float(s.elapsed_time(e)) for s, e in cuda_events)
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, total_bytes


def memory_ledger(parameters: list[torch.nn.Parameter], optimizer: ShardedStateAdamW, world_size: int) -> dict[str, float]:
    """Theoretical per-rank state ownership under ZeRO-1, in bytes."""

    full_parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    full_state_bytes = 2 * full_parameter_bytes
    return {
        "parameters_full_bytes": float(full_parameter_bytes),
        "gradients_full_bytes": float(full_parameter_bytes),
        "optimizer_state_local_bytes": float(optimizer.state_bytes()),
        "optimizer_state_full_bytes": float(full_state_bytes),
        "state_saving_bytes": float(full_state_bytes - optimizer.state_bytes()),
        "state_shard_fraction": optimizer.state_bytes() / full_state_bytes if full_state_bytes else 0.0,
    }
