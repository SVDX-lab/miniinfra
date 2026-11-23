"""Chapter 33 engine copied verbatim: ZeRO-2 — the Chapter 34 baseline arm.

Building on Chapter 32 (sharded AdamW state), the full gradient never
survives the backward: each parameter's gradient is Reduce-Scattered across
the data-parallel ranks, and every rank keeps ONLY its own shard — exactly
the shard its optimizer state lives on. The remaining parameters stay fully
replicated (still not ZeRO-3). Chapter 34 changes exactly one thing on top of
this engine: whether the parameters themselves also live as shards.

Mechanics per step, per parameter (all parameters participate):

1. After ``backward``, the parameter gradient exists as a full tensor on each
   rank (autograd just produced it — the FULL tensor is briefly unavoidable).
2. ``dist.reduce_scatter`` sums the ranks' gradients elementwise and delivers
   rank ``r`` only chunk ``r``. For world size 2 this is the SAME elementwise
   sum an AllReduce computes, so the local shard is bitwise equal to the
   corresponding slice of the Chapter 29 AllReduced gradient.
3. The full gradient's storage is released immediately (``param.grad=None``);
   the shard lands in a persistent per-parameter buffer allocated with the
   optimizer state.
4. The sharded AdamW step runs on the shard (identical elementwise formulas
   to Chapter 32), then a chunked AllGather refreshes all parameters.

Non-divisible tensor sizes are padded with zeros on every rank, so the sums
over padding stay zero and the shard's padded tail is ignored. Nearly all
Qwen3 tensors are already divisible by 2, in which case the Reduce-Scatter
reads the gradient storage DIRECTLY through views — no copy. Real ZeRO
implementations flatten parameters into large buckets to amortize the
per-call latency; this teaching path calls Reduce-Scatter once per parameter
and reports that cost honestly.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist


def shard_boundaries(numel: int, world_size: int, rank: int) -> tuple[int, int]:
    base, remainder = divmod(numel, world_size)
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def padded_shard_length(numel: int, world_size: int) -> int:
    return (numel + world_size - 1) // world_size


class ShardedGradAdamW:
    """ZeRO-2 optimizer: 1/world_size of gradients AND of the AdamW state.

    The gradient shard buffers are allocated once and reused every step; they
    are filled by ``reduce_scatter_gradients``. Parameter updates touch only
    the local shard; ``all_gather_parameters`` (Chapter 32) refreshes the
    replicas afterwards.
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
        self.grad_shards: list[torch.Tensor] = []
        self.exp_avg: list[torch.Tensor] = []
        self.exp_avg_sq: list[torch.Tensor] = []
        self.bounds: list[tuple[int, int]] = []
        for parameter in self.parameters:
            start, end = shard_boundaries(parameter.numel(), world_size, rank)
            length = end - start
            self.bounds.append((start, end))
            # Buffers are padded to the collective chunk length; the optimizer
            # only ever touches the [0, length) slice that belongs to this rank.
            padded = padded_shard_length(parameter.numel(), world_size)
            device = parameter.device
            self.grad_shards.append(torch.zeros(padded, dtype=parameter.dtype, device=device))
            self.exp_avg.append(torch.zeros(padded, dtype=parameter.dtype, device=device))
            self.exp_avg_sq.append(torch.zeros(padded, dtype=parameter.dtype, device=device))

    def state_bytes(self) -> int:
        tensors = self.grad_shards + self.exp_avg + self.exp_avg_sq
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.beta1, self.beta2
        bias_correction1 = 1 - beta1 ** self.step_count
        bias_correction2 = 1 - beta2 ** self.step_count
        bias_correction2_sqrt = bias_correction2 ** 0.5
        step_size = self.lr / bias_correction1
        for parameter, (start, end), grad_shard, exp_avg, exp_avg_sq in zip(
            self.parameters, self.bounds, self.grad_shards, self.exp_avg, self.exp_avg_sq,
        ):
            length = end - start
            grad_shard = grad_shard[:length]
            exp_avg = exp_avg[:length]
            exp_avg_sq = exp_avg_sq[:length]
            parameter_shard = parameter.view(-1)[start:end]
            if self.weight_decay != 0:
                parameter_shard.mul_(1 - self.lr * self.weight_decay)
            exp_avg.lerp_(grad_shard, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_shard, grad_shard, value=1 - beta2)
            denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            parameter_shard.addcdiv_(exp_avg, denom, value=-step_size)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)


def ring_reduce_scatter(full: torch.Tensor, output: torch.Tensor, world_size: int, rank: int) -> None:
    """Reduce-Scatter a 1-D tensor via the ring walk on backends without the collective.

    Reduce-Scatter phase of a ring AllReduce: ``W - 1`` steps, after which
    rank ``r`` owns the fully reduced chunk ``(r + 1) mod W``. Used for Gloo
    CPU smoke tests; the NCCL path calls ``dist.reduce_scatter`` directly.
    """

    chunk = full.numel() // world_size
    work = full.clone()
    for step in range(world_size - 1):
        send_chunk = (rank - step) % world_size
        recv_chunk = (rank - step - 1) % world_size
        send_req = dist.isend(work[send_chunk * chunk : (send_chunk + 1) * chunk], dst=(rank + 1) % world_size)
        incoming = torch.empty_like(output)
        dist.recv(incoming, src=(rank - 1) % world_size)
        work[recv_chunk * chunk : (recv_chunk + 1) * chunk].add_(incoming)
        send_req.wait()
    # After the walk rank r owns chunk (r + 1) % W, but dist.reduce_scatter
    # semantics want rank r to own chunk r: one neighbor exchange fixes it.
    owned = (rank + 1) % world_size
    send_req = dist.isend(work[owned * chunk : (owned + 1) * chunk].contiguous(), dst=(rank + 1) % world_size)
    incoming = torch.empty_like(output)
    dist.recv(incoming, src=(rank - 1) % world_size)
    send_req.wait()
    output.copy_(incoming)


def reduce_scatter_gradients(
    parameters: list[torch.nn.Parameter],
    optimizer: ShardedGradAdamW,
    world_size: int,
) -> tuple[float, int]:
    """Reduce-Scatter every parameter gradient into the optimizer's shard buffers.

    Returns (milliseconds, full-gradient bytes consumed as input per rank).
    The full gradient storage is released parameter by parameter as soon as
    its Reduce-Scatter input has been taken.
    """

    if world_size == 1:
        total_local = 0
        for parameter, (start, end), grad_shard in zip(parameters, optimizer.bounds, optimizer.grad_shards):
            if parameter.grad is None:
                raise ValueError("every parameter needs a gradient before Reduce-Scatter")
            grad_shard[: end - start].copy_(parameter.grad.view(-1)[start:end])
            parameter.grad = None
            total_local += (end - start) * parameter.element_size()
        return 0.0, total_local
    total_bytes = 0
    on_cuda = parameters[0].is_cuda
    start = torch.cuda.Event(enable_timing=True) if on_cuda else None
    end = torch.cuda.Event(enable_timing=True) if on_cuda else None
    wall_start = time.perf_counter()
    if start is not None:
        start.record()
    for parameter, grad_shard in zip(parameters, optimizer.grad_shards):
        if parameter.grad is None:
            raise ValueError("every parameter needs a gradient before Reduce-Scatter")
        numel = parameter.numel()
        total_bytes += numel * parameter.element_size()
        if numel % world_size == 0:
            chunk = numel // world_size
            input_list = [parameter.grad.view(-1)[r * chunk : (r + 1) * chunk] for r in range(world_size)]
        else:
            padded_length = padded_shard_length(numel, world_size) * world_size
            flat = torch.zeros(padded_length, dtype=parameter.dtype, device=parameter.device)
            flat[:numel] = parameter.grad.view(-1)
            chunk = padded_length // world_size
            input_list = [flat[r * chunk : (r + 1) * chunk] for r in range(world_size)]
        if dist.get_backend() == "gloo":
            full = torch.cat(input_list)
            ring_reduce_scatter(full, grad_shard, world_size, dist.get_rank())
        else:
            dist.reduce_scatter(grad_shard, input_list, op=dist.ReduceOp.SUM)
        parameter.grad = None
    if end is not None:
        end.record()
        torch.cuda.synchronize()
        elapsed = float(start.elapsed_time(end))
    else:
        elapsed = (time.perf_counter() - wall_start) * 1000
    return elapsed, total_bytes


def all_gather_parameters(
    parameters: list[torch.nn.Parameter],
    world_size: int,
    rank: int,
    chunk_bytes: int = 128 * 1024 * 1024,
) -> tuple[float, int]:
    """Refresh every parameter from its owner via chunked AllGathers."""

    if world_size == 1:
        return 0.0, 0
    on_cuda = parameters[0].is_cuda
    start = time.perf_counter()
    cuda_events = []
    pending: list[tuple[torch.nn.Parameter, int, int]] = []
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


def memory_ledger(parameters: list[torch.nn.Parameter], optimizer: ShardedGradAdamW) -> dict[str, float]:
    """Theoretical per-rank state ownership under ZeRO-2, in bytes."""

    full_parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    full_state_bytes = 2 * full_parameter_bytes
    return {
        "parameters_full_bytes": float(full_parameter_bytes),
        "gradients_full_bytes": float(full_parameter_bytes),
        "gradients_local_bytes": float(sum(t.numel() * t.element_size() for t in optimizer.grad_shards)),
        "optimizer_state_local_bytes": float(sum(t.numel() * t.element_size() for t in optimizer.exp_avg + optimizer.exp_avg_sq)),
        "optimizer_state_full_bytes": float(full_state_bytes),
        "state_shard_fraction": float(sum(t.numel() * t.element_size() for t in optimizer.exp_avg + optimizer.exp_avg_sq)) / full_state_bytes if full_state_bytes else 0.0,
        "grad_shard_fraction": float(sum(g.numel() * g.element_size() for g in optimizer.grad_shards)) / full_parameter_bytes if full_parameter_bytes else 0.0,
    }


def clip_grad_norm_sharded(
    optimizer: ShardedGradAdamW,
    max_grad_norm: float,
    world_size: int,
) -> tuple[float, float]:
    """Gradient clipping when only shards exist.

    The full gradient norm is the root-sum-square over the disjoint shards,
    so each rank reduces one scalar and scales its own shard — mirroring
    ``torch.nn.utils.clip_grad_norm_`` (total norm, coefficient clamped at 1).
    Returns (global_preclip_norm, milliseconds for the scalar AllReduce).
    """

    import time as _time

    local_sq = torch.zeros((), dtype=torch.float32, device=optimizer.grad_shards[0].device)
    for parameter, grad_shard, (start, end) in zip(optimizer.parameters, optimizer.grad_shards, optimizer.bounds):
        local_sq += grad_shard[: end - start].pow(2).sum()
    total_sq = local_sq.clone()
    on_cuda = local_sq.is_cuda
    if on_cuda:
        begin = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        begin.record()
    else:
        begin = _time.perf_counter()
    dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
    if on_cuda:
        finish.record()
        torch.cuda.synchronize()
        elapsed_ms = float(begin.elapsed_time(finish))
    else:
        elapsed_ms = (_time.perf_counter() - begin) * 1000
    total_norm = float(total_sq.sqrt())
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    clip_coef_clamped = max(0.0, min(1.0, clip_coef))
    if clip_coef_clamped < 1.0:
        with torch.no_grad():
            for grad_shard in optimizer.grad_shards:
                grad_shard.mul_(clip_coef_clamped)
    return total_norm, elapsed_ms

def zero2_train_step(
    model: torch.nn.Module,
    optimizer: ShardedGradAdamW,
    host_batch,
    loss_scale: float,
    device: torch.device,
    max_grad_norm: float,
    world_size: int,
    rank: int,
    collect_telemetry: bool = True,
) -> dict[str, float]:
    """One ZeRO-2 optimizer step with per-segment event timing.

    Segments mirror the Chapter 29 body; the communication segment
    (``grad_allreduce_ms``) here is the per-parameter Reduce-Scatter chain,
    and ``param_allgather_ms`` the post-step parameter refresh.
    """

    import subprocess

    from mini_batch import batch_loss
    from ddp_core import to_device_batch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    rs_end = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    h2d_start.record()
    batch = to_device_batch(host_batch, device)
    h2d_end.record()
    logits = model(batch.input_ids, attention_mask=batch.attention_mask)
    loss = batch_loss(logits, batch.labels, "token_mean") * loss_scale
    forward_end.record()
    loss.backward()
    backward_end.record()
    rs_ms, rs_bytes = reduce_scatter_gradients(list(model.parameters()), optimizer, world_size)
    rs_end.record()
    # After Reduce-Scatter no full parameter.grad exists; clip in shard space.
    grad_norm, clip_sync_ms = clip_grad_norm_sharded(optimizer, max_grad_norm, world_size)
    optimizer.step()
    allgather_ms, allgather_bytes = all_gather_parameters(list(model.parameters()), world_size, rank)
    optimizer_end.record()
    torch.cuda.synchronize(device)
    step_ms = (time.perf_counter() - wall_start) * 1000
    metrics = {
        "h2d_time_ms": float(h2d_start.elapsed_time(h2d_end)),
        "forward_time_ms": float(h2d_end.elapsed_time(forward_end)),
        "backward_time_ms": float(forward_end.elapsed_time(backward_end)),
        "grad_allreduce_ms": rs_ms,
        "reduce_scatter_bytes": float(rs_bytes),
        "param_allgather_ms": allgather_ms,
        "param_allgather_bytes": float(allgather_bytes),
        "optimizer_time_ms": float(rs_end.elapsed_time(optimizer_end)) - allgather_ms,
        "step_time_ms": step_ms,
        "local_loss": float(loss.detach().item()),
        "gradient_norm_before_clip": float(grad_norm),
        "clip_scalar_sync_ms": clip_sync_ms,
        "physical_tokens": float(batch.physical_tokens),
        "non_padding_tokens": float(batch.non_padding_tokens),
        "supervised_tokens": float(batch.supervised_tokens),
        "padding_tokens": float(batch.padding_tokens),
        "padding_ratio": batch.padding_ratio,
        "batch_max_length": float(max(batch.lengths)),
        "samples": float(batch.batch_size),
    }
    seconds = step_ms / 1000
    metrics.update({
        "physical_tokens_per_second": batch.physical_tokens / seconds,
        "non_padding_tokens_per_second": batch.non_padding_tokens / seconds,
        "supervised_tokens_per_second": batch.supervised_tokens / seconds,
    })
    if collect_telemetry:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits", f"--id={rank}"],
            check=True, capture_output=True, text=True,
        )
        power, temperature, clock = result.stdout.strip().split(",")
        metrics.update({"power_watts": float(power), "temperature_celsius": float(temperature), "sm_clock_mhz": float(clock)})
    return metrics
