"""Chapter 34 core: ZeRO-3 — the parameters are sharded too.

Chapters 17/18 shard the AdamW state and the gradients; every rank still
resides on a FULL parameter copy. This chapter flips the last piece: between
steps a rank holds only its 1/world_size shard of every parameter. The full
parameter exists only TRANSIENTLY, materialized by an AllGather right before
it is needed and freed (resharded) as soon as its use is over — the exact
lifecycle from the course outline:

    Sharded Parameter
      -> AllGather before Forward
      -> compute
      -> Reshard (free the full storage)
      -> on-demand AllGather before Backward
      -> Reduce-Scatter Gradient
      -> Sharded Optimizer Step

Implementation shape (deliberately different from Chapters 17/18):

Chapters 17/18 kept full ``nn.Parameter`` objects and only moved state/gradients
into shards. Once the PARAMETER itself is sharded, a plain module can no longer
run its own forward — the full weight must be a temporary. Each compute unit is
therefore a hand-written ``torch.autograd.Function`` whose

- ``forward``  AllGathers the resident shard into the full tensor, computes with
  it under the Function's no-grad region (so autograd CANNOT save the full
  tensor), and lets its storage die on return — the Reshard is real;
- ``backward`` re-AllGathers the full tensor ON DEMAND from the resident shard
  (it was freed!), computes input/weight gradients, and Reduce-Scatters the
  weight gradient so that only the local shard ever lands in ``.grad``.

Because the full weight is never saved by autograd, its peak lifetime is one
unit's compute, exactly like FSDP's per-unit flat parameter (without FSDP's
flat buckets, prefetch, and overlap — those remain out of scope and the cost
of small per-parameter collectives is honestly visible).

Shard layout (Chapter 32 boundary convention, kept for the whole 17->18->19
chain): the flat parameter of ``numel`` elements is cut by
``shard_boundaries`` into contiguous logical chunks; rank ``r``'s resident
buffer is ITS chunk zero-padded to ``ceil(numel / world_size)``. Padding means
a gathered stack of chunk buffers is in CHUNK-INTERLEAVED order, not logical
order — ``_gather_logical_flat`` reassembles the logical tensor with
``shard_boundaries`` (the same placement Chapter 32's AllGather uses to
refresh replicas), and the inverse Reduce-Scatter writes gradients straight
into the padded buffers.

Units provided:

- ``sharded_linear(x, weight_shard, weight_shape, bias_shard=None)`` — the
  workhorse; a Transformer block is composed from it with ordinary autograd
  ops (see ``sharded_swiglu_mlp``).
- ``sharded_embedding(input_ids, weight_shard, num_embeddings, embedding_dim)``
- ``sharded_scale(x, weight_shard, weight_shape)`` — broadcast multiply for
  RMSNorm-style vector weights, so the tiny demo models shard EVERY parameter.

The resident shard is a plain padded flat ``nn.Parameter``.
``ShardedLeafAdamW`` runs the exact Chapter 32 AdamW formulas on those leaves,
so a sharded update is bitwise identical to torch AdamW on the full
parameters. ``unshard_parameter`` rebuilds the full tensor for checks and
exports; ``memory_ledger_zero3`` reports the resident-state fractions.

Gloo/CPU: ``dist.reduce_scatter`` has no Gloo backend, so the ring walk from
Chapter 33 is reused. ``GATHER_CALLS``/``LAST_FULL_GATHER`` expose the
lifecycle mechanically: the smoke test asserts that the full tensor's storage
is already freed when ``forward`` returns and that backward issues one
on-demand AllGather per unit.
"""

from __future__ import annotations

import weakref
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def shard_boundaries(numel: int, world_size: int, rank: int) -> tuple[int, int]:
    """Contiguous [start, end) of this rank's shard of a flat parameter."""

    base, remainder = divmod(numel, world_size)
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def padded_shard_length(numel: int, world_size: int) -> int:
    return (numel + world_size - 1) // world_size


# Lifecycle observability. ``GATHER_CALLS`` counts AllGathers by phase
# (forward = pre-compute gather, backward = on-demand re-gather);
# ``LAST_FULL_GATHER`` holds a weak reference to the most recently gathered
# full tensor so tests can assert it was FREED (resharded) at the right time.
GATHER_CALLS = {"forward": 0, "backward": 0}
REDUCE_SCATTER_CALLS = {"backward": 0}
LAST_FULL_GATHER = {"phase": None, "weakref": None}


def reset_lifecycle_counters() -> None:
    GATHER_CALLS["forward"] = 0
    GATHER_CALLS["backward"] = 0
    REDUCE_SCATTER_CALLS["backward"] = 0
    LAST_FULL_GATHER["phase"] = None
    LAST_FULL_GATHER["weakref"] = None


def _gather_logical_flat(shard: torch.Tensor, numel: int, world_size: int, phase: str) -> torch.Tensor:
    """AllGather the padded chunks and reassemble the LOGICAL flat tensor.

    The gathered chunks arrive in chunk-interleaved order (each rank's buffer
    is padded to ``ceil(numel / W)``), so chunk ``r`` lands at logical
    positions ``[start_r, end_r)`` — the inverse of ``make_parameter_shard``.
    The result has exactly ``numel`` elements; the caller must NOT keep it
    alive past its compute window: keeping it would silently turn ZeRO-3 back
    into parameter replication.
    """

    GATHER_CALLS[phase] += 1
    if world_size == 1:
        full = shard.detach()[:numel].clone()
    else:
        chunks = [torch.empty_like(shard.detach()) for _ in range(world_size)]
        dist.all_gather(chunks, shard.detach())
        full = torch.empty(numel, dtype=shard.dtype, device=shard.device)
        for owner in range(world_size):
            owner_start, owner_end = shard_boundaries(numel, world_size, owner)
            full[owner_start:owner_end] = chunks[owner][: owner_end - owner_start]
    LAST_FULL_GATHER["phase"] = phase
    LAST_FULL_GATHER["weakref"] = weakref.ref(full)
    return full


def ring_reduce_scatter(full: torch.Tensor, output: torch.Tensor, world_size: int, rank: int) -> None:
    """Reduce-Scatter a 1-D tensor via the ring walk on backends without the collective.

    Chapter 33 implementation reused verbatim: after ``W - 1`` steps rank ``r``
    owns the fully reduced chunk ``(r + 1) mod W``; one neighbor exchange maps
    it to chunk ``r``. Used on the Gloo/CPU path; NCCL calls
    ``dist.reduce_scatter`` directly.
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
    owned = (rank + 1) % world_size
    send_req = dist.isend(work[owned * chunk : (owned + 1) * chunk].contiguous(), dst=(rank + 1) % world_size)
    incoming = torch.empty_like(output)
    dist.recv(incoming, src=(rank - 1) % world_size)
    send_req.wait()
    output.copy_(incoming)


def _reduce_scatter_to_shard(
    grad_full: torch.Tensor,
    numel: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Reduce-Scatter one full gradient into this rank's padded shard buffer.

    ``dist.reduce_scatter`` needs equal-sized chunks, but ``shard_boundaries``
    distributes a remainder unevenly, so EVERY input chunk is a padded buffer
    that carries logical elements ``[start_r, end_r)`` — chunk ``r`` of the
    sum is then exactly this rank's shard of the global gradient, for any
    ``numel``/``world_size`` pair. Padding positions receive the sum of zeros
    and the optimizer ignores them. Returns a ``[padded]`` tensor.
    """

    REDUCE_SCATTER_CALLS["backward"] += 1
    padded = padded_shard_length(numel, world_size)
    shard = torch.zeros(padded, dtype=dtype, device=device)
    if world_size == 1:
        shard[:numel] = grad_full
        return shard
    flat = grad_full.reshape(-1)
    input_list = []
    for owner in range(world_size):
        owner_start, owner_end = shard_boundaries(numel, world_size, owner)
        buffer = torch.zeros(padded, dtype=dtype, device=device)
        buffer[: owner_end - owner_start] = flat[owner_start:owner_end]
        input_list.append(buffer)
    if dist.is_initialized() and dist.get_backend() == "gloo":
        ring_reduce_scatter(torch.cat(input_list), shard, world_size, rank)
    else:
        dist.reduce_scatter(shard, input_list, op=dist.ReduceOp.SUM)
    return shard


def make_parameter_shard(
    full: torch.Tensor,
    world_size: int,
    rank: int,
    requires_grad: bool = True,
) -> nn.Parameter:
    """Cut one full tensor into this rank's persistent padded flat shard.

    This Parameter IS the resident state between steps; the full tensor is
    gone after this call.
    """

    flat = full.detach().reshape(-1)
    numel = flat.numel()
    start, end = shard_boundaries(numel, world_size, rank)
    shard = torch.zeros(padded_shard_length(numel, world_size), dtype=flat.dtype, device=flat.device)
    shard[: end - start] = flat[start:end]
    return nn.Parameter(shard, requires_grad=requires_grad)


def unshard_parameter(shard: torch.Tensor, shape: Sequence[int], world_size: int) -> torch.Tensor:
    """Rebuild the full tensor from resident shards (checks and exports only)."""

    numel = 1
    for dimension in shape:
        numel *= dimension
    if world_size == 1:
        return shard.detach()[:numel].reshape(tuple(shape)).clone()
    chunks = [torch.empty_like(shard.detach()) for _ in range(world_size)]
    dist.all_gather(chunks, shard.detach())
    full = torch.empty(numel, dtype=shard.dtype, device=shard.device)
    for owner in range(world_size):
        owner_start, owner_end = shard_boundaries(numel, world_size, owner)
        full[owner_start:owner_end] = chunks[owner][: owner_end - owner_start]
    return full.reshape(tuple(shape)).clone()


class _ShardedLinearFunction(torch.autograd.Function):
    """Linear unit with the full ZeRO-3 parameter lifecycle.

    forward:  AllGather(weight shard) -> x @ W^T (+ b) -> free W  [Reshard]
    backward: on-demand AllGather -> grad_x, grad_W(full) -> Reduce-Scatter
    """

    @staticmethod
    def forward(ctx, x, weight_shard, bias_shard, weight_shape, world_size, rank):
        out_features, in_features = weight_shape
        weight_full = _gather_logical_flat(weight_shard, out_features * in_features, world_size, phase="forward")
        weight = weight_full.view(out_features, in_features)
        # Autograd is disabled inside Function.forward: the matmul below cannot
        # save `weight`, so the gathered storage dies when this function
        # returns. That is the Reshard step, and it is real.
        y = x.matmul(weight.t())
        if bias_shard is not None:
            bias_full = _gather_logical_flat(bias_shard, out_features, world_size, phase="forward")
            y = y + bias_full
        ctx.save_for_backward(x, weight_shard, bias_shard)
        ctx.lifecycle_meta = (weight_shape, world_size, rank)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight_shard, bias_shard = ctx.saved_tensors
        weight_shape, world_size, rank = ctx.lifecycle_meta
        out_features, in_features = weight_shape
        needs_x, needs_w, needs_b = ctx.needs_input_grad[:3]
        weight_full = None
        if needs_x or needs_w:
            # On-demand AllGather: the forward's full weight was freed.
            weight_full = _gather_logical_flat(weight_shard, out_features * in_features, world_size, phase="backward")
        grad_x = None
        if needs_x:
            grad_x = grad_y.matmul(weight_full.view(out_features, in_features))
        grad_w_shard = None
        if needs_w:
            grad_w_full = (
                grad_y.reshape(-1, out_features).t().matmul(x.reshape(-1, in_features)).reshape(-1)
            )
            grad_w_shard = _reduce_scatter_to_shard(
                grad_w_full, out_features * in_features, world_size, rank, grad_y.dtype, grad_y.device
            )
        grad_b_shard = None
        if needs_b:
            grad_b_full = grad_y.reshape(-1, out_features).sum(dim=0)
            grad_b_shard = _reduce_scatter_to_shard(
                grad_b_full, out_features, world_size, rank, grad_y.dtype, grad_y.device
            )
        del weight_full
        return grad_x, grad_w_shard, grad_b_shard, None, None, None


def sharded_linear(
    x: torch.Tensor,
    weight_shard: torch.Tensor,
    weight_shape: tuple[int, int],
    bias_shard: torch.Tensor | None = None,
    world_size: int | None = None,
    rank: int | None = None,
) -> torch.Tensor:
    """y = x @ W^T + b where W and b exist only as flat shards on this rank.

    x: [batch, ..., in_features]; weight_shape: (out_features, in_features);
    the shards are the padded flat ``nn.Parameter`` objects from
    ``make_parameter_shard``. A present bias always has out_features logical
    elements (standard Linear semantics).
    """

    if world_size is None or rank is None:
        world_size, rank = effective_world_rank()
    return _ShardedLinearFunction.apply(
        x, weight_shard, bias_shard, tuple(weight_shape), world_size, rank
    )


class _ShardedEmbeddingFunction(torch.autograd.Function):
    """Embedding lookup over a sharded weight (the tied lm_head reuses
    ``sharded_linear`` over the SAME shard — no double counting)."""

    @staticmethod
    def forward(ctx, input_ids, weight_shard, num_embeddings, embedding_dim, world_size, rank):
        weight_full = _gather_logical_flat(
            weight_shard, num_embeddings * embedding_dim, world_size, phase="forward"
        )
        weight = weight_full.view(num_embeddings, embedding_dim)
        output = F.embedding(input_ids, weight)
        ctx.save_for_backward(input_ids, weight_shard)
        ctx.lifecycle_meta = (num_embeddings, embedding_dim, world_size, rank)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_ids, weight_shard = ctx.saved_tensors
        num_embeddings, embedding_dim, world_size, rank = ctx.lifecycle_meta
        grad_w_shard = None
        if ctx.needs_input_grad[1]:
            weight_full = _gather_logical_flat(
                weight_shard, num_embeddings * embedding_dim, world_size, phase="backward"
            )
            grad_w_full = torch.zeros(
                num_embeddings * embedding_dim, dtype=grad_output.dtype, device=grad_output.device
            )
            # grad w.r.t. a gathered row = the upstream gradient of that row.
            # The flat storage is viewed as [num_embeddings, embedding_dim] so
            # index_add_ scatters row gradients to the right logical rows.
            grad_w_full.view(num_embeddings, embedding_dim).index_add_(
                0, input_ids.reshape(-1), grad_output.reshape(-1, embedding_dim)
            )
            grad_w_shard = _reduce_scatter_to_shard(
                grad_w_full,
                num_embeddings * embedding_dim,
                world_size,
                rank,
                grad_output.dtype,
                grad_output.device,
            )
            del weight_full
        # input_ids are integers: no gradient flows to them.
        return None, grad_w_shard, None, None, None, None


def sharded_embedding(
    input_ids: torch.Tensor,
    weight_shard: torch.Tensor,
    num_embeddings: int,
    embedding_dim: int,
    world_size: int | None = None,
    rank: int | None = None,
) -> torch.Tensor:
    if world_size is None or rank is None:
        world_size, rank = effective_world_rank()
    return _ShardedEmbeddingFunction.apply(
        input_ids, weight_shard, num_embeddings, embedding_dim, world_size, rank
    )


class _ShardedScaleFunction(torch.autograd.Function):
    """y = x * w with a broadcast vector weight (RMSNorm-style), sharded.

    Lets the tiny demo models shard EVERY parameter, including norm vectors,
    so "parameters are also sharded" holds without exceptions.
    """

    @staticmethod
    def forward(ctx, x, weight_shard, weight_shape, world_size, rank):
        weight_full = _gather_logical_flat(weight_shard, weight_shape[0], world_size, phase="forward")
        weight = weight_full.view(weight_shape)
        y = x * weight
        ctx.save_for_backward(x, weight_shard)
        ctx.lifecycle_meta = (weight_shape, world_size, rank)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight_shard = ctx.saved_tensors
        weight_shape, world_size, rank = ctx.lifecycle_meta
        needs_x, needs_w = ctx.needs_input_grad[:2]
        weight_full = None
        if needs_x or needs_w:
            weight_full = _gather_logical_flat(weight_shard, weight_shape[0], world_size, phase="backward")
        grad_x = None
        if needs_x:
            grad_x = grad_y * weight_full.view(weight_shape)
        grad_w_shard = None
        if needs_w:
            reduce_dims = tuple(range(grad_y.ndim - 1))
            grad_w_full = (grad_y * x).sum(dim=reduce_dims)
            grad_w_shard = _reduce_scatter_to_shard(
                grad_w_full, weight_shape[0], world_size, rank, grad_y.dtype, grad_y.device
            )
        del weight_full
        return grad_x, grad_w_shard, None, None, None


def sharded_scale(
    x: torch.Tensor,
    weight_shard: torch.Tensor,
    weight_shape: tuple[int, ...],
    world_size: int | None = None,
    rank: int | None = None,
) -> torch.Tensor:
    if world_size is None or rank is None:
        world_size, rank = effective_world_rank()
    return _ShardedScaleFunction.apply(x, weight_shard, tuple(weight_shape), world_size, rank)


def sharded_swiglu_mlp(
    x: torch.Tensor,
    gate_shard: torch.Tensor,
    up_shard: torch.Tensor,
    down_shard: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    world_size: int | None = None,
    rank: int | None = None,
) -> torch.Tensor:
    """One Qwen3 SwiGLU MLP block built only from sharded units.

    The nonlinear glue (silu, multiply) is ordinary autograd: each sharded
    linear gathers/resards/reduces-scatters ITS OWN weight, and backward runs
    the units in reverse order — deterministic collective order on every rank.
    """

    if world_size is None or rank is None:
        world_size, rank = effective_world_rank()
    gate = F.silu(sharded_linear(x, gate_shard, (intermediate_size, hidden_size), None, world_size, rank))
    up = sharded_linear(x, up_shard, (intermediate_size, hidden_size), None, world_size, rank)
    return sharded_linear(gate * up, down_shard, (hidden_size, intermediate_size), None, world_size, rank)


def effective_world_rank() -> tuple[int, int]:
    """(world_size, rank) from the default process group, or (1, 0) single-process."""

    if dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


class ShardedLeafAdamW:
    """AdamW over resident shard leaves — the Chapter 34 optimizer step.

    Each leaf IS the padded flat shard, and its ``.grad`` (delivered by the
    unit's Reduce-Scatter) is the padded shard of the global gradient. The
    update mirrors ``torch.optim.AdamW(foreach=False, fused=False,
    amsgrad=False)`` operation for operation, so updating shard ``r`` with
    shard ``r`` of the gradient is bitwise identical to updating the full
    tensor (AdamW never mixes elements):

        param.mul_(1 - lr * weight_decay)            # decoupled weight decay
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / sqrt(1 - beta2**t)).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-lr / (1 - beta1**t))

    After the step every owner's shard is fresh; there is NO post-step
    parameter AllGather (that was ZeRO-1/2). The next forward's AllGather
    refreshes replicas by construction. The optimizer only ever touches
    ``[0, end - start)`` of each leaf; the padded tail stays zero.
    """

    def __init__(
        self,
        parameters: list[nn.Parameter],
        lr: float = 2e-5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.exp_avg: list[torch.Tensor] = []
        self.exp_avg_sq: list[torch.Tensor] = []
        for parameter in self.parameters:
            self.exp_avg.append(torch.zeros_like(parameter))
            self.exp_avg_sq.append(torch.zeros_like(parameter))

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
        for parameter, exp_avg, exp_avg_sq in zip(self.parameters, self.exp_avg, self.exp_avg_sq):
            if parameter.grad is None:
                raise ValueError("every sharded parameter needs a gradient before the step")
            if self.weight_decay != 0:
                parameter.mul_(1 - self.lr * self.weight_decay)
            exp_avg.lerp_(parameter.grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(parameter.grad, parameter.grad, value=1 - beta2)
            denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            parameter.addcdiv_(exp_avg, denom, value=-step_size)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)


def resident_bytes(parameters: Sequence[nn.Parameter]) -> dict[str, int]:
    """Bytes actually resident on this rank, by category."""

    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    gradient_bytes = sum(
        (parameter.grad.numel() * parameter.grad.element_size()) if parameter.grad is not None else 0
        for parameter in parameters
    )
    return {"parameters": parameter_bytes, "gradients": gradient_bytes}


def memory_ledger_zero3(
    parameters: Sequence[nn.Parameter],
    optimizer: ShardedLeafAdamW,
    full_parameter_numels: Sequence[int],
    element_size: int,
) -> dict[str, float]:
    """Theoretical per-rank ownership under ZeRO-3, in bytes.

    The outline's ledger: parameters, gradients, and optimizer state each
    collapse to 1/world_size, i.e. 16 bytes/param / W for FP32 AdamW training.
    """

    full_parameter_bytes = sum(numel for numel in full_parameter_numels) * element_size
    resident = resident_bytes(parameters)
    state_bytes = optimizer.state_bytes()
    full_state_bytes = 2 * full_parameter_bytes
    return {
        "parameters_full_bytes": float(full_parameter_bytes),
        "parameters_shard_bytes": float(resident["parameters"]),
        "gradients_full_bytes": float(full_parameter_bytes),
        "gradients_shard_bytes": float(resident["gradients"]),
        "optimizer_state_full_bytes": float(full_state_bytes),
        "optimizer_state_shard_bytes": float(state_bytes),
        "param_shard_fraction": resident["parameters"] / full_parameter_bytes if full_parameter_bytes else 0.0,
        "grad_shard_fraction": resident["gradients"] / full_parameter_bytes if full_parameter_bytes else 0.0,
        "state_shard_fraction": state_bytes / full_state_bytes if full_state_bytes else 0.0,
        "resident_total_bytes": float(resident["parameters"] + resident["gradients"] + state_bytes),
        "full_replica_total_bytes": float(2 * full_parameter_bytes + full_state_bytes),
    }
