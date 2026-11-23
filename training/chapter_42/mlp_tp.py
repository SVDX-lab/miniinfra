"""Chapter 42 copy of the stable Chapter 36 MLP TP implementation.

Kept byte-compatible with chapter_35/mlp_tp.py and chapter_36/mlp_tp.py (the
ParallelMLP, its Column/Row parallel linears, the f/g communication operators,
weight split helpers, and the exact-norm clip).  The hybrid chapter reuses
every operator unchanged; only the GROUP handed in through ``group=`` changes
(it is now this rank's TP group inside the 3-D process-group layout).

The MLP math this file implements (unchanged from chapter 20).  For tp_size = N:

- ``gate_proj`` / ``up_proj`` are COLUMN parallel: the full weight
  ``[intermediate, hidden]`` is split along dim 0 into N shards of
  ``[intermediate/N, hidden]``.  The input is replicated (every rank holds the
  same activation); the output is a rank-local shard of the intermediate.
- ``down_proj`` is ROW parallel: the full weight ``[hidden, intermediate]`` is
  split along dim 1 into N shards of ``[hidden, intermediate/N]``.  Its input
  is the rank-local intermediate shard; the partial outputs are summed across
  the TP group with one AllReduce, which reproduces the full MLP output:

      MLP(x) = down(silu(gate(x)) * up(x))
             = sum_r  down_r(silu(gate_r(x)) * up_r(x))

The MLP is the ONLY thing this file shards.  Chapter 36's actual variable —
whether the Attention heads are ALSO sharded — lives in ``attention_tp.py``,
which reuses the linears, the f/g operators and the recorder defined here.

GRADIENT OWNERSHIP for the MLP block (the chapter-35 core, still load-bearing
in chapter 21 because every decoder layer keeps a sharded MLP)
---------------------------------------------------------
Per decoder layer, rank r computes locally:

    G_r = x @ W_gate_r^T ,  U_r = x @ W_up_r^T ,  H_r = silu(G_r) * U_r
    P_r = H_r @ W_down_r^T ,  y = AllReduce_sum(P_r)          (forward comm)

Forward AllReduce (the "g" op): forward = SUM, backward = identity, so every
rank receives the same dY.  Backward AllReduce (the "f" op): forward =
identity, backward = SUM of the MLP-input gradient, issued ONCE at the entry
of ParallelMLP (not once per gate/up branch).  Three ownership classes result:

1. MLP shard weights (gate/up/down of every layer): their gradients are
   rank-local.  dW_down_r = dY^T @ H_r and dW_gate_r / dW_up_r use only the
   rank-local H_r and the replicated x, so concatenating the shards over
   ranks reproduces the full weight gradient.  NO cross-rank communication.
2. The residual-stream gradient at each MLP input: rank r only computes its
   own contribution dX_r = dG_r @ W_gate_r + dU_r @ W_up_r, while the true
   gradient is sum_r dX_r.  The "f" op AllReduces exactly this tensor — this
   is the backward communication of MLP TP (one AllReduce per layer).  The
   forward AllReduce alone does NOT make the backward correct: the "g"
   backward is an identity, so the partial dX_r would silently propagate a
   wrong (partial) gradient into the attention path below the MLP.
3. Replicated weights (attention, norms, embedding/LM head): pure TP
   replicates the batch on every rank, so every rank computes the SAME full
   gradient for them from rank-identical activations (everything after a
   forward AllReduce — and before a backward AllReduce — is rank-identical).
   An extra AllReduce SUM would multiply them by tp_size, i.e. it would be
   WRONG, not just wasteful.  A gradient AllReduce for these weights is only
   needed when ranks hold DIFFERENT data (a DP dimension; chapter 27) — and
   even then across DP groups, never inside a TP group.

``compare_correctness.py`` verifies class 3 empirically: the replicated
gradients must be identical across ranks WITHOUT any communication.
"""

from __future__ import annotations

import time
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from mini_batch import PaddedBatch, batch_loss


GATE_KEY_SUFFIX = "mlp.gate_proj.weight"
UP_KEY_SUFFIX = "mlp.up_proj.weight"
DOWN_KEY_SUFFIX = "mlp.down_proj.weight"


def shard_error(message: str, weight_shape, dtype, device, tp_rank: int, tp_size: int) -> ValueError:
    """Build a ValueError that carries every fact needed to debug sharding."""

    return ValueError(
        f"{message}; full_weight_shape={tuple(weight_shape)}, dtype={dtype}, "
        f"device={device}, tp_rank={tp_rank}, tp_size={tp_size}"
    )


def _allreduce_(tensor: torch.Tensor, tp_size: int, group) -> None:
    """In-place SUM AllReduce; a no-op for tp_size == 1 (replicated baseline)."""

    if tp_size <= 1:
        return
    if not tensor.is_contiguous():
        raise ValueError(f"AllReduce needs a contiguous tensor, got strides {tensor.stride()}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)


class TPCommRecorder:
    """Accumulates the time spent inside MLP-TP AllReduce collectives.

    The collectives run inside forward and inside autograd backward, so
    wall-clock windows around ``loss.backward()`` cannot separate them.  Each
    call records a CUDA event pair; ``resolve()`` synchronizes ONCE per step
    and sums the elapsed times.  On CPU (gloo smoke tests)
    ``time.perf_counter`` is used instead.  All TP communication is exposed
    by construction: the forward AllReduce blocks the rest of forward and the
    backward AllReduce blocks the rest of backward.
    """

    def __init__(self) -> None:
        self._pending: list[tuple[str, object, object]] = []
        self.forward_ms = 0.0
        self.backward_ms = 0.0
        self.forward_count = 0
        self.backward_count = 0

    def reset(self) -> None:
        self._pending.clear()
        self.forward_ms = 0.0
        self.backward_ms = 0.0
        self.forward_count = 0
        self.backward_count = 0

    def all_reduce(self, tensor: torch.Tensor, tp_size: int, group, phase: str) -> None:
        if tp_size <= 1:
            return
        if tensor.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _allreduce_(tensor, tp_size, group)
            end.record()
            self._pending.append((phase, start, end))
        else:
            begin = time.perf_counter()
            _allreduce_(tensor, tp_size, group)
            self._accumulate(phase, (time.perf_counter() - begin) * 1000)

    def _accumulate(self, phase: str, elapsed_ms: float) -> None:
        if phase == "forward":
            self.forward_ms += elapsed_ms
            self.forward_count += 1
        elif phase == "backward":
            self.backward_ms += elapsed_ms
            self.backward_count += 1
        else:
            raise ValueError(f"unknown TP comm phase {phase!r}")

    def resolve(self) -> dict[str, float]:
        """Synchronize once, fold the pending event pairs into the totals."""

        if self._pending:
            torch.cuda.synchronize()
            for phase, start, end in self._pending:
                self._accumulate(phase, float(start.elapsed_time(end)))
            self._pending.clear()
        return {
            "tp_allreduce_forward_ms": self.forward_ms,
            "tp_allreduce_backward_ms": self.backward_ms,
            "tp_allreduce_forward_count": float(self.forward_count),
            "tp_allreduce_backward_count": float(self.backward_count),
        }


COMM_RECORDER = TPCommRecorder()


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """The Megatron "f" op: forward identity, backward AllReduce SUM.

    Applied once at the input of ParallelMLP.  In-place reduction of
    ``grad_output`` is safe here: autograd hands every node its own gradient
    tensor, and the MLP input (the post-LayerNorm activation) feeds no other
    consumer — the residual add uses the pre-LayerNorm activation.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, tp_size: int, group, recorder: TPCommRecorder):
        ctx.tp_size = tp_size
        ctx.group = group
        ctx.recorder = recorder
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = grad_output.contiguous()
        ctx.recorder.all_reduce(grad, ctx.tp_size, ctx.group, "backward")
        return grad, None, None, None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """The Megatron "g" op: forward AllReduce SUM, backward identity.

    The input is the fresh output of ``F.linear``; the clone keeps the
    autograd version-counter bookkeeping trivial.  The transient buffer is
    one MLP output in size and freed as soon as the reduction returns — this
    is part of TP's activation-memory cost and shows up in the peak numbers.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, tp_size: int, group, recorder: TPCommRecorder):
        output = input_.clone()
        recorder.all_reduce(output, tp_size, group, "forward")
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None, None


class ColumnParallelLinear(nn.Module):
    """Weight sharded along the OUTPUT dim; input replicated, output sharded.

    ``weight`` holds ``full[rank * local_out : (rank + 1) * local_out, :]``.
    With ``input_gradient_allreduce=True`` (standalone use) the backward of
    this layer AllReduces its input gradient.  ``ParallelMLP`` sets the flag
    to False and issues ONE shared AllReduce for both gate and up branches —
    two branches consuming the same input must not pay for the same sum
    twice.
    """

    def __init__(
        self,
        in_features: int,
        full_out_features: int,
        tp_rank: int,
        tp_size: int,
        input_gradient_allreduce: bool = True,
        group=None,
        recorder: TPCommRecorder | None = None,
    ) -> None:
        super().__init__()
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
        if full_out_features % tp_size != 0:
            raise shard_error(
                f"ColumnParallelLinear output dim {full_out_features} is not divisible by tp_size {tp_size}",
                (full_out_features, in_features),
                torch.float32,
                torch.device("cpu"),
                tp_rank,
                tp_size,
            )
        local_out = full_out_features // tp_size
        self.in_features = in_features
        self.full_out_features = full_out_features
        self.local_out_features = local_out
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.input_gradient_allreduce = input_gradient_allreduce
        self.group = group
        self.recorder = recorder if recorder is not None else COMM_RECORDER
        self.weight = nn.Parameter(torch.empty(local_out, in_features))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.input_gradient_allreduce:
            input_ = _CopyToTensorParallelRegion.apply(input_, self.tp_size, self.group, self.recorder)
        return F.linear(input_, self.weight)


class RowParallelLinear(nn.Module):
    """Weight sharded along the INPUT dim; input sharded, output AllReduce.

    ``weight`` holds ``full[:, rank * local_in : (rank + 1) * local_in]``.
    The forward partial outputs are summed across the TP group, which
    reconstructs the full ``y = x @ W^T``.
    """

    def __init__(
        self,
        full_in_features: int,
        out_features: int,
        tp_rank: int,
        tp_size: int,
        reduce_output: bool = True,
        group=None,
        recorder: TPCommRecorder | None = None,
    ) -> None:
        super().__init__()
        if tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {tp_size}")
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
        if full_in_features % tp_size != 0:
            raise shard_error(
                f"RowParallelLinear input dim {full_in_features} is not divisible by tp_size {tp_size}",
                (out_features, full_in_features),
                torch.float32,
                torch.device("cpu"),
                tp_rank,
                tp_size,
            )
        local_in = full_in_features // tp_size
        self.full_in_features = full_in_features
        self.out_features = out_features
        self.local_in_features = local_in
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.reduce_output = reduce_output
        self.group = group
        self.recorder = recorder if recorder is not None else COMM_RECORDER
        self.weight = nn.Parameter(torch.empty(out_features, local_in))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        partial = F.linear(input_, self.weight)
        if self.reduce_output:
            return _ReduceFromTensorParallelRegion.apply(partial, self.tp_size, self.group, self.recorder)
        return partial


class ParallelMLP(nn.Module):
    """SwiGLU MLP with the intermediate dimension sharded over the TP group.

    Forward per rank:  silu(x @ gate_r^T) * (x @ up_r^T) @ down_r^T, summed
    over ranks by the Row-parallel AllReduce.  Mathematically identical to
    the replicated MLP up to floating-point reduction order.  The entry "f"
    op is issued once (its backward AllReduces the MLP input gradient); the
    two column branches then run without further communication.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_rank: int,
        tp_size: int,
        group=None,
        recorder: TPCommRecorder | None = None,
    ) -> None:
        super().__init__()
        if intermediate_size % tp_size != 0:
            raise shard_error(
                f"ParallelMLP intermediate size {intermediate_size} is not divisible by tp_size {tp_size}",
                (intermediate_size, hidden_size),
                torch.float32,
                torch.device("cpu"),
                tp_rank,
                tp_size,
            )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.group = group
        self.recorder = recorder if recorder is not None else COMM_RECORDER
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, tp_rank, tp_size,
            input_gradient_allreduce=False, group=group, recorder=self.recorder,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, tp_rank, tp_size,
            input_gradient_allreduce=False, group=group, recorder=self.recorder,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, tp_rank, tp_size,
            reduce_output=True, group=group, recorder=self.recorder,
        )

    def intermediate_slice(self) -> slice:
        """This rank's slice of the full intermediate dim (for comparisons)."""

        local = self.intermediate_size // self.tp_size
        start = self.tp_rank * local
        return slice(start, start + local)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = _CopyToTensorParallelRegion.apply(
            hidden_states, self.tp_size, self.group, self.recorder
        )
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


def full_mlp_reference(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    """The replicated SwiGLU definition this chapter must reproduce:

    silu(x @ gate^T) * (x @ up^T), then @ down^T.  Kept next to the parallel
    implementation so the equivalence is one glance away.
    """

    return (F.silu(x @ gate_w.transpose(0, 1)) * (x @ up_w.transpose(0, 1))) @ down_w.transpose(0, 1)


def validate_shard_key(key: str, weight: torch.Tensor, tp_size: int) -> None:
    """Divisibility check for one MLP weight of a full state dict."""

    if key.endswith(GATE_KEY_SUFFIX) or key.endswith(UP_KEY_SUFFIX):
        split_dim, full_dim = 0, weight.shape[0]
    elif key.endswith(DOWN_KEY_SUFFIX):
        split_dim, full_dim = 1, weight.shape[1]
    else:
        return
    if full_dim % tp_size != 0:
        raise shard_error(
            f"state dict key {key!r} splits along dim {split_dim} of size {full_dim}, "
            f"which is not divisible by tp_size {tp_size}",
            tuple(weight.shape),
            weight.dtype,
            weight.device,
            -1,
            tp_size,
        )


def shard_weight(key: str, weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """This rank's shard of one MLP weight (gate/up dim0, down dim1)."""

    if key.endswith(GATE_KEY_SUFFIX) or key.endswith(UP_KEY_SUFFIX):
        local = weight.shape[0] // tp_size
        return weight[tp_rank * local : (tp_rank + 1) * local, :]
    if key.endswith(DOWN_KEY_SUFFIX):
        local = weight.shape[1] // tp_size
        return weight[:, tp_rank * local : (tp_rank + 1) * local]
    return weight


def split_tensor_parallel_weights(state_dict: dict, tp_size: int) -> list[dict]:
    """Build one state dict per TP rank from a full state dict.

    gate/up are split along dim 0, down along dim 1; every other tensor is
    copied unchanged into all ranks.  Raises a shard_error listing shape,
    dtype, device and tp_size when an MLP dimension is not divisible.
    """

    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    for key, weight in state_dict.items():
        validate_shard_key(key, weight, tp_size)
    return [
        {key: shard_weight(key, weight, tp_rank, tp_size).clone() for key, weight in state_dict.items()}
        for tp_rank in range(tp_size)
    ]


def assemble_tensor_parallel_weights(shards: Sequence[dict]) -> dict:
    """Inverse of ``split_tensor_parallel_weights`` (rank order preserved).

    Replicated keys must be bitwise identical across ranks; a mismatch means
    the TP contract (same input, same non-MLP weights) is already broken.
    """

    if not shards:
        raise ValueError("need at least one shard state dict")
    full: dict = {}
    for key in shards[0]:
        parts = [shard[key] for shard in shards]
        if key.endswith(GATE_KEY_SUFFIX) or key.endswith(UP_KEY_SUFFIX):
            full[key] = torch.cat(parts, dim=0).contiguous()
        elif key.endswith(DOWN_KEY_SUFFIX):
            full[key] = torch.cat(parts, dim=1).contiguous()
        else:
            for rank, part in enumerate(parts[1:], start=1):
                if not torch.equal(part, parts[0]):
                    raise ValueError(
                        f"replicated key {key!r} differs between rank 0 and rank {rank}; "
                        "the TP group does not share identical non-MLP weights"
                    )
            full[key] = parts[0].clone()
    return full


def is_tensor_parallel_shard(module: nn.Module) -> bool:
    return isinstance(module, (ColumnParallelLinear, RowParallelLinear))


def split_local_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition this rank's parameters into (replicated, shard) lists.

    Replicated parameters exist identically on every rank; shard parameters
    exist only as this rank's piece of the full weight.
    """

    shard_ids = set()
    for module in model.modules():
        if is_tensor_parallel_shard(module):
            shard_ids.update(id(parameter) for parameter in module.parameters())
    replicated: list[nn.Parameter] = []
    shard: list[nn.Parameter] = []
    for parameter in model.parameters():
        (shard if id(parameter) in shard_ids else replicated).append(parameter)
    return replicated, shard


def _first_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def clip_grad_norm_tp(
    model: nn.Module,
    max_norm: float,
    tp_size: int,
    group=None,
) -> torch.Tensor:
    """Gradient-norm clipping with the EXACT global norm under MLP TP.

    The full-model squared norm is  replicated_sq + sum_r shard_sq_r.  The
    replicated part is identical on every rank (ownership class 3) and must
    be counted ONCE, while the shard parts must be summed across ranks — so
    exactly one scalar AllReduce reconstructs the global norm.  A naive local
    clip would use a partial norm and scale differently from the reference
    path; a naive full AllReduce of all gradients would double-count the
    replicated part tp_size times.
    """

    device = _first_device(model)
    replicated, shard = split_local_parameters(model)
    replicated_sq = torch.zeros((), dtype=torch.float64, device=device)
    shard_sq = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in replicated:
        if parameter.grad is not None:
            replicated_sq += parameter.grad.double().pow(2).sum()
    for parameter in shard:
        if parameter.grad is not None:
            shard_sq += parameter.grad.double().pow(2).sum()
    if tp_size > 1:
        dist.all_reduce(shard_sq, op=dist.ReduceOp.SUM, group=group)
    total_norm = torch.sqrt(replicated_sq + shard_sq)
    clip_coef = float(max_norm) / (float(total_norm.item()) + 1e-6)
    if clip_coef < 1.0:
        for parameter in replicated + shard:
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coef)
    return total_norm


def _shard_dim_of_owner(model: nn.Module, parameter: torch.Tensor) -> int:
    for module in model.modules():
        if isinstance(module, ColumnParallelLinear) and module.weight is parameter:
            return 0
        if isinstance(module, RowParallelLinear) and module.weight is parameter:
            return 1
    raise ValueError("parameter does not belong to any parallel linear layer")


def gather_tensor_parallel_parameters(model: nn.Module, tp_size: int, group=None) -> dict:
    """Reassemble the FULL parameter state on every rank.

    Replicated parameters are taken from this rank; shard parameters are
    AllGathered in rank order and concatenated (gate/up dim0, down dim1).
    Used by correctness checks — never by the training hot path.
    """

    replicated, shard = split_local_parameters(model)
    del replicated
    gathered: dict[int, torch.Tensor] = {}
    for parameter in shard:
        if tp_size > 1:
            buffer = [torch.empty_like(parameter) for _ in range(tp_size)]
            dist.all_gather(buffer, parameter.detach(), group=group)
        else:
            buffer = [parameter.detach()]
        gathered[id(parameter)] = torch.cat(buffer, dim=_shard_dim_of_owner(model, parameter)).contiguous()
    state: dict = {}
    for name, parameter in model.named_parameters():
        state[name] = gathered[id(parameter)] if id(parameter) in gathered else parameter.detach().clone()
    return state


def tp_memory_ledger(model: nn.Module, optimizer: torch.optim.Optimizer | None = None, tp_size: int = 1) -> dict:
    """Theory-style byte ledger of what THIS rank owns.

    ``shard_parameter_bytes_full`` is what the MLP weights would occupy on a
    replicated rank; the gap to ``local_parameter_bytes`` is the parameter
    memory the TP shard saves.  Optimizer state follows the same ownership
    split: replicated parameters keep full moments, shard parameters keep
    moments for the local shard only.
    """

    replicated, shard = split_local_parameters(model)
    replicated_bytes = sum(parameter.numel() * parameter.element_size() for parameter in replicated)
    shard_local_bytes = sum(parameter.numel() * parameter.element_size() for parameter in shard)
    ledger = {
        "replicated_parameter_bytes": float(replicated_bytes),
        "shard_parameter_bytes_local": float(shard_local_bytes),
        "shard_parameter_bytes_full": float(shard_local_bytes * tp_size),
        "local_parameter_bytes": float(replicated_bytes + shard_local_bytes),
        "full_model_parameter_bytes": float(replicated_bytes + shard_local_bytes * tp_size),
        "tp_size": float(tp_size),
    }
    if optimizer is not None:
        shard_ids = {id(parameter) for parameter in shard}
        replicated_state = 0
        shard_state_local = 0
        for parameter, state in optimizer.state.items():
            state_bytes = sum(
                value.numel() * value.element_size() for value in state.values() if torch.is_tensor(value)
            )
            if id(parameter) in shard_ids:
                shard_state_local += state_bytes
            else:
                replicated_state += state_bytes
        ledger["replicated_optimizer_state_bytes"] = float(replicated_state)
        ledger["shard_optimizer_state_bytes_local"] = float(shard_state_local)
        ledger["local_optimizer_state_bytes"] = float(replicated_state + shard_state_local)
    return ledger


def to_device_batch(batch: PaddedBatch, device: torch.device) -> PaddedBatch:
    return PaddedBatch(
        input_ids=batch.input_ids.to(device, non_blocking=True),
        attention_mask=batch.attention_mask.to(device, non_blocking=True),
        labels=batch.labels.to(device, non_blocking=True),
        source_lines=batch.source_lines,
        lengths=batch.lengths,
        supervised_tokens_per_sample=batch.supervised_tokens_per_sample,
    )


def gpu_telemetry(rank: int) -> dict[str, float]:
    """Local-GPU power/temperature/clock; the physical GPU follows LOCAL_RANK."""

    import os
    import subprocess

    local_id = int(os.environ.get("LOCAL_RANK", rank))
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits", f"--id={local_id}"],
        check=True, capture_output=True, text=True,
    )
    power, temperature, clock = result.stdout.strip().split(",")
    return {"power_watts": float(power), "temperature_celsius": float(temperature), "sm_clock_mhz": float(clock)}


def tp_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    host_batch: PaddedBatch,
    loss_scale: float,
    device: torch.device,
    max_grad_norm: float,
    tp_size: int,
    rank: int,
    collect_telemetry: bool = True,
) -> dict[str, float]:
    """One MLP-TP optimizer step with every segment timed.

    TP replicates data: every rank of the group runs the SAME batch, so
    ``loss_scale`` is exactly 1.0 in this chapter (kept as an argument for
    interface parity with the DDP/ZeRO chapters and the DP*TP combination of
    chapter 27).  The local ``token_mean`` loss IS the global loss; no scalar
    AllReduce aggregates it.

    Communication inside the step (all of it exposed on the critical path):
    forward  - one output AllReduce per layer inside RowParallelLinear;
    backward - one input-gradient AllReduce per layer via the "f" op.
    After backward there is deliberately NO parameter-gradient AllReduce:
    shard gradients are rank-local (ownership class 1) and replicated
    gradients are already identical across ranks (ownership class 3) — see
    the module docstring.  ``clip_grad_norm_tp`` performs the single extra
    scalar AllReduce needed to reconstruct the exact global norm.

    Segment timing mirrors the DDP chapters: ``step_time_ms`` is wall clock
    between device synchronizations; forward/backward segments are
    event-timed INCLUSIVE of their AllReduce and reported again with the
    communication subtracted, so compute and communication are separately
    visible.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)
    recorder = COMM_RECORDER
    recorder.reset()
    if device.type != "cuda":
        start = time.perf_counter()
        h2d_begin = time.perf_counter()
        batch = to_device_batch(host_batch, device)
        h2d_ms = (time.perf_counter() - h2d_begin) * 1000
        forward_begin = time.perf_counter()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean") * loss_scale
        forward_incl_ms = (time.perf_counter() - forward_begin) * 1000
        backward_begin = time.perf_counter()
        loss.backward()
        backward_incl_ms = (time.perf_counter() - backward_begin) * 1000
        comm = recorder.resolve()
        forward_ms = forward_incl_ms - comm["tp_allreduce_forward_ms"]
        backward_ms = backward_incl_ms - comm["tp_allreduce_backward_ms"]
        loss_value = float(loss.item())
        optimizer_begin = time.perf_counter()
        grad_norm = clip_grad_norm_tp(model, max_grad_norm, tp_size)
        optimizer.step()
        optimizer_ms = (time.perf_counter() - optimizer_begin) * 1000
        step_ms = (time.perf_counter() - start) * 1000
    else:
        torch.cuda.synchronize(device)
        wall_start = time.perf_counter()
        events = {
            name: torch.cuda.Event(enable_timing=True)
            for name in ("h2d_start", "h2d_end", "forward_end", "backward_end", "optimizer_start", "optimizer_end")
        }
        events["h2d_start"].record()
        batch = to_device_batch(host_batch, device)
        events["h2d_end"].record()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean") * loss_scale
        events["forward_end"].record()
        loss.backward()
        events["backward_end"].record()
        # resolve() synchronizes the device once; the per-call event pairs of
        # the TP collectives are folded into comm totals here.
        comm = recorder.resolve()
        loss_value = float(loss.item())
        events["optimizer_start"].record()
        grad_norm = clip_grad_norm_tp(model, max_grad_norm, tp_size)
        optimizer.step()
        events["optimizer_end"].record()
        torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - wall_start) * 1000
        h2d_ms = float(events["h2d_start"].elapsed_time(events["h2d_end"]))
        forward_ms = float(events["h2d_end"].elapsed_time(events["forward_end"])) - comm["tp_allreduce_forward_ms"]
        backward_ms = float(events["forward_end"].elapsed_time(events["backward_end"])) - comm["tp_allreduce_backward_ms"]
        optimizer_ms = float(events["optimizer_start"].elapsed_time(events["optimizer_end"]))
    metrics = {
        "h2d_time_ms": h2d_ms,
        "forward_time_ms": forward_ms,
        "backward_time_ms": backward_ms,
        "optimizer_time_ms": optimizer_ms,
        "step_time_ms": step_ms,
        "local_loss": loss_value,
        "global_loss": loss_value,  # data is replicated: the local loss IS the global loss
        "gradient_norm_before_clip": float(grad_norm.item()),
        "physical_tokens": float(batch.physical_tokens),
        "non_padding_tokens": float(batch.non_padding_tokens),
        "supervised_tokens": float(batch.supervised_tokens),
        "padding_tokens": float(batch.padding_tokens),
        "padding_ratio": batch.padding_ratio,
        "batch_max_length": float(max(batch.lengths)),
        "samples": float(batch.batch_size),
    }
    metrics.update(comm)
    total_comm_ms = comm["tp_allreduce_forward_ms"] + comm["tp_allreduce_backward_ms"]
    metrics["tp_allreduce_total_ms"] = total_comm_ms
    metrics["tp_allreduce_exposed_ratio"] = (
        total_comm_ms / metrics["step_time_ms"] if metrics["step_time_ms"] > 0 else 0.0
    )
    seconds = metrics["step_time_ms"] / 1000
    metrics.update({
        "physical_tokens_per_second": batch.physical_tokens / seconds,
        "non_padding_tokens_per_second": batch.non_padding_tokens / seconds,
        "supervised_tokens_per_second": batch.supervised_tokens / seconds,
    })
    if device.type == "cuda" and collect_telemetry:
        metrics.update(gpu_telemetry(rank))
    return metrics
