"""Chapter 29 core: hand-written Distributed Data Parallel.

Every rank owns a full FP32 model copy, full gradients, and a full AdamW
state. The global batch plan is the Chapter 22 shuffle built once per epoch;
each rank executes only its member shard ``members[rank::world_size]`` of
every batch, so the union of the rank shards is the global batch.

The loss reduction stays ``token_mean`` over the GLOBAL batch: each rank
scales its local ``token_mean`` loss by ``supervised_local / supervised_global``
before ``backward``, and the post-backward AllReduce SUMS the partial
gradients. Sum over ranks of ``grad((S_r / S_global) * loss_sum_r)`` is
exactly ``grad(sum_r loss_sum_r / S_global)`` — the gradient of the global
``token_mean`` objective — so no division by world size happens anywhere.
Chapter 24 proved the same supervised-token weighting for micro-batches; this
chapter reuses it across ranks.

The gradient AllReduce runs strictly after the whole backward pass, so its
entire duration is exposed communication on the critical path. Chapter 30
changes WHEN the AllReduce starts, not the data volume.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import socket
import subprocess
import time
from typing import Callable, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from assistant_loss import EncodedConversation
from mini_batch import PaddedBatch, batch_loss, pad_samples
from qwen3_model import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, load_handwritten_model, resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import seed_everything


MODEL_ID = DEFAULT_MODEL_ID
MODEL_REVISION = DEFAULT_MODEL_REVISION


def load_tokenizer_and_model(device: torch.device) -> tuple[Qwen3Tokenizer, torch.nn.Module]:
    directory = resolve_model_directory(MODEL_ID, MODEL_REVISION)
    return Qwen3Tokenizer(directory), load_handwritten_model(directory, device, dtype=torch.float32)


def parameter_count_bytes() -> int:
    """Parameter count read from safetensors headers without loading weights."""

    from safetensors import safe_open

    directory = resolve_model_directory(MODEL_ID, MODEL_REVISION)
    total = 0
    for shard in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():
                count = 1
                for dimension in handle.get_slice(key).get_shape():
                    count *= dimension
                total += count
    return total


def setup_distributed(rank: int | None = None, world_size: int | None = None, backend: str = "nccl") -> tuple[int, int, torch.device]:
    """Process-group setup; each rank binds its LOCAL CUDA device.

    Rank and world size come from arguments, or from RANK/WORLD_SIZE when
    omitted (env://-style launchers). The CUDA device always follows
    LOCAL_RANK so the same code works under mp.spawn on one node and under
    torchrun across nodes; when LOCAL_RANK is absent (mp.spawn), the global
    rank is used, which is identical on a single node.
    """

    if rank is None:
        rank = int(os.environ["RANK"])
    if world_size is None:
        world_size = int(os.environ["WORLD_SIZE"])
    if backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"rank {rank} local_rank {local_rank} has no CUDA device (visible: {torch.cuda.device_count()})"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def shard_members(batch_members: tuple[int, ...], rank: int, world_size: int) -> tuple[int, ...]:
    """The rank-local member shard of one global batch; order preserved."""

    if len(batch_members) < world_size:
        raise ValueError(f"global batch of {len(batch_members)} samples cannot serve {world_size} ranks")
    return tuple(sorted(batch_members[rank::world_size]))


def build_global_batch(
    samples: Sequence[EncodedConversation],
    batch_members: tuple[int, ...],
    rank: int,
    world_size: int,
    pad_token_id: int,
) -> tuple[PaddedBatch, dict[str, float]]:
    """Pad the rank-local shard and account for supervised tokens."""

    local_members = shard_members(batch_members, rank, world_size)
    batch = pad_samples([samples[index] for index in local_members], pad_token_id)
    supervised_global = sum(samples[index].supervised_tokens for index in batch_members)
    accounting = {
        "local_members": local_members,
        "global_members": tuple(batch_members),
        "supervised_local": float(batch.supervised_tokens),
        "supervised_global": float(supervised_global),
        "loss_scale": batch.supervised_tokens / supervised_global,
        "physical_tokens_local": float(batch.physical_tokens),
        "non_padding_tokens_local": float(batch.non_padding_tokens),
    }
    return batch, accounting


def to_device_batch(batch: PaddedBatch, device: torch.device) -> PaddedBatch:
    return PaddedBatch(
        input_ids=batch.input_ids.to(device, non_blocking=True),
        attention_mask=batch.attention_mask.to(device, non_blocking=True),
        labels=batch.labels.to(device, non_blocking=True),
        source_lines=batch.source_lines,
        lengths=batch.lengths,
        supervised_tokens_per_sample=batch.supervised_tokens_per_sample,
    )


def gradient_bytes(parameters: Sequence[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in parameters if parameter.grad is not None)


def all_reduce_gradients(
    model: torch.nn.Module,
    world_size: int,
    op: int = dist.ReduceOp.SUM,
) -> tuple[float, int]:
    """SUM every parameter gradient in place; returns (milliseconds, bytes)."""

    if world_size == 1:
        return 0.0, 0
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    total_bytes = gradient_bytes(parameters)
    if not parameters:
        return 0.0, 0
    if parameters[0].is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for parameter in parameters:
            dist.all_reduce(parameter.grad, op=op)
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)), total_bytes
    start = time.perf_counter()
    for parameter in parameters:
        dist.all_reduce(parameter.grad, op=op)
    return (time.perf_counter() - start) * 1000, total_bytes


def all_reduce_scalars(value: float, world_size: int, device: torch.device, op: int = dist.ReduceOp.SUM) -> float:
    """Sum one scalar across ranks; used for the global loss value."""

    if world_size == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def _telemetry(rank: int) -> dict[str, float]:
    local_id = int(os.environ.get("LOCAL_RANK", rank))
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits", f"--id={local_id}"],
        check=True, capture_output=True, text=True,
    )
    power, temperature, clock = result.stdout.strip().split(",")
    return {"power_watts": float(power), "temperature_celsius": float(temperature), "sm_clock_mhz": float(clock)}


def ddp_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    host_batch: PaddedBatch,
    loss_scale: float,
    device: torch.device,
    max_grad_norm: float,
    world_size: int,
    rank: int,
    collect_telemetry: bool = True,
    after_allreduce: Callable[[], None] | None = None,
) -> dict[str, float]:
    """One DDP optimizer step with every segment timed.

    Segments: h2d, forward, backward, grad_allreduce (fully exposed
    communication), scalar_sync (global-loss AllReduce, kept out of the
    gradient AllReduce), optimizer (clip + AdamW). ``step_time_ms`` is wall
    clock with a device synchronize at both ends.

    ``after_allreduce`` runs once between the gradient AllReduce and the
    clipping, on the AllReduced gradient. The correctness script uses it to
    inspect the objective gradient at the same point where the big-batch
    reference stashes its pre-clip gradient; benchmarks leave it ``None``.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type != "cuda":
        start = time.perf_counter()
        batch = to_device_batch(host_batch, device)
        h2d_ms = (time.perf_counter() - start) * 1000
        forward_start = time.perf_counter()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean") * loss_scale
        forward_ms = (time.perf_counter() - forward_start) * 1000
        backward_start = time.perf_counter()
        loss.backward()
        backward_ms = (time.perf_counter() - backward_start) * 1000
        allreduce_ms, _ = all_reduce_gradients(model, world_size)
        if after_allreduce is not None:
            after_allreduce()
        loss_value = float(loss.item())
        scalar_start = time.perf_counter()
        global_loss = all_reduce_scalars(loss_value, world_size, device)
        scalar_ms = (time.perf_counter() - scalar_start) * 1000
        optimizer_start = time.perf_counter()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer_ms = (time.perf_counter() - optimizer_start) * 1000
        step_ms = (time.perf_counter() - start) * 1000
        metrics = {
            "h2d_time_ms": h2d_ms, "forward_time_ms": forward_ms, "backward_time_ms": backward_ms,
            "grad_allreduce_ms": allreduce_ms, "scalar_sync_ms": scalar_ms, "optimizer_time_ms": optimizer_ms,
            "step_time_ms": step_ms,
        }
    else:
        torch.cuda.synchronize(device)
        wall_start = time.perf_counter()
        events = {name: torch.cuda.Event(enable_timing=True) for name in ("h2d_start", "h2d_end", "forward_end", "backward_end", "allreduce_end", "scalar_end", "optimizer_end")}
        events["h2d_start"].record()
        batch = to_device_batch(host_batch, device)
        events["h2d_end"].record()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean") * loss_scale
        events["forward_end"].record()
        loss.backward()
        events["backward_end"].record()
        allreduce_ms, _ = all_reduce_gradients(model, world_size)
        events["allreduce_end"].record()
        if after_allreduce is not None:
            after_allreduce()
        local_loss = loss.detach().clone()
        global_loss = all_reduce_scalars(float(local_loss.item()), world_size, device)
        events["scalar_end"].record()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        events["optimizer_end"].record()
        torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - wall_start) * 1000
        metrics = {
            "h2d_time_ms": float(events["h2d_end"].elapsed_time(events["forward_end"])),
            "forward_time_ms": float(events["forward_end"].elapsed_time(events["backward_end"])),
            "backward_time_ms": float(events["backward_end"].elapsed_time(events["allreduce_end"])) - allreduce_ms,
            "grad_allreduce_ms": allreduce_ms,
            "scalar_sync_ms": float(events["allreduce_end"].elapsed_time(events["scalar_end"])),
            "optimizer_time_ms": float(events["scalar_end"].elapsed_time(events["optimizer_end"])),
            "step_time_ms": step_ms,
        }
        loss_value = float(local_loss.item())
    metrics.update({
        "local_loss": loss_value,
        "global_loss": global_loss,
        "gradient_norm_before_clip": float(grad_norm.item()),
        "physical_tokens": float(batch.physical_tokens),
        "non_padding_tokens": float(batch.non_padding_tokens),
        "supervised_tokens": float(batch.supervised_tokens),
        "padding_tokens": float(batch.padding_tokens),
        "padding_ratio": batch.padding_ratio,
        "batch_max_length": float(max(batch.lengths)),
        "samples": float(batch.batch_size),
    })
    seconds = metrics["step_time_ms"] / 1000
    metrics.update({
        "physical_tokens_per_second": batch.physical_tokens / seconds,
        "non_padding_tokens_per_second": batch.non_padding_tokens / seconds,
        "supervised_tokens_per_second": batch.supervised_tokens / seconds,
    })
    if device.type == "cuda" and collect_telemetry:
        metrics.update(_telemetry(rank))
    return metrics


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def spawn_workers(worker: Callable[..., None], world_size: int, extra_args: tuple) -> None:
    """Set env:// variables and launch ``world_size`` worker processes.

    Worker exceptions propagate: ``mp.spawn`` re-raises the first failure and
    the launcher exits non-zero.
    """

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(free_port())
    mp.spawn(worker, args=(world_size, *extra_args), nprocs=world_size, join=True)


def encode_samples(
    tokenizer: Qwen3Tokenizer,
    records: list[dict],
    max_length: int,
) -> list[EncodedConversation]:
    """Encode the fixed corpus once; batch membership never re-encodes."""

    from mini_batch import trim_sample
    from assistant_loss import encode_conversation

    return [
        trim_sample(encode_conversation(tokenizer, record, max_length, "assistant_only"))
        for record in records
    ]


def make_batch_plan(num_samples: int, batch_size: int, seed: int) -> list[tuple[int, ...]]:
    """The exact Chapter 22 shuffle: members sorted inside each batch."""

    rng = random.Random(f"{seed}|1|shuffle")
    order = list(range(num_samples))
    rng.shuffle(order)
    return [tuple(sorted(order[i : i + batch_size])) for i in range(0, len(order), batch_size)]
