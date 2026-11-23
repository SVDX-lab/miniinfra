"""Chapter 31 core: delaying gradient synchronization across accumulation steps.

One optimizer step consists of ``num_micro`` micro-steps. On every micro-step
each rank computes a local objective
``token_mean(micro-batch) * supervised_micro / supervised_global`` — the same
supervised-token weighting as Chapters 09 and 14 — so the SUM over ranks and
micro-steps of the partial gradients equals the gradient of the global
``token_mean`` objective over the whole global batch.

Two synchronization modes, identical in math and in total work per
optimizer step, differ ONLY in when AllReduce runs:

- ``every`` (baseline): after each micro-step backward, AllReduce SUM the
  fresh local gradient into an accumulator. ``num_micro`` AllReduces per
  optimizer step, each fully exposed, each moving the full FP32 gradient.
- ``last`` (no_sync): backward every micro-step with pure local accumulation,
  then ONE AllReduce SUM of the total. One AllReduce per optimizer step.

The baseline cannot simply AllReduce the accumulating ``.grad`` each
micro-step: after the first reduce the buffer already holds cross-rank sums,
and reducing again would multiply earlier micro-step contributions by the
world size. The accumulator implementation is the honest "every micro-step
sees the correct global gradient" baseline.
"""

from __future__ import annotations

import subprocess
import time

import torch
import torch.distributed as dist

from mini_batch import pad_samples
from ddp_core import shard_members, to_device_batch


SYNC_MODES = ("every", "last")


def build_micro_batches(
    samples,
    batch_members: tuple[int, ...],
    rank: int,
    world_size: int,
    num_micro: int,
    pad_token_id: int,
) -> tuple[list, dict]:
    """Split this rank's shard into ``num_micro`` contiguous micro-batches."""

    rank_members = shard_members(batch_members, rank, world_size)
    if len(rank_members) % num_micro != 0:
        raise ValueError(f"{len(rank_members)} shard members cannot split into {num_micro} equal micro-batches")
    chunk = len(rank_members) // num_micro
    supervised_global = sum(samples[index].supervised_tokens for index in batch_members)
    micros = []
    for index in range(num_micro):
        group = rank_members[index * chunk : (index + 1) * chunk]
        batch = pad_samples([samples[position] for position in group], pad_token_id)
        micros.append((batch, batch.supervised_tokens))
    accounting = {
        "supervised_global": float(supervised_global),
        "supervised_local_total": float(sum(supervised for _, supervised in micros)),
        "num_micro": num_micro,
    }
    if accounting["supervised_local_total"] == 0:
        raise ValueError("rank received no supervised tokens")
    return micros, accounting


def _telemetry(rank: int) -> dict[str, float]:
    import os

    local_id = int(os.environ.get("LOCAL_RANK", rank))
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits", f"--id={local_id}"],
        check=True, capture_output=True, text=True,
    )
    power, temperature, clock = result.stdout.strip().split(",")
    return {"power_watts": float(power), "temperature_celsius": float(temperature), "sm_clock_mhz": float(clock)}


def all_reduce_gradients_timed(model: torch.nn.Module, world_size: int) -> tuple[float, int]:
    """SUM all gradients in place, event-timed; returns (ms, bytes)."""

    if world_size == 1:
        return 0.0, 0
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    if parameters[0].is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for parameter in parameters:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)), total_bytes
    start = time.perf_counter()
    for parameter in parameters:
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    return (time.perf_counter() - start) * 1000, total_bytes


def accumulate_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    micros: list,
    supervised_global: float,
    device: torch.device,
    max_grad_norm: float,
    world_size: int,
    rank: int,
    sync_mode: str,
    collect_telemetry: bool = True,
) -> dict[str, float]:
    """One optimizer step built from micro-steps under the given sync mode."""

    if sync_mode not in SYNC_MODES:
        raise ValueError(f"unknown sync mode: {sync_mode}")
    from mini_batch import batch_loss

    model.train()
    optimizer.zero_grad(set_to_none=True)
    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    if on_cuda:
        events = {name: torch.cuda.Event(enable_timing=True) for name in ("h2d_end", "h2d_stop", "backward_end", "allreduce_end", "optimizer_end")}
        events["h2d_end"].record()
    else:
        events = None
    device_micros = [(to_device_batch(batch, device), supervised) for batch, supervised in micros]
    if on_cuda:
        events["h2d_stop"].record()

    forward_pairs: list[tuple] = []
    backward_pairs: list[tuple] = []
    forward_ms = 0.0
    backward_ms = 0.0
    local_losses: list[float] = []
    if sync_mode == "every":
        accumulator: list[torch.Tensor] | None = None
        allreduce_ms = 0.0
        allreduce_count = 0
        comm_bytes = 0
        for batch, supervised in device_micros:
            optimizer.zero_grad(set_to_none=True)
            if on_cuda:
                seg_start = torch.cuda.Event(enable_timing=True)
                seg_end = torch.cuda.Event(enable_timing=True)
                seg_start.record()
            else:
                seg_start = time.perf_counter()
            logits = model(batch.input_ids, attention_mask=batch.attention_mask)
            loss = batch_loss(logits, batch.labels, "token_mean") * (supervised / supervised_global)
            if on_cuda:
                seg_end.record()
                forward_pairs.append((seg_start, seg_end))
                seg_start.record()
            local_losses.append(float(loss.detach().item()))
            loss.backward()
            if on_cuda:
                seg_end.record()
                backward_pairs.append((seg_start, seg_end))
            reduced_ms, reduced_bytes = all_reduce_gradients_timed(model, world_size)
            allreduce_ms += reduced_ms
            comm_bytes += reduced_bytes
            allreduce_count += 1
            if accumulator is None:
                accumulator = [parameter.grad.detach().clone() for parameter in model.parameters()]
            else:
                for target, parameter in zip(accumulator, model.parameters()):
                    target.add_(parameter.grad)
        if on_cuda:
            events["allreduce_end"].record()
        for parameter, value in zip(model.parameters(), accumulator):
            parameter.grad = value
        gradient_scope = "sum of per-micro-step AllReduced gradients"
    else:
        for batch, supervised in device_micros:
            if on_cuda:
                seg_start = torch.cuda.Event(enable_timing=True)
                seg_end = torch.cuda.Event(enable_timing=True)
                seg_start.record()
            else:
                seg_start = time.perf_counter()
            logits = model(batch.input_ids, attention_mask=batch.attention_mask)
            loss = batch_loss(logits, batch.labels, "token_mean") * (supervised / supervised_global)
            if on_cuda:
                seg_end.record()
                forward_pairs.append((seg_start, seg_end))
                seg_start.record()
            local_losses.append(float(loss.detach().item()))
            loss.backward()
            if on_cuda:
                seg_end.record()
                backward_pairs.append((seg_start, seg_end))
        if on_cuda:
            events["backward_end"].record()
        allreduce_ms, comm_bytes = all_reduce_gradients_timed(model, world_size)
        allreduce_count = 1 if world_size > 1 else 0
        if on_cuda:
            events["allreduce_end"].record()
        gradient_scope = "single AllReduce over locally accumulated gradients"

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    if on_cuda:
        events["optimizer_end"].record()
        torch.cuda.synchronize(device)
        # elapsed_time requires completed events; settle after the final sync
        forward_ms = sum(float(s.elapsed_time(e)) for s, e in forward_pairs)
        backward_ms = sum(float(s.elapsed_time(e)) for s, e in backward_pairs)
    step_ms = (time.perf_counter() - wall_start) * 1000

    metrics = {
        "h2d_time_ms": 0.0 if not on_cuda else float(events["h2d_end"].elapsed_time(events["h2d_stop"])),
        "forward_time_ms": forward_ms,
        "backward_time_ms": backward_ms,
        "grad_allreduce_ms": allreduce_ms,
        "allreduce_count": float(allreduce_count),
        "allreduce_bytes_total": float(comm_bytes),
        "optimizer_time_ms": 0.0 if not on_cuda else float(events["allreduce_end"].elapsed_time(events["optimizer_end"])),
        "step_time_ms": step_ms,
        "sync_mode": sync_mode,
        "gradient_scope": gradient_scope,
        "micro_steps": float(len(device_micros)),
    }
    supervised_measured = sum(batch.supervised_tokens for batch, _ in device_micros)
    metrics.update({
        "local_loss": sum(local_losses),
        "gradient_norm_before_clip": float(grad_norm.item()),
        "physical_tokens": float(sum(batch.physical_tokens for batch, _ in device_micros)),
        "non_padding_tokens": float(sum(batch.non_padding_tokens for batch, _ in device_micros)),
        "supervised_tokens": float(supervised_measured),
        "batch_max_length": float(max(max(batch.lengths) for batch, _ in device_micros)),
        "samples": float(sum(batch.batch_size for batch, _ in device_micros)),
    })
    seconds = step_ms / 1000
    metrics.update({
        "supervised_tokens_per_second": supervised_measured / seconds,
    })
    if device.type == "cuda" and collect_telemetry:
        metrics.update(_telemetry(rank))
    return metrics
