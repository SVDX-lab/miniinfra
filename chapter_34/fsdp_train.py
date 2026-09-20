"""Chapter 34 variant engine: the official FSDP over the full Qwen3-0.6B.

The handwritten ``zero3_core.py`` explains the parameter lifecycle on tiny
units; this module puts the REAL model through the same lifecycle with
``torch.distributed.fsdp`` (FSDP1), the official implementation the outline
names as the full-model counterpart:

- ``ShardingStrategy.FULL_SHARD``: parameters, gradients, and optimizer state
  are all sharded (ZeRO-3), not just state (ZeRO-1) or state+gradients
  (ZeRO-2, the chapter baseline).
- ``transformer_auto_wrap_policy`` over ``Qwen3DecoderLayer``: each decoder
  layer becomes one resharding unit; embedding, final norm, and lm_head live
  in the root unit. Units AllGather their flat parameter before their forward,
  free it afterwards, and re-gather in backward (BACKWARD_PRE prefetch).
- ``use_orig_params=True``: REQUIRED for ``tie_word_embeddings=True`` —
  Qwen3-0.6B shares one Parameter between ``embed_tokens`` and ``lm_head``,
  and FSDP refuses shared parameters with the legacy flat-parameter view.
  With original parameters the tie stays one object (and one shard), so
  parameter checksums must not count it twice.
- ``CPUOffload=None`` and no mixed precision: FP32 everywhere, matching
  Chapters 14-18.

One semantic difference MUST be compensated: FSDP Reduce-Scatters gradients
with the world-size AVERAGE, while the Chapter 29+ objective is the global
``token_mean`` whose partial gradients are SUMMED across ranks (each rank
scales its local loss by ``supervised_local / supervised_global``). This
module therefore multiplies the local loss by ``world_size`` before backward:
``AVG(W * (S_r / S_global) * grad l_r)`` sums back to the global objective
gradient. ``compare_correctness.py`` verifies this numerically at step 0.

FSDP embeds all AllGather/Reduce-Scatter work inside the forward/backward
phases — unlike Chapter 33's exposed post-backward chain — so the step
metrics report the communication as embedded and do not pretend a separate
segment exists.
"""

from __future__ import annotations

import time

import torch

from mini_batch import PaddedBatch, batch_loss


def wrap_model_fsdp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Wrap the handwritten Qwen3 model with FSDP FULL_SHARD.

    Same-process reuse: the wrap mutates the module in place and returns it as
    an ``FullyShardedDataParallel`` instance; callers drop both references.
    ``sync_module_states`` needs a CUDA device id (on the CPU smoke path all
    ranks already build identical weights from the same seed).
    """

    from functools import partial

    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from qwen3_model import Qwen3DecoderLayer

    options = {
        "auto_wrap_policy": partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Qwen3DecoderLayer}
        ),
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "cpu_offload": None,
        "mixed_precision": None,
        "forward_prefetch": False,
        "limit_all_gathers": True,
        "use_orig_params": True,
    }
    if device.type == "cuda":
        options["device_id"] = device
        options["sync_module_states"] = True
    return FullyShardedDataParallel(model, **options)


def fsdp_train_step(
    fsdp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    host_batch: PaddedBatch,
    loss_scale: float,
    device: torch.device,
    max_grad_norm: float,
    world_size: int,
    rank: int,
    collect_telemetry: bool = False,
) -> dict[str, float]:
    """One FSDP optimizer step with per-segment timing, Chapter 33-style keys.

    ``loss_scale`` keeps the Chapter 29 convention (supervised_local /
    supervised_global); the world-size factor for FSDP's averaged
    Reduce-Scatter is applied inside. ``grad_allreduce_ms`` /
    ``param_allgather_ms`` stay 0.0 by definition: the communication lives
    inside the forward/backward segments under FULL_SHARD.
    """

    import subprocess

    from ddp_core import to_device_batch

    fsdp_model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type != "cuda":
        start = time.perf_counter()
        batch = to_device_batch(host_batch, device)
        logits = fsdp_model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean") * (loss_scale * world_size)
        forward_ms = (time.perf_counter() - start) * 1000
        backward_start = time.perf_counter()
        loss.backward()
        backward_ms = (time.perf_counter() - backward_start) * 1000
        clip_start = time.perf_counter()
        grad_norm = fsdp_model.clip_grad_norm_(max_grad_norm)
        optimizer.step()
        optimizer_ms = (time.perf_counter() - clip_start) * 1000
        step_ms = (time.perf_counter() - start) * 1000
        metrics = {
            "h2d_time_ms": 0.0, "forward_time_ms": forward_ms, "backward_time_ms": backward_ms,
            "grad_allreduce_ms": 0.0, "param_allgather_ms": 0.0, "optimizer_time_ms": optimizer_ms,
            "step_time_ms": step_ms,
        }
        loss_value = float(loss.detach().item()) / world_size
    else:
        torch.cuda.synchronize(device)
        wall_start = time.perf_counter()
        events = {name: torch.cuda.Event(enable_timing=True) for name in ("h2d_start", "h2d_end", "forward_end", "backward_end", "optimizer_end")}
        events["h2d_start"].record()
        batch = to_device_batch(host_batch, device)
        events["h2d_end"].record()
        logits = fsdp_model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean") * (loss_scale * world_size)
        events["forward_end"].record()
        loss.backward()
        events["backward_end"].record()
        # FSDP's own clip computes the TRUE global norm from the shards (same
        # math as zero2_core.clip_grad_norm_sharded) and scales the shards.
        grad_norm = fsdp_model.clip_grad_norm_(max_grad_norm)
        optimizer.step()
        events["optimizer_end"].record()
        torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - wall_start) * 1000
        metrics = {
            "h2d_time_ms": float(events["h2d_start"].elapsed_time(events["h2d_end"])),
            "forward_time_ms": float(events["h2d_end"].elapsed_time(events["forward_end"])),
            "backward_time_ms": float(events["forward_end"].elapsed_time(events["backward_end"])),
            "grad_allreduce_ms": 0.0,
            "param_allgather_ms": 0.0,
            "optimizer_time_ms": float(events["backward_end"].elapsed_time(events["optimizer_end"])),
            "step_time_ms": step_ms,
        }
        loss_value = float(loss.detach().item()) / world_size
    metrics.update({
        # Report the UNSCALED local token_mean so FSDP and zero2 losses are
        # directly comparable; the backward saw the loss * world_size.
        "local_loss": loss_value,
        "global_loss": None,
        "communication_embedded": True,
        "gradient_norm_before_clip": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
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
        metrics.update(telemetry(rank))
    return metrics


def telemetry(rank: int) -> dict[str, float]:
    """Local-GPU power/temperature/clock following LOCAL_RANK (multi-node)."""

    import os
    import subprocess

    local_id = int(os.environ.get("LOCAL_RANK", rank))
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits", f"--id={local_id}"],
        check=True, capture_output=True, text=True,
    )
    power, temperature, clock = result.stdout.strip().split(",")
    return {"power_watts": float(power), "temperature_celsius": float(temperature), "sm_clock_mhz": float(clock)}


def set_full_state_dict_type(fsdp_model: torch.nn.Module) -> None:
    """Configure FULL_STATE_DICT for model AND optimizer, offloaded to CPU.

    ``rank0_only=False`` so every rank receives the identical full state and
    can rebuild a fresh sharded model without an extra broadcast.
    """

    from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType
    from torch.distributed.fsdp.api import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
    )

    # torch 2.8 turned ``set_state_dict_type`` into a staticmethod whose first
    # argument is the module; calling it through the instance shifts every
    # argument one slot left (the config lands where the enum is expected).
    FullyShardedDataParallel.set_state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )


def get_full_state_dicts(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[dict, dict]:
    """(model state dict, optimizer state dict) as plain full tensors on CPU."""

    set_full_state_dict_type(fsdp_model)
    model_state = fsdp_model.state_dict()
    optimizer_state = fsdp_optim_state(fsdp_model, optimizer)
    return model_state, optimizer_state


def fsdp_optim_state(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    from torch.distributed.fsdp import FullyShardedDataParallel

    return FullyShardedDataParallel.optim_state_dict(fsdp_model, optimizer)


def load_full_state_dicts(
    fsdp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_state: dict,
    optimizer_state: dict,
) -> None:
    """Restore full state into an FSDP model + fresh torch optimizer."""

    from torch.distributed.fsdp import FullyShardedDataParallel

    set_full_state_dict_type(fsdp_model)
    fsdp_model.load_state_dict(model_state)
    converted = FullyShardedDataParallel.optim_state_dict_to_load(fsdp_model, optimizer, optimizer_state)
    optimizer.load_state_dict(converted)


def fsdp_theoretical_ledger(numel: int, world_size: int, element_size: int = 4) -> dict[str, float]:
    """The outline's ZeRO-3 ledger: everything lives at 1/world_size.

    16 bytes per parameter for FP32 params+grads+AdamW moments -> 16N/W per
    rank, against 16N on a ZeRO-2 rank (full params + full grads peak).
    """

    parameter_bytes = numel * element_size
    state_bytes = 2 * parameter_bytes
    return {
        "parameters_full_bytes": float(parameter_bytes),
        "parameters_shard_bytes": float(parameter_bytes / world_size),
        "gradients_full_bytes": float(parameter_bytes),
        "gradients_shard_bytes": float(parameter_bytes / world_size),
        "optimizer_state_full_bytes": float(state_bytes),
        "optimizer_state_shard_bytes": float(state_bytes / world_size),
        "resident_total_bytes": float(4 * parameter_bytes / world_size),
        "full_replica_total_bytes": float(4 * parameter_bytes),
        "resident_fraction_of_full": 1.0 / world_size,
    }
