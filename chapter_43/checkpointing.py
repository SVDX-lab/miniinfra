"""Chapter 43 core: the FULL training state, saved centralized or sharded.

The only variable of this chapter: whether the state is written as ONE
central file (rank 0 aggregates, the other ranks wait at a barrier) or as
PER-RANK shards written in parallel plus a manifest.  Both modes carry the
SAME inventory and must restore an identical training state:

- model parameters (full FP32 state dict);
- optimizer state (AdamW moments) and LR scheduler;
- global step, consumed supervised tokens, consumed samples;
- sampler state: the deterministic batch plan seed and the batch cursor;
- RNG: Python ``random``, NumPy, torch CPU, torch CUDA — PER RANK;
- DP/TP/PP topology metadata and per-rank shard descriptions;
- software versions (python/torch/cuda/nccl) and the training config.

REPLICA SEMANTICS (the load-bearing fact of the DDP layout): after every
gradient AllReduce, all ranks hold BITWISE identical parameters, gradients
and optimizer moments, so every rank's shard is a complete, INTERCHANGEABLE
copy of the model state.  This is what "sharded saving" means for plain DDP:
the I/O is parallelized and each shard is self-sufficient, but no bytes are
saved versus the centralized file (the interesting compression cases are
ZeRO/FSDP shards, chapter 19).  Consequences the code makes explicit:

- validation: all ranks' model checksums MUST agree — a mismatch means the
  DDP all-reduce contract broke and the checkpoint would silently fork;
- resharding (save on W_old ranks, load on W_new ranks): new rank r reads
  shard ``r % W_old`` (replica selection) after rank 0 verified that the
  manifest lists identical checksums for every shard.  RNG states are
  per-rank bookkeeping; the new rank r adopts the state of old rank
  ``r % W_old`` — deterministic, and irrelevant to the loss trajectory
  because the chapter's training loop consumes no randomness (deterministic
  plan, no dropout); any stochastic module would need per-rank streams
  re-derived from a seeded generator (documented, not hidden).

``torch.distributed.checkpoint`` (DCP) is available as the OFFICIAL
comparison path (``dcp_save_checkpoint``/``dcp_load_checkpoint``): it
deduplicates replicated tensors on save and reloads them under a different
world size — the resharding semantics this chapter's hand-written modes
degenerate to for DDP.  Missing DCP raises instead of silently skipping.

Atomic writes everywhere: ``*.tmp`` + ``os.replace`` so a killed rank can
never leave a half-written checkpoint that a resume would trust.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from experiment_utils import git_revision


SCHEMA_VERSION = "chapter43.checkpoint.v1"


# ---------------------------------------------------------------------------
# RNG state machine
# ---------------------------------------------------------------------------


def capture_rng() -> dict[str, Any]:
    """The per-rank RNG state: Python, NumPy, torch CPU, torch CUDA."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    """Inverse of ``capture_rng``; unknown keys raise instead of being ignored."""

    random.setstate(_as_python_rng(state["python"]))
    np.random.set_state(_as_numpy_rng(state["numpy"]))
    torch.set_rng_state(_as_byte_tensor(state["torch_cpu"]))
    if torch.cuda.is_available():
        if "torch_cuda" not in state:
            raise ValueError("checkpoint has no CUDA RNG state but CUDA is in use")
        torch.cuda.set_rng_state_all([_as_byte_tensor(item) for item in state["torch_cuda"]])


def _as_python_rng(value: Any):
    """Normalize a saved python RNG state to the ``random.getstate()`` form.

    The live state is the 3-tuple ``(version, state_vector_tuple, gauss_next)``
    and pickle (torch.save / all_gather_object) keeps it intact.  A JSON-style
    round trip instead yields lists — and ``random.setstate`` rejects a list
    state vector ("state vector must be a tuple"), so the vector is rebuilt as
    a tuple of ints before restoring.
    """

    if isinstance(value, (tuple, list)) and len(value) == 3:
        version, keys, gauss = value
        return (int(version), tuple(int(k) for k in keys), gauss)
    raise ValueError(f"unrecognized python RNG state: {type(value)!r}")


def _as_numpy_rng(value: Any):
    """Normalize a saved NumPy RNG state to the ``np.random.set_state`` form.

    The legacy global stream (``np.random.get_state()``) returns the 5-tuple
    ``('MT19937', uint32 keys[624], pos, has_gauss, cached_gauss)``; pickle
    (torch.save / all_gather_object) preserves exactly that tuple, while a
    JSON-style round trip yields a plain list.  A ``np.random.Generator``
    instead exposes its state as a DICT — an MT19937 dict is converted to the
    legacy tuple (same MT19937 core, so the uniform/integer stream continues
    identically; the legacy-only gaussian cache is restored empty), any other
    dict raises instead of being guessed at.
    """

    if isinstance(value, dict):
        inner = value.get("state")
        if value.get("bit_generator") != "MT19937" or not isinstance(inner, dict) \
                or "key" not in inner or "pos" not in inner:
            raise ValueError(
                f"unsupported numpy Generator state (keys {sorted(value)}): only an "
                "MT19937 dict can be restored onto the legacy global stream"
            )
        return ("MT19937", np.ascontiguousarray(inner["key"], dtype=np.uint32),
                int(inner["pos"]), 0, 0.0)
    if isinstance(value, (tuple, list)) and len(value) == 5:
        kind, keys, pos, has_gauss, cached = value
        return (str(kind), np.ascontiguousarray(keys, dtype=np.uint32),
                int(pos), int(has_gauss), float(cached))
    raise ValueError(f"unrecognized numpy RNG state: {type(value)!r}")


def _as_byte_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.uint8)
    return torch.tensor(list(value), dtype=torch.uint8)


def rng_fingerprint() -> str:
    """Stable digest of the torch CPU RNG state, for trajectory records."""

    state = torch.get_rng_state().numpy().tobytes()
    return hashlib.sha256(state).hexdigest()[:16]


# ---------------------------------------------------------------------------
# state collection and checksums
# ---------------------------------------------------------------------------


def sync_scheduler_lr(optimizer: torch.optim.Optimizer, scheduler) -> None:
    """Re-apply the scheduler's last LR to the optimizer after a resume.

    ``LRScheduler.load_state_dict`` updates the scheduler's bookkeeping
    (last_epoch, _last_lr) but NOT the optimizer's param_groups: a freshly
    constructed scheduler has already set the optimizer to its INITIAL lr
    (the warmup start), so the first resumed step would silently train with
    the wrong LR.  ``_last_lr`` is part of the state dict — copy it back.
    (Chapter 16's checkpoint test happened to pass through this trap only
    because its flat-warmup LR made the stale and correct values equal.)
    """

    last_lrs = scheduler.get_last_lr()
    if len(last_lrs) != len(optimizer.param_groups):
        raise ValueError(
            f"scheduler covers {len(last_lrs)} lr values for {len(optimizer.param_groups)} param groups"
        )
    for group, lr in zip(optimizer.param_groups, last_lrs):
        group["lr"] = float(lr)


def software_record() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "nccl": torch.cuda.nccl.version() if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "git_revision": git_revision(),
    }


def _chunked_sum_of_squares(tensor: torch.Tensor, chunk_elements: int = 100_000_000) -> torch.Tensor:
    """float64 sum of squares, moved to the host in <= chunk_elements slices.

    The previous one-shot GPU .double().pow(2) materialized up to ~2x the
    tensor bytes on the GPU and deterministically OOMed 12 GB cards right
    after the training step (embedding: +1.16 GiB on top of 10.06 GiB
    already allocated).  Chunked host-side accumulation is a measurement-
    helper memory fix only: it does not change what is saved or restored,
    and it stays deterministic (fixed chunk split, fixed reduction order),
    so the exact checksum equality checks across ranks and across
    save/resume still hold.
    """

    flat = tensor.detach().reshape(-1)
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start:start + chunk_elements].to("cpu", non_blocking=False)
        total += chunk.double().pow(2).sum()
    return total


def model_checksum(model: torch.nn.Module) -> float:
    """Sum-of-squares checksum of the parameters (float64)."""

    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        total += _chunked_sum_of_squares(parameter)
    return float(total.item())


def optimizer_checksum(optimizer: torch.optim.Optimizer) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += _chunked_sum_of_squares(value)
    return float(total.item())


def collect_local_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        scheduler, progress: dict, sampler: dict, config: dict,
                        topology: dict) -> dict:
    """Everything THIS rank must contribute to a checkpoint."""

    return {
        "model": {name: parameter.detach().to("cpu", copy=True)
                  for name, parameter in model.state_dict().items()},
        "optimizer": _to_cpu_optimizer_state(optimizer),
        "scheduler": scheduler.state_dict(),
        "progress": dict(progress),
        "sampler": dict(sampler),
        "rng": capture_rng(),
        "model_checksum": model_checksum(model),
        "optimizer_checksum": optimizer_checksum(optimizer),
    }


def _to_cpu_optimizer_state(optimizer: torch.optim.Optimizer) -> dict:
    state = optimizer.state_dict()
    cloned: dict[str, Any] = {"state": {}, "param_groups": state["param_groups"]}
    for index, per_parameter in state["state"].items():
        cloned["state"][index] = {
            key: value.detach().to("cpu", copy=True) if torch.is_tensor(value) else value
            for key, value in per_parameter.items()
        }
    return cloned


def _validate_progress_consistency(states: list[dict]) -> None:
    reference = states[0]["progress"]
    for rank, state in enumerate(states[1:], start=1):
        if state["progress"] != reference:
            raise AssertionError(
                f"rank {rank} progress {state['progress']} diverges from rank 0 {reference}; "
                "the training loop is supposed to be deterministic"
            )


# ---------------------------------------------------------------------------
# mode 1: centralized (rank 0 writes ONE file)
# ---------------------------------------------------------------------------


def save_centralized(directory: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     scheduler, progress: dict, sampler: dict, config: dict, topology: dict,
                     rank: int, world_size: int) -> dict:
    """Rank 0 aggregates every rank's RNG + checksums and writes one file.

    The model/optimizer tensors are taken from rank 0 only — the DDP replica
    contract makes them identical, and the gathered checksums PROVE it before
    the file is written.  Returns timing/size metrics.
    """

    local = collect_local_state(model, optimizer, scheduler, progress, sampler, config, topology)
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, {
        "rng": local["rng"],
        "model_checksum": local["model_checksum"],
        "optimizer_checksum": local["optimizer_checksum"],
        "progress": local["progress"],
    })
    _validate_progress_consistency([{"progress": item["progress"]} for item in gathered])
    checksums = [item["model_checksum"] for item in gathered]
    if max(checksums) - min(checksums) > 0.0:
        raise AssertionError(
            f"DDP replica divergence: model checksums differ across ranks {checksums[0]=}, "
            f"{min(checksums)=}, {max(checksums)=}"
        )

    directory.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"mode": "centralized", "world_size": world_size}
    dist.barrier()
    begin = time.perf_counter()
    if rank == 0:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "mode": "centralized",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "software": software_record(),
            "config": dict(config),
            "topology": dict(topology),
            "progress": local["progress"],
            "sampler": local["sampler"],
            "model": local["model"],
            "optimizer": local["optimizer"],
            "scheduler": local["scheduler"],
            "model_checksum": local["model_checksum"],
            "optimizer_checksum": local["optimizer_checksum"],
            "model_checksum_per_rank": checksums,
            "rng_per_rank": [item["rng"] for item in gathered],
        }
        target = directory / "checkpoint.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        metrics["bytes"] = float(target.stat().st_size)
    dist.barrier()
    metrics["seconds"] = time.perf_counter() - begin
    if rank == 0:
        metrics["file"] = str(directory / "checkpoint.pt")
    return metrics


def load_centralized(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     scheduler, device: torch.device, rank: int) -> dict:
    """Every rank reads the single file; RNG adopted from own rank slot."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema {payload.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    scheduler.load_state_dict(payload["scheduler"])
    sync_scheduler_lr(optimizer, scheduler)
    rng_states = payload["rng_per_rank"]
    restore_rng(rng_states[rank % len(rng_states)])
    checksum = model_checksum(model)
    if abs(checksum - payload["model_checksum"]) > 0.0:
        raise AssertionError(
            f"restored model checksum {checksum} != recorded {payload['model_checksum']}"
        )
    return {
        "progress": payload["progress"],
        "sampler": payload["sampler"],
        "config": payload["config"],
        "topology": payload["topology"],
        "saved_rng_slot": rank % len(rng_states),
    }


# ---------------------------------------------------------------------------
# mode 2: sharded (every rank writes its own shard + manifest)
# ---------------------------------------------------------------------------


def save_sharded(directory: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler, progress: dict, sampler: dict, config: dict, topology: dict,
                 rank: int, world_size: int) -> dict:
    """Parallel per-rank shards plus a rank-0 manifest.

    Each rank writes a SELF-SUFFICIENT shard (full model + optimizer + its
    RNG); rank 0's manifest records the shared progress/sampler/config and
    every shard's file name, checksum and size.  Writes overlap; the barrier
    bounds the timing window.  Returns timing/size metrics.
    """

    local = collect_local_state(model, optimizer, scheduler, progress, sampler, config, topology)
    directory.mkdir(parents=True, exist_ok=True)
    shard_name = f"shard-{rank:05d}-of-{world_size:05d}.pt"
    metrics: dict[str, Any] = {"mode": "sharded", "world_size": world_size}
    dist.barrier()
    begin = time.perf_counter()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "sharded",
        "rank": rank,
        "world_size": world_size,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "software": software_record(),
        "config": dict(config),
        "topology": dict(topology),
        "progress": local["progress"],
        "sampler": local["sampler"],
        "model": local["model"],
        "optimizer": local["optimizer"],
        "scheduler": local["scheduler"],
        "model_checksum": local["model_checksum"],
        "optimizer_checksum": local["optimizer_checksum"],
        "rng": local["rng"],
    }
    target = directory / shard_name
    temporary = target.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    metrics["bytes"] = float(target.stat().st_size)
    shard_meta = {
        "file": shard_name,
        "rank": rank,
        "model_checksum": local["model_checksum"],
        "optimizer_checksum": local["optimizer_checksum"],
        "bytes": metrics["bytes"],
        "progress": local["progress"],
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, shard_meta)
    _validate_progress_consistency([{"progress": item["progress"]} for item in gathered])
    if rank == 0:
        checksums = [item["model_checksum"] for item in gathered]
        if max(checksums) - min(checksums) > 0.0:
            raise AssertionError(f"DDP replica divergence: shard checksums differ {checksums}")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "mode": "sharded",
            "world_size": world_size,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "software": software_record(),
            "config": dict(config),
            "topology": dict(topology),
            "progress": local["progress"],
            "sampler": local["sampler"],
            "shards": sorted(gathered, key=lambda item: item["rank"]),
        }
        manifest_path = directory / "manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    dist.barrier()
    metrics["seconds"] = time.perf_counter() - begin
    metrics["file"] = str(target)
    return metrics


def read_manifest(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest.json in {directory}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def detect_mode(directory: Path) -> str:
    """'centralized' | 'sharded' from what the directory contains."""

    if (directory / "manifest.json").is_file():
        return "sharded"
    if (directory / "checkpoint.pt").is_file():
        return "centralized"
    raise FileNotFoundError(f"no checkpoint found in {directory}")


def load_sharded(directory: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler, device: torch.device, rank: int, world_size: int) -> dict:
    """Load this rank's shard; resharding = deterministic replica selection.

    With ``world_size`` equal to the saving world size each rank reads its
    own shard.  With a DIFFERENT world size, new rank ``r`` reads shard
    ``r % saved_world`` — valid precisely because DDP shards are
    interchangeable full replicas; the manifest checksums were verified
    equal at save time and are re-verified here against the loaded shard.
    """

    manifest = read_manifest(directory)
    if manifest["mode"] != "sharded":
        raise ValueError(f"expected a sharded checkpoint in {directory}")
    saved_world = int(manifest["world_size"])
    checksums = {shard["rank"]: shard["model_checksum"] for shard in manifest["shards"]}
    if len(set(checksums.values())) != 1:
        raise AssertionError(f"manifest lists divergent shard checksums: {checksums}")

    shard_file = directory / f"shard-{rank % saved_world:05d}-of-{saved_world:05d}.pt"
    payload = torch.load(shard_file, map_location="cpu", weights_only=False)
    if payload["model_checksum"] != manifest["shards"][0]["model_checksum"]:
        raise AssertionError(
            f"shard {shard_file.name} checksum {payload['model_checksum']} disagrees with the manifest"
        )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    scheduler.load_state_dict(payload["scheduler"])
    sync_scheduler_lr(optimizer, scheduler)
    restore_rng(payload["rng"])
    checksum = model_checksum(model)
    if abs(checksum - payload["model_checksum"]) > 0.0:
        raise AssertionError(f"restored model checksum {checksum} != recorded {payload['model_checksum']}")
    return {
        "progress": manifest["progress"],
        "sampler": manifest["sampler"],
        "config": manifest["config"],
        "topology": manifest["topology"],
        "loaded_shard": shard_file.name,
        "resharded": saved_world != world_size,
        "saved_world_size": saved_world,
    }


def load_for_resume(directory: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, device: torch.device, rank: int, world_size: int,
                    mode: str | None = None) -> dict:
    """Auto-detect the mode (or force one) and restore the full state."""

    mode = mode or detect_mode(directory)
    if mode == "centralized":
        return load_centralized(directory / "checkpoint.pt", model, optimizer, scheduler, device, rank)
    if mode == "sharded":
        return load_sharded(directory, model, optimizer, scheduler, device, rank, world_size)
    raise ValueError(f"unknown checkpoint mode {mode!r}")


# ---------------------------------------------------------------------------
# official comparison path: torch.distributed.checkpoint (DCP)
# ---------------------------------------------------------------------------


def dcp_available() -> bool:
    try:
        import torch.distributed.checkpoint  # noqa: F401
        import torch.distributed.checkpoint.state_dict  # noqa: F401
        return True
    except ImportError:
        return False


def _dcp_state_dict(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_sd, optim_sd = get_state_dict(model, optimizer)
    return {"model": model_sd, "optim": optim_sd}


def dcp_save_checkpoint(directory: Path, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer) -> dict:
    """Official DCP save (replicated tensors deduplicated across ranks)."""

    if not dcp_available():
        raise RuntimeError("torch.distributed.checkpoint is not available in this torch build")
    import torch.distributed.checkpoint as dcp

    directory.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    begin = time.perf_counter()
    state = _dcp_state_dict(model, optimizer)
    dcp.save(state, checkpoint_id=str(directory))
    dist.barrier()
    seconds = time.perf_counter() - begin
    bytes_written = float(sum(path.stat().st_size for path in sorted(directory.glob("**/*")) if path.is_file()))
    return {"mode": "dcp", "seconds": seconds, "bytes": bytes_written, "world_size": dist.get_world_size()}


def dcp_load_checkpoint(directory: Path, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer) -> dict:
    """Official DCP load; redistributes tensor reads under the new world size."""

    if not dcp_available():
        raise RuntimeError("torch.distributed.checkpoint is not available in this torch build")
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import set_state_dict

    begin = time.perf_counter()
    state = _dcp_state_dict(model, optimizer)
    dcp.load(state, checkpoint_id=str(directory))
    set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"])
    dist.barrier()
    return {"mode": "dcp", "seconds": time.perf_counter() - begin}


# ---------------------------------------------------------------------------
# single-card inference export
# ---------------------------------------------------------------------------


def export_inference_weights(model: torch.nn.Module, output: Path, config: dict) -> dict:
    """Rank 0 writes the full FP32 state dict as one safetensors file.

    This is the '最终导出单卡可推理权重' deliverable: the exported file is a
    plain Qwen3 layout (`model.embed_tokens.weight`, `model.layers.*`,
    `lm_head.weight`) loadable by the chapter-16 inference stack.
    """

    from safetensors.torch import save_file

    output.parent.mkdir(parents=True, exist_ok=True)
    state = {name: parameter.detach().to("cpu").clone() for name, parameter in model.state_dict().items()}
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("state dict contains non-tensor entries")
    begin = time.perf_counter()
    temporary = output.with_suffix(".safetensors.tmp")
    save_file(state, str(temporary), metadata={"format": "pt", "schema_version": SCHEMA_VERSION,
                                               "config": json.dumps(config, ensure_ascii=False)})
    os.replace(temporary, output)
    return {"bytes": float(output.stat().st_size), "seconds": time.perf_counter() - begin,
            "tensors": float(len(state))}
