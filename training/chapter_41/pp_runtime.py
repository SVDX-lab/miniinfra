"""Chapter 41 runtime glue: data pipeline and process-group plumbing.

Inherited from the DDP/TP chapters: deterministic corpus encoding, the
Chapter 22 shuffle, env:// style process-group setup, and padded batch
assembly.  The chapter mechanism (virtual stages + the interleaved 1F1B
schedule) lives in ``interleaved.py``; the data rule is identical to
chapters 24/25 because the placement/schedule is the only variable.  The
world size is the PHYSICAL pipeline size here; total virtual stages are
``world_size * vpp``.

Pipeline-specific data rule: with ``num_stages`` ranks and ``num_micro``
micro-batches of ``samples_per_micro`` samples each, the global batch is
``num_micro * samples_per_micro`` and EVERY rank assembles the SAME
micro-batches from the same deterministic plan.  Stages k > 0 never touch the
token ids (their layers consume received activations) but they still need the
attention masks, the supervised-token counts and the per-micro loss scales —
all computable locally, so the P2P payload stays activations-only.

World size IS the number of stages; rank r hosts stage r.
"""

from __future__ import annotations

import os
import random
import socket
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist

from assistant_loss import EncodedConversation, IGNORE_INDEX
from mini_batch import PaddedBatch, pad_samples
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer


MODEL_ID = DEFAULT_MODEL_ID
MODEL_REVISION = DEFAULT_MODEL_REVISION


def load_tokenizer(device: torch.device) -> Qwen3Tokenizer:
    directory = resolve_model_directory(MODEL_ID, MODEL_REVISION)
    return Qwen3Tokenizer(directory)


def setup_distributed(rank: int | None = None, world_size: int | None = None,
                      backend: str = "nccl") -> tuple[int, int, torch.device]:
    """Process-group setup; each rank binds its LOCAL CUDA device.

    Rank and world size come from arguments or from RANK/WORLD_SIZE (env://
    launchers); the CUDA device always follows LOCAL_RANK so the same code
    runs under mp.spawn and torchrun.
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


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


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


def build_micro_batches(samples: Sequence[EncodedConversation], batch_members: tuple[int, ...],
                        num_micro: int, pad_token_id: int) -> list[PaddedBatch]:
    """One global batch -> ``num_micro`` padded micro-batches.

    The batch members are split row-wise AFTER padding to the global batch
    maximum, so every micro-batch shares one sequence axis (comparable P2P
    shapes across micro-batches; shapes are known on every rank without
    communication because the plan is deterministic).
    """

    batch = pad_samples([samples[index] for index in batch_members], pad_token_id)
    if batch.batch_size % num_micro != 0:
        raise ValueError(f"global batch {batch.batch_size} is not divisible by num_micro {num_micro}")
    per_micro = batch.batch_size // num_micro
    micros = []
    for index in range(num_micro):
        rows = slice(index * per_micro, (index + 1) * per_micro)
        micros.append(PaddedBatch(
            input_ids=batch.input_ids[rows].clone(),
            attention_mask=batch.attention_mask[rows].clone(),
            labels=batch.labels[rows].clone(),
            source_lines=batch.source_lines[rows],
            lengths=batch.lengths[rows],
            supervised_tokens_per_sample=batch.supervised_tokens_per_sample[rows],
        ))
    return micros


def micro_dicts(micro_batches: Sequence[PaddedBatch], supervised_global: int) -> list[dict]:
    """Attach the per-micro loss scales used by the pipeline step functions."""

    dicts = []
    for batch in micro_batches:
        supervised = float((batch.labels[:, 1:] != IGNORE_INDEX).sum())
        dicts.append({
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "labels": batch.labels,
            "batch_size": batch.batch_size,
            "sequence_length": batch.input_ids.shape[1],
            "physical_tokens": float(batch.physical_tokens),
            "supervised_tokens": supervised,
            "scale": supervised / float(supervised_global),
        })
    return dicts


def gpu_telemetry(rank: int) -> dict[str, float]:
    """Local-GPU power/temperature/clock; the physical GPU follows LOCAL_RANK."""

    import subprocess

    local_id = int(os.environ.get("LOCAL_RANK", rank))
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits", f"--id={local_id}"],
        check=True, capture_output=True, text=True,
    )
    power, temperature, clock = result.stdout.strip().split(",")
    return {"power_watts": float(power), "temperature_celsius": float(temperature), "sm_clock_mhz": float(clock)}


def parameter_count_bytes() -> int:
    """Full-model parameter count read from safetensors headers (no weights)."""

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
