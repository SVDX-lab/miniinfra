"""Chapter 38 runtime glue: data pipeline and process-group plumbing.

Everything here is chapter-independent infrastructure inherited from the
Distributed Data Parallel chapters: deterministic corpus encoding, the
Chapter 22 shuffle, env:// style process-group setup, and padded batch
assembly.  The chapter mechanism itself lives in ``sequence_parallel.py`` on
top of ``attention_tp.py`` / ``mlp_tp.py``; the TP-arm loaders live in
``qwen3_model.py`` and the SP loader in ``sequence_parallel.py``.

The one genuinely TP-specific piece is ``build_replicated_batch``: a TP group
does NOT shard data, so every rank receives the whole global batch and the
supervised-token loss scale is exactly 1.0 (kept in the accounting for
interface parity with the DDP/ZeRO chapters).  Sequence Parallel inherits
this: ranks shard ACTIVATIONS along the sequence dim, never the dataset.
"""

from __future__ import annotations

import os
import random
import socket
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist

from assistant_loss import EncodedConversation
from mini_batch import PaddedBatch, pad_samples
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model_atp,
    load_handwritten_model_tp,
    resolve_model_directory,
)
from qwen3_tokenizer import Qwen3Tokenizer


MODEL_ID = DEFAULT_MODEL_ID
MODEL_REVISION = DEFAULT_MODEL_REVISION


def load_tokenizer(device: torch.device) -> Qwen3Tokenizer:
    directory = resolve_model_directory(MODEL_ID, MODEL_REVISION)
    return Qwen3Tokenizer(directory)


def load_tokenizer_and_model_tp(device: torch.device, tp_rank: int, tp_size: int, group=None,
                                attention_parallel: bool = False):
    """The chapter-36 baseline layout(s), used by the ``tp_wN`` arm.

    ``attention_parallel=True`` loads attention AND MLP shards — the pure TP
    baseline arm of THIS chapter (replicated activations).  The False variant
    (MLP shards only) is kept for interface parity with chapter 20/21 tooling;
    the chapter-38 arms do not use it.
    """

    directory = resolve_model_directory(MODEL_ID, MODEL_REVISION)
    loader = load_handwritten_model_atp if attention_parallel else load_handwritten_model_tp
    return Qwen3Tokenizer(directory), loader(
        directory, device, tp_rank, tp_size, dtype=torch.float32, group=group
    )


def load_tokenizer_and_model_sp(device: torch.device, tp_rank: int, tp_size: int, group=None):
    """This chapter's variant layout: attention+MLP shards PLUS sequence sharding.

    Same weight shards as the ``tp_wN`` arm; only the activation layout and
    the collective operators between the TP regions differ (see
    ``sequence_parallel.load_handwritten_model_sp``).
    """

    from sequence_parallel import load_handwritten_model_sp

    directory = resolve_model_directory(MODEL_ID, MODEL_REVISION)
    return Qwen3Tokenizer(directory), load_handwritten_model_sp(
        directory, device, tp_rank, tp_size, dtype=torch.float32, group=group
    )


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


def setup_distributed(rank: int | None = None, world_size: int | None = None, backend: str = "nccl") -> tuple[int, int, torch.device]:
    """Process-group setup; each rank binds its LOCAL CUDA device.

    The TP group of this chapter is the whole world, so tp_rank == rank and
    tp_size == world_size.  Rank and world size come from arguments or from
    RANK/WORLD_SIZE (env:// launchers); the CUDA device always follows
    LOCAL_RANK so the same code runs under mp.spawn and torchrun.
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


def build_replicated_batch(
    samples: Sequence[EncodedConversation],
    batch_members: tuple[int, ...],
    pad_token_id: int,
) -> tuple[PaddedBatch, dict[str, float]]:
    """Pad the WHOLE global batch; every TP rank runs this identical batch.

    TP splits weights, not data: ``supervised_global`` equals the local count
    and ``loss_scale`` is exactly 1.0.  The fields mirror the DDP accounting
    so the step functions and result schemas stay comparable across chapters.
    """

    batch = pad_samples([samples[index] for index in batch_members], pad_token_id)
    supervised = float(batch.supervised_tokens)
    accounting = {
        "local_members": tuple(batch_members),
        "global_members": tuple(batch_members),
        "supervised_local": supervised,
        "supervised_global": supervised,
        "loss_scale": 1.0,
        "physical_tokens_local": float(batch.physical_tokens),
        "non_padding_tokens_local": float(batch.non_padding_tokens),
    }
    return batch, accounting
