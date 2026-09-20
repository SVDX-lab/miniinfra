"""Chapter 42 runtime glue: process groups, data pipeline, node mapping.

Inherited from the DDP/TP/PP chapters: deterministic corpus encoding, the
Chapter 22 shuffle, env:// style process-group setup, and padded batch
assembly.  The chapter mechanism lives in ``hybrid_parallel.py``.

The one genuinely hybrid-specific piece is the DATA RULE: data is sharded
ONLY along DP.  The global batch (``dp_size * num_micro * samples_per_micro``
samples, chapter-22 shuffle plan) is stride-split across DP ranks
(``members[dp_rank::dp_size]``, chapter 14's rule); a DP rank's shard is then
split into ``num_micro`` contiguous micro-batches that flow through the
pipeline, and every rank of the rank's TP/PP groups assembles the SAME
micro-batches from the same deterministic plan — TP splits weights, not data,
and stages k > 0 need only masks/shapes/scales, so the P2P payload stays
activations-only.  The pipeline micro loop IS the gradient accumulation:
global batch = num_micro x samples_per_micro x dp_size.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import socket
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist

from assistant_loss import IGNORE_INDEX, EncodedConversation
from hybrid_parallel import ParallelTopology
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
    launchers, torchrun on one node or two); the CUDA device always follows
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


def anonymous_node_tag() -> str:
    """A stable per-host identifier that leaks no hostname: sha256(tag)[0:8]."""

    hostname = platform.node()
    return hashlib.sha256(hostname.encode("utf-8")).hexdigest()[:8] if hostname else "unknown-host"


def resolve_node_index() -> tuple[int, str]:
    """This rank's node index (first-seen order of anonymous host tags).

    Dual-node runs get node ids without publishing hostnames into results.
    """

    tag = anonymous_node_tag()
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, tag)
    ordered = sorted({item for item in gathered if item is not None})
    return ordered.index(tag), tag


def node_mapping_record(topology: ParallelTopology, require_dp_alignment: bool) -> dict:
    """Rank -> (node index, anonymous tag, local rank) table for the results.

    For the dual-node arm the layout assumes one full TP x PP replica per
    node, i.e. node_index == dp_rank.  A cluster that violates this turns the
    "TP/PP intra-node, DP cross-node" claim false, so the check refuses
    unless ``CHAPTER27_ALLOW_CROSS_NODE_TP_PP=1`` is set (the run is then
    flagged in the results and its numbers lose the topology interpretation).
    """

    node_index, tag = resolve_node_index()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    record = {
        "node_index": node_index,
        "node_tag": tag,
        "local_rank": local_rank,
    }
    gathered: list[dict | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, record)
    table = {}
    aligned = True
    for rank, item in enumerate(gathered):
        dp_rank, _, _ = topology.decompose(rank)
        aligned = aligned and item["node_index"] == dp_rank
        table[str(rank)] = {
            "dp_rank": dp_rank,
            "pp_rank": topology.decompose(rank)[1],
            "tp_rank": topology.decompose(rank)[2],
            "node_index": item["node_index"],
            "local_rank": item["local_rank"],
        }
    if require_dp_alignment and not aligned:
        raise RuntimeError(
            "node layout does not match dp_rank (one TP x PP replica per node); "
            "set CHAPTER27_ALLOW_CROSS_NODE_TP_PP=1 to record it as a case study"
        )
    return {
        "node_count": len({item["node_index"] for item in gathered}),
        "node_tags": sorted({item["node_tag"] for item in gathered}),
        "rank_table": table,
        "node_mapping_matches_dp": aligned,
        "topology_note": "TP/PP groups intra-node, DP groups cross-node (experiment D layout)" if aligned
        else "node layout deviates from the dp_rank mapping; topology interpretation invalid",
    }


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


def global_batch_members(num_samples: int, topology: ParallelTopology,
                         num_micro: int, samples_per_micro: int, seed: int) -> tuple[int, ...]:
    """The global batch of one optimizer step: dp x micros x per-micro samples."""

    size = topology.dp_size * num_micro * samples_per_micro
    if size > num_samples:
        raise ValueError(f"global batch {size} exceeds the corpus ({num_samples} samples)")
    plan = make_batch_plan(num_samples, size, seed)
    return plan[0]


def shard_micro_batches(
    samples: Sequence[EncodedConversation],
    global_members: tuple[int, ...],
    topology: ParallelTopology,
    dp_rank: int,
    num_micro: int,
    pad_token_id: int,
) -> tuple[list[PaddedBatch], dict[str, float]]:
    """THIS DP rank's micro-batches plus the supervised-token accounting.

    ``members[dp_rank::dp_size]`` (chapter 14's stride rule) partitions the
    global batch exactly; the rank shard is split into ``num_micro``
    contiguous micro-batches.  The loss scale of every micro is
    ``supervised_micro / supervised_global`` so the DP-side AllReduce SUM
    reproduces the global ``token_mean`` objective.
    """

    expected_local = len(global_members) // topology.dp_size
    if len(global_members) % topology.dp_size != 0 or expected_local % num_micro != 0:
        raise ValueError(
            f"global batch {len(global_members)} must split evenly into dp {topology.dp_size}"
            f" x micro {num_micro}"
        )
    local_members = tuple(sorted(global_members[dp_rank::topology.dp_size]))
    per_micro = expected_local // num_micro
    supervised_global = float(sum(samples[index].supervised_tokens for index in global_members))
    micro_batches: list[PaddedBatch] = []
    for index in range(num_micro):
        chunk = local_members[index * per_micro : (index + 1) * per_micro]
        micro_batches.append(pad_samples([samples[i] for i in chunk], pad_token_id))
    accounting = {
        "local_members": local_members,
        "global_members": tuple(global_members),
        "supervised_global": supervised_global,
        "supervised_local": float(sum(batch.supervised_tokens for batch in micro_batches)),
        "global_batch_sequences": float(len(global_members)),
    }
    return micro_batches, accounting


def micro_dicts(micro_batches: Sequence[PaddedBatch], supervised_global: float) -> list[dict]:
    """Attach the per-micro loss scales used by the pipeline step function."""

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
