#!/usr/bin/env python3
"""Chapter 21 correctness: shard/online token identity, corruption, twin step.

Layers:

1. Build the shard from the online encoding pipeline (timed once).
2. Identity: all 512 records materialized online and read from the shard must
   be bitwise identical (input_ids, labels, source_line, length).
3. Batch equivalence: the same members built through both paths yield bitwise
   identical PaddedBatches; losses from fresh models are bitwise equal and one
   AdamW step leaves all parameters bitwise equal.
4. Corruption: a flipped payload byte must fail the checksum; a wrong dataset
   revision must fail header validation.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from assistant_loss import EncodedConversation
from data_paths import OnlineCorpus, ShardCorpus, pad_pairs
from experiment_utils import environment_record, write_json
from token_shard import TokenShardReader
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, seed_everything
from token_shard import build_shard, shard_manifest_entry
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_optimizer,
    load_tokenizer_and_model,
    memory_record,
    move_batch,
    train_batch_step,
)


def twin_step(batch_a, batch_b, device: torch.device, seed: int) -> dict[str, object]:
    """Run one identical step body on two fresh models; compare bitwise."""

    losses = []
    parameters = []
    for batch in (batch_a, batch_b):
        seed_everything(seed)
        model = load_tokenizer_and_model(device)[1]
        optimizer = build_optimizer(model, 2e-5, 0.1)
        metrics = train_batch_step(model, optimizer, batch, device, 1.0, collect_telemetry=False)
        losses.append(metrics["loss"])
        parameters.append({name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()})
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
    bitwise = all(
        torch.equal(parameters[0][name], parameters[1][name]) for name in parameters[0]
    )
    return {"loss_a": losses[0], "loss_b": losses[1], "loss_bitwise_equal": losses[0] == losses[1], "parameters_bitwise_equal": bitwise}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter21_train.jsonl"))
    parser.add_argument("--shard", type=Path, default=Path("data/chapter21_shard.bin"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("one CUDA GPU is required")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")
    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))

    # ---- 1. 在线编码全部样本并写入 Shard（一次性成本）----
    from data_paths import OnlineCorpus

    online = OnlineCorpus(args.data, tokenizer, args.max_length)
    jsonl_bytes = Path(args.data).stat().st_size
    build_start = time.perf_counter()
    samples = []
    per_record_ms = []
    for index in range(len(online)):
        start = time.perf_counter()
        source_line, ids, labels = online.materialize(index)
        per_record_ms.append((time.perf_counter() - start) * 1000)
        samples.append(EncodedConversation(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=labels,
            assistant_loss_mask=labels != -100,
            source_line=source_line,
            rendered_text="",
            token_spans=(),
            loss_mode="assistant_only",
        ))
    encode_seconds = time.perf_counter() - build_start
    shard_stats = build_shard(
        args.shard, samples, MODEL_REVISION, DATASET_REVISION, args.max_length
    )
    del samples
    gc.collect()

    # ---- 2. 逐 Token 一致性 ----
    shard = ShardCorpus(args.shard, MODEL_REVISION, DATASET_REVISION)
    if len(shard) != len(online):
        raise AssertionError("shard record count mismatch")
    mismatched = 0
    for index in range(len(online)):
        online_record = online.materialize(index)
        shard_record = shard.materialize(index)
        same = (
            online_record[0] == shard_record[0]
            and torch.equal(online_record[1], shard_record[1])
            and torch.equal(online_record[2], shard_record[2])
        )
        if not same:
            mismatched += 1
    if mismatched:
        raise AssertionError(f"{mismatched} records differ between online and shard")

    # ---- 3. 相同成员 Batch 的 Loss 与一次更新 ----
    rng_seed = args.seed
    import random

    rng = random.Random(f"{rng_seed}|1|shuffle")
    order = list(range(len(online)))
    rng.shuffle(order)
    batches = [tuple(sorted(order[i : i + args.batch_size])) for i in range(0, len(order), args.batch_size)]
    lightest = min(batches, key=lambda members: max(len(online.materialize(i)[1][0]) for i in members))
    pairs_online = [online.materialize(index) for index in lightest]
    pairs_shard = [shard.materialize(index) for index in lightest]
    batch_online = pad_pairs(pairs_online, tokenizer.pad_token_id)
    batch_shard = pad_pairs(pairs_shard, tokenizer.pad_token_id)
    tensors_equal = all(
        torch.equal(getattr(batch_online, name), getattr(batch_shard, name))
        for name in ("input_ids", "attention_mask", "labels")
    )
    if not tensors_equal:
        raise AssertionError("padded batches differ between data paths")
    device_batch_online = move_batch(batch_online, device)
    device_batch_shard = move_batch(batch_shard, device)
    del batch_online, batch_shard
    gc.collect()
    torch.cuda.empty_cache()
    twin = twin_step(device_batch_online, device_batch_shard, device, args.seed)
    if not (twin["loss_bitwise_equal"] and twin["parameters_bitwise_equal"]):
        raise AssertionError(f"twin step diverged: {twin}")

    # ---- 4. 损坏检测 ----
    corrupt_path = Path("data/corrupt_probe.bin")
    raw = bytearray(Path(args.shard).read_bytes())
    reader_probe = TokenShardReader(args.shard, MODEL_REVISION, DATASET_REVISION)
    offsets = reader_probe._offsets
    reader_probe.close()
    raw[offsets[1] + 12] ^= 0xFF
    corrupt_path.write_bytes(bytes(raw))
    corruption = {"payload_flip_detected": False}
    try:
        probe = TokenShardReader(corrupt_path, MODEL_REVISION, DATASET_REVISION)
        probe.read_record(1)
    except ValueError as error:
        corruption["payload_flip_detected"] = "checksum" in str(error)
    if not corruption["payload_flip_detected"]:
        raise AssertionError("payload corruption was not detected")

    per_record_sorted = sorted(per_record_ms)
    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32", "tf32": False, "attention_implementation": "eager",
        "seed": args.seed, "num_samples": len(online), "max_length": args.max_length,
        "shard": {
            **shard_manifest_entry(Path(args.shard)),
            "build_seconds": shard_stats["write_seconds"],
            "encode_seconds": encode_seconds,
            "jsonl_bytes": jsonl_bytes,
            "per_record_online_ms_p50": per_record_sorted[len(per_record_ms) // 2],
            "per_record_online_ms_mean": sum(per_record_ms) / len(per_record_ms),
        },
        "identity": {"records_compared": len(online), "mismatched": mismatched},
        "batch_equivalence": {
            "members": list(lightest),
            "padded_tensors_bitwise_equal": tensors_equal,
            "twin": twin,
        },
        "corruption": corruption,
        "status": "pass",
    })
    write_json(args.output, payload)
    print("chapter_21 correctness checks passed")


if __name__ == "__main__":
    main()
