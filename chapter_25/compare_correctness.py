#!/usr/bin/env python3
"""Chapter 25 correctness: bitwise parity, official cross-check, capacity.

Layers:

1. Full-epoch trajectory: baseline versus per-layer checkpoint, FP32 eager,
   must match bitwise in per-step loss and gradient norm.
2. Official cross-check: the same epoch driven through a model whose segments
   use ``torch.utils.checkpoint`` (``use_reentrant=False``) must also match
   the hand-written checkpoint bitwise.
3. Granularity sweep {1,2,4,7,14,28} layers per segment: one-step loss,
   gradient norm, peak memory and phase timings.
4. Capacity probes with synthetic long sequences: batch sweep at length 512
   and sequence sweep at batch 4, baseline versus per-layer checkpoint.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from activation_checkpoint import CheckpointedQwen3ForCausalLM
from experiment_utils import environment_record, write_json
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    build_batch,
    build_optimizer,
    encode_samples,
    load_checkpointed_model,
    load_tokenizer_and_model,
    make_batch_plan,
    to_device_batch,
    train_batch_step,
)


class OfficialCheckpointedModel(torch.nn.Module):
    """Same segment structure, official ``torch.utils.checkpoint`` primitive."""

    def __init__(self, checkpointed_model):
        super().__init__()
        self.inner = checkpointed_model

    def forward(self, input_ids, attention_mask=None):
        from torch.utils.checkpoint import checkpoint

        model = self.inner.model
        hidden_states = model.embed_tokens(input_ids)
        batch_size, sequence_length = input_ids.shape
        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        cosine, sine = model.rotary_emb(position_ids, hidden_states.dtype)
        causal_mask = model.make_attention_mask(
            attention_mask, sequence_length, hidden_states.dtype, hidden_states.device
        )
        layers = list(model.layers)
        step = model.layers_per_segment
        for start in range(0, len(layers), step):
            segment_layers = layers[start : start + step]

            def run_segment(current_hidden, _layers=segment_layers, _cos=cosine, _sin=sine, _mask=causal_mask):
                for layer in _layers:
                    current_hidden = layer(current_hidden, _cos, _sin, _mask)
                return current_hidden

            hidden_states = checkpoint(run_segment, hidden_states, use_reentrant=False)
        return self.inner.lm_head(model.norm(hidden_states))


def run_epoch(args, samples, plan, device, model) -> dict:
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    steps = []
    torch.cuda.reset_peak_memory_stats(device)
    for batch_indices in plan:
        batch = to_device_batch(build_batch(samples, batch_indices, args.pad_token_id), device)
        metrics = train_batch_step(model, optimizer, batch, device, args.max_grad_norm, collect_telemetry=False)
        steps.append({
            "loss": metrics["loss"],
            "gradient_norm": metrics["gradient_norm_before_clip"],
        })
    memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {"steps": steps, "memory": memory}


def synthetic_batch(tokenizer, batch_size: int, sequence_length: int, seed: int, device):
    """Capacity probes use synthetic ids: capacity, not language, is measured."""

    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        4, len(tokenizer.vocab), (batch_size, sequence_length), generator=generator
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :2] = -100
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def step_once(args, model, batch_dict, device) -> dict:
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    from mini_batch import batch_loss

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    events[0].record()
    logits = model(batch_dict["input_ids"], attention_mask=batch_dict["attention_mask"])
    loss = batch_loss(logits, batch_dict["labels"], "token_mean")
    forward_end = torch.cuda.Event(enable_timing=True)
    forward_end.record()
    loss.backward()
    events[1].record()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    events[2].record()
    torch.cuda.synchronize(device)
    result = {
        "loss": float(loss.item()),
        "gradient_norm": float(grad_norm.item()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "forward_ms": float(events[0].elapsed_time(forward_end)),
        "backward_ms": float(forward_end.elapsed_time(events[1])),
        "optimizer_ms": float(events[1].elapsed_time(events[2])),
    }
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def granularity_sweep(args, tokenizer, samples, plan, device) -> list[dict]:
    natural = to_device_batch(build_batch(samples, plan[0], args.pad_token_id), device)
    batch = {
        "input_ids": natural.input_ids,
        "attention_mask": natural.attention_mask,
        "labels": natural.labels,
    }
    reference = None
    results = []
    for segments in (None, 1, 2, 4, 7, 14, 28):
        if segments is None:
            _, model = load_tokenizer_and_model(device)
            label = "baseline"
        else:
            model = load_checkpointed_model(device, layers_per_segment=segments)
            label = f"segments={segments}"
        result = step_once(args, model, batch, device)
        if reference is None:
            reference = result
            result["bitwise_matches_baseline"] = True
        else:
            result["bitwise_matches_baseline"] = (
                result["loss"] == reference["loss"]
                and result["gradient_norm"] == reference["gradient_norm"]
            )
        if not result["bitwise_matches_baseline"]:
            raise AssertionError(f"granularity {label} diverged: {result}")
        result["configuration"] = label
        results.append(result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"granularity {label}: peak={result['peak_allocated_bytes'] / 2**30:.2f}GiB step={result['forward_ms'] + result['backward_ms'] + result['optimizer_ms']:.0f}ms", flush=True)
    return results


def capacity_probe(args, tokenizer, device, layers_per_segment) -> list[dict]:
    results = []
    for batch_size in (4, 6, 8, 12, 16):
        batch = synthetic_batch(tokenizer, batch_size, 512, args.seed, device)
        if layers_per_segment is None:
            _, model = load_tokenizer_and_model(device)
        else:
            model = load_checkpointed_model(device, layers_per_segment=layers_per_segment)
        try:
            result = step_once(args, model, batch, device)
            result.update({"batch_size": batch_size, "sequence_length": 512, "status": "success"})
        except torch.cuda.OutOfMemoryError as error:
            result = {
                "batch_size": batch_size,
                "sequence_length": 512,
                "status": "oom",
                "allocated_at_failure_bytes": torch.cuda.memory_allocated(),
                "reserved_at_failure_bytes": torch.cuda.memory_reserved(),
                "message": str(error).splitlines()[0][:160],
            }
        results.append(result)
        print(f"capacity batch={batch_size} ckpt={layers_per_segment}: {result['status']}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if result["status"] == "oom":
            break
    return results


def sequence_capacity_probe(args, tokenizer, device, layers_per_segment) -> list[dict]:
    results = []
    for sequence_length in (512, 768, 1024, 1536, 2048):
        batch = synthetic_batch(tokenizer, 4, sequence_length, args.seed, device)
        if layers_per_segment is None:
            _, model = load_tokenizer_and_model(device)
        else:
            model = load_checkpointed_model(device, layers_per_segment=layers_per_segment)
        try:
            result = step_once(args, model, batch, device)
            result.update({"batch_size": 4, "sequence_length": sequence_length, "status": "success"})
        except torch.cuda.OutOfMemoryError as error:
            result = {
                "batch_size": 4,
                "sequence_length": sequence_length,
                "status": "oom",
                "allocated_at_failure_bytes": torch.cuda.memory_allocated(),
                "reserved_at_failure_bytes": torch.cuda.memory_reserved(),
                "message": str(error).splitlines()[0][:160],
            }
        results.append(result)
        print(f"seq_capacity len={sequence_length} ckpt={layers_per_segment}: {result['status']}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if result["status"] == "oom":
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter25_train.jsonl"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("correctness-results.local.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("one CUDA GPU is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer, _ = load_tokenizer_and_model(torch.device("cpu"))
    args.pad_token_id = tokenizer.pad_token_id
    records = load_jsonl_records(args.data)
    try:
        samples = encode_samples(tokenizer, records, args.max_length)
        filtered_no_supervision = 0
    except ValueError:
        # 12GB calibration: at reduced max_length a few corpus samples lose all
        # supervised targets to truncation. Drop them (counted) so every phase
        # compares on identical samples. Same pattern as chapter_24.
        kept = []
        filtered_no_supervision = 0
        for record in records:
            try:
                kept.extend(encode_samples(tokenizer, [record], args.max_length))
            except ValueError:
                filtered_no_supervision += 1
        samples = kept
        print(
            f"encoded {len(samples)} samples, filtered_no_supervision={filtered_no_supervision}",
            flush=True,
        )
    plan = make_batch_plan(len(samples), args.batch_size, args.seed)

    # ---- 1/2. epoch trajectories, hand-written and official checkpoint ----
    _, baseline_model = load_tokenizer_and_model(device)
    baseline_epoch = run_epoch(args, samples, plan, device, baseline_model)
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    checkpoint_model = load_checkpointed_model(device, layers_per_segment=1)
    checkpoint_epoch = run_epoch(args, samples, plan, device, checkpoint_model)
    del checkpoint_model
    gc.collect()
    torch.cuda.empty_cache()

    official_model = load_checkpointed_model(device, layers_per_segment=1)
    official_wrapped = OfficialCheckpointedModel(official_model)
    official_epoch = run_epoch(args, samples, plan, device, official_wrapped)
    del official_model, official_wrapped
    gc.collect()
    torch.cuda.empty_cache()

    for name, epoch in (("checkpoint", checkpoint_epoch), ("official", official_epoch)):
        if len(epoch["steps"]) != len(baseline_epoch["steps"]):
            raise AssertionError(f"{name} trajectory length differs")
        for index, (base_step, other_step) in enumerate(zip(baseline_epoch["steps"], epoch["steps"])):
            if base_step["loss"] != other_step["loss"]:
                raise AssertionError(f"{name} loss diverged at step {index}")
            if base_step["gradient_norm"] != other_step["gradient_norm"]:
                raise AssertionError(f"{name} grad norm diverged at step {index}")

    # ---- 3. granularity sweep ----
    granularity = granularity_sweep(args, tokenizer, samples, plan, device)

    # ---- 4. capacity probes ----
    baseline_batch_capacity = capacity_probe(args, tokenizer, device, None)
    checkpoint_batch_capacity = capacity_probe(args, tokenizer, device, 1)
    baseline_seq_capacity = sequence_capacity_probe(args, tokenizer, device, None)
    checkpoint_seq_capacity = sequence_capacity_probe(args, tokenizer, device, 1)

    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype": "float32",
        "tf32": False,
        "attention_implementation": "eager",
        "seed": args.seed,
        "num_samples": len(samples),
        "filtered_no_supervision": filtered_no_supervision,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "trajectory": {
            "steps": len(baseline_epoch["steps"]),
            "first_loss": baseline_epoch["steps"][0]["loss"],
            "last_loss": baseline_epoch["steps"][-1]["loss"],
            "checkpoint_bitwise_identical": True,
            "official_checkpoint_bitwise_identical": True,
        },
        "memory": {
            "baseline_epoch": baseline_epoch["memory"],
            "checkpoint_epoch": checkpoint_epoch["memory"],
        },
        "granularity_sweep": granularity,
        "capacity_batch_sweep": {
            "baseline": baseline_batch_capacity,
            "checkpoint_layers_per_segment_1": checkpoint_batch_capacity,
        },
        "capacity_sequence_sweep": {
            "baseline": baseline_seq_capacity,
            "checkpoint_layers_per_segment_1": checkpoint_seq_capacity,
        },
        "status": "pass",
    })
    write_json(args.output, payload)
    print("chapter_25 correctness checks passed")


if __name__ == "__main__":
    main()
