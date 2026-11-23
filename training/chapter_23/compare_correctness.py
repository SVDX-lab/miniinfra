#!/usr/bin/env python3
"""Chapter 23 correctness: FP32 anchor, BF16 divergence accounting, dtype census.

Layers:

1. FP32 anchor: the baseline arm re-runs the Chapter 22 step body on the same
   batch plan; the first-step loss must reproduce the Chapter 22 recorded
   value (bitwise when kernels are deterministic across processes).
2. Gradient parity: one backward on identical weights and batch, per-parameter
   absolute/relative error and cosine similarity between FP32 and BF16 grads.
3. One optimizer step: parameter deltas of both arms against the same init.
4. Epoch trajectory: full-epoch loss/grad-norm trajectories, divergence
   summary, and a NaN/Inf sweep after every step.
5. Dtype census on CUDA: master parameters, gradients and AdamW states stay
   FP32 in the autocast arm, while linear outputs run BF16.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from experiment_utils import environment_record, write_json
from mixed_precision import (
    assert_no_nan_or_inf,
    autocast_policy_census,
    compare_trajectories,
    state_dtype_census,
)
from sft_core import DATASET_FILE, DATASET_REPO, DATASET_REVISION, DATASET_SHA256, load_jsonl_records, seed_everything
from training import (
    MODEL_ID,
    MODEL_REVISION,
    autocast_context,
    build_batch,
    build_optimizer,
    encode_samples,
    load_tokenizer_and_model,
    make_batch_plan,
    train_batch_step,
)

# 第 22 期在同机同版本环境下记录的整 Epoch 轨迹首步 Loss（FP32、同种子、同批次计划）。
# RTX 3080 Ti 12GB 校准：锚定值按运行长度查表，两个值都来自第 22 期同配置实测
# （batch 1、sft_mini_512 @9b06bee、seed 20260912）。L512 峰值贴近 11.63GB 可用
# 显存，整 Epoch 偶发 OOM；默认档取 L384。历史值：旧 24GB 平台 4.247612953186035。
CHAPTER07_ANCHOR_BY_LENGTH = {
    384: 4.1608357429504395,
    512: 4.102053642272949,
}


def load_corpus_and_batches(args, device_for_tokenizer):
    tokenizer, _ = load_tokenizer_and_model(device_for_tokenizer)
    records = load_jsonl_records(args.data)
    samples = encode_samples(tokenizer, records, args.max_length)
    plan = make_batch_plan(len(samples), args.batch_size, args.seed)
    return tokenizer, samples, plan


def fresh_training(args, samples, plan, device, mode):
    seed_everything(args.seed)
    tokenizer, model = load_tokenizer_and_model(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    return tokenizer, model, optimizer


def gradient_parity(args, samples, plan, device) -> dict:
    """One backward on identical weights; compare FP32 vs BF16 gradients."""

    grads: dict[str, dict[str, torch.Tensor]] = {}
    norms: dict[str, float] = {}
    init: dict[str, torch.Tensor] = {}
    batch = build_batch(samples, plan[0], pad_token_id=args.pad_token_id)
    batch = type(batch)(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        labels=batch.labels.to(device),
        source_lines=batch.source_lines,
        lengths=batch.lengths,
        supervised_tokens_per_sample=batch.supervised_tokens_per_sample,
    )
    for mode in ("fp32", "bf16_autocast"):
        seed_everything(args.seed)
        _, model, _ = fresh_training(args, samples, plan, device, mode)
        model.train()
        with autocast_context(mode, device):
            logits = model(batch.input_ids, attention_mask=batch.attention_mask)
            from mini_batch import batch_loss

            loss = batch_loss(logits, batch.labels, "token_mean")
        loss.backward()
        selected = {}
        for name, parameter in model.named_parameters():
            if name in SELECTED_PARAMETERS:
                selected[name] = parameter.grad.detach().clone()
        grads[mode] = selected
        norms[mode] = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
        )
        if mode == "fp32":
            # Keep the init snapshot on the CPU: a GPU-resident full copy
            # (2.4 GB) overlaps with the next arm's backward and overflows 12 GB.
            init = {name: parameter.detach().to("cpu").clone() for name, parameter in model.named_parameters()}
        # The stashed grads are enough; release the full gradient before the
        # next arm so its backward has room.
        model.zero_grad(set_to_none=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    parameters_report = {}
    for name in SELECTED_PARAMETERS:
        reference = grads["fp32"][name]
        variant = grads["bf16_autocast"][name]
        error = (reference - variant).abs()
        relative = error / reference.abs().clamp_min(1e-12)
        # 逐元素相对误差会被接近零的梯度元素放大；significant 阈值过滤后
        # 反映真实数值精度的量级。
        significant = reference.abs() > (1e-3 * reference.abs().max())
        relative_significant = error[significant] / reference.abs()[significant]
        cosine = torch.nn.functional.cosine_similarity(
            reference.flatten().double(), variant.flatten().double(), dim=0
        )
        parameters_report[name] = {
            "max_abs_error": float(error.max()),
            "max_rel_error": float(relative.max()),
            "max_rel_error_significant": float(relative_significant.max()),
            "cosine_similarity": float(cosine),
            "norm_ratio": float(variant.norm() / reference.norm().clamp_min(1e-12)),
        }
    return {"selected_parameters": parameters_report, "global_norm": norms}


SELECTED_PARAMETERS = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.13.mlp.down_proj.weight",
    "model.layers.27.mlp.down_proj.weight",
    "model.norm.weight",
)


def one_step_parameters(args, samples, plan, device) -> dict:
    """Both arms train from the same init for exactly one accumulated step on
    batch 0.

    12GB calibration: the step is split into per-sample micro-steps whose
    gradients sum to the same global token_mean objective (the chapter 09
    weighting), and only the compared parameters are snapshotted, on the CPU.
    A GPU-resident clone of every parameter for both arms (about 9.6 GB) does
    not coexist with the optimizer states at L512."""

    from mini_batch import batch_loss

    endpoints = {}
    members = plan[0]
    supervised_total = sum(samples[index].supervised_tokens for index in members)
    for mode in ("fp32", "bf16_autocast"):
        _, model, optimizer = fresh_training(args, samples, plan, device, mode)
        init = {
            name: parameter.detach().to("cpu").clone()
            for name, parameter in model.named_parameters()
            if name in SELECTED_PARAMETERS
        }
        optimizer.zero_grad(set_to_none=True)
        for index in members:
            batch = build_batch(samples, (index,), pad_token_id=args.pad_token_id)
            with autocast_context(mode, device):
                logits = model(batch.input_ids.to(device), attention_mask=batch.attention_mask.to(device))
                loss = batch_loss(logits, batch.labels.to(device), "token_mean")
            scale = float(batch.supervised_tokens_per_sample[0]) / supervised_total
            (loss * scale).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        del grad_norm
        endpoints[mode] = (
            init,
            {
                name: parameter.detach().to("cpu").clone()
                for name, parameter in model.named_parameters()
                if name in SELECTED_PARAMETERS
            },
        )
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    init, fp32_end = endpoints["fp32"]
    _init_b, bf16_end = endpoints["bf16_autocast"]
    report = {}
    for name in SELECTED_PARAMETERS:
        reference = fp32_end[name]
        variant = bf16_end[name]
        delta_reference = reference - init[name]
        delta_variant = variant - init[name]
        cosine = torch.nn.functional.cosine_similarity(
            delta_reference.flatten().double(), delta_variant.flatten().double(), dim=0
        )
        report[name] = {
            "param_max_abs_error": float((reference - variant).abs().max()),
            "param_max_rel_error": float(((reference - variant).abs() / reference.abs().clamp_min(1e-12)).max()),
            "delta_cosine_similarity": float(cosine),
            "delta_norm_ratio": float(
                delta_variant.norm() / delta_reference.norm().clamp_min(1e-12)
            ),
        }
    return report


def run_epoch(args, samples, plan, device, mode) -> dict:
    _, model, optimizer = fresh_training(args, samples, plan, device, mode)
    steps = []
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for batch_indices in plan:
            batch = build_batch(samples, batch_indices, pad_token_id=args.pad_token_id)
            metrics = train_batch_step(
                model, optimizer, batch, device, args.max_grad_norm, mode=mode, collect_telemetry=False
            )
            assert_no_nan_or_inf(model)
            if len(steps) % 64 == 0:
                print(f'epoch {mode} step {len(steps)} alloc', torch.cuda.memory_allocated(device)/2**30, flush=True)
            step_record = {
                "loss": metrics["loss"],
                "gradient_norm": metrics["gradient_norm_before_clip"],
                "supervised_tokens": metrics["supervised_tokens"],
            }
            # 12GB margin is thin at L512: release every step's Python-side
            # references so allocator blocks can be reused, not accumulated.
            del batch, metrics
            gc.collect()
            torch.cuda.empty_cache()
            steps.append(step_record)
        memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return {"mode": mode, "steps": steps, "memory": memory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter23_train.jsonl"))
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

    tokenizer_cpu, samples, plan = load_corpus_and_batches(args, torch.device("cpu"))
    args.pad_token_id = tokenizer_cpu.pad_token_id

    # ---- 1. FP32 anchor against Chapter 22 ----
    print('PHASE fp32_epoch begin', torch.cuda.memory_allocated()/2**30, flush=True)
    fp32_epoch = run_epoch(args, samples, plan, device, "fp32")
    first_loss = fp32_epoch["steps"][0]["loss"]
    chapter22_anchor = CHAPTER07_ANCHOR_BY_LENGTH[args.max_length]
    anchor_abs_error = abs(first_loss - chapter22_anchor)
    if anchor_abs_error > 1e-5:
        raise AssertionError(
            f"FP32 anchor drifted from Chapter 22: {first_loss} vs {chapter22_anchor}"
        )

    # ---- 2/3. gradient parity and one-step parameters ----
    gc.collect()
    print('PHASE parity begin', torch.cuda.memory_allocated()/2**30, flush=True)
    torch.cuda.empty_cache()
    parity = gradient_parity(args, samples, plan, device)
    one_step = one_step_parameters(args, samples, plan, device)

    # ---- 4. BF16 epoch trajectory ----
    gc.collect()
    print('PHASE bf16_epoch begin', torch.cuda.memory_allocated()/2**30, flush=True)
    torch.cuda.empty_cache()
    bf16_epoch = run_epoch(args, samples, plan, device, "bf16_autocast")
    loss_summary = compare_trajectories(
        [step["loss"] for step in fp32_epoch["steps"]],
        [step["loss"] for step in bf16_epoch["steps"]],
    )
    norm_summary = compare_trajectories(
        [step["gradient_norm"] for step in fp32_epoch["steps"]],
        [step["gradient_norm"] for step in bf16_epoch["steps"]],
    )

    # ---- 5. dtype census on the real model ----
    gc.collect()
    print('PHASE census begin', torch.cuda.memory_allocated()/2**30, flush=True)
    torch.cuda.empty_cache()
    _, model, optimizer = fresh_training(args, samples, plan, device, "bf16_autocast")
    batch = build_batch(samples, plan[0], args.pad_token_id)
    batch = type(batch)(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        labels=batch.labels.to(device),
        source_lines=batch.source_lines,
        lengths=batch.lengths,
        supervised_tokens_per_sample=batch.supervised_tokens_per_sample,
    )

    def forward_and_loss():
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        from mini_batch import batch_loss

        return batch_loss(logits, batch.labels, "token_mean")

    census = autocast_policy_census(
        model,
        forward_and_loss,
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    )
    from mini_batch import batch_loss

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean")
    loss.backward()
    # AdamW 状态在第一次 step 时才分配；先走一步再普查状态 dtype。
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    states = state_dtype_census(model, optimizer)
    del model, optimizer, loss, logits, grad_norm
    gc.collect()
    torch.cuda.empty_cache()

    payload = environment_record()
    payload.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": DATASET_FILE, "sha256": DATASET_SHA256},
        "dtype_arms": ["float32", "bf16_autocast"],
        "tf32": False,
        "attention_implementation": "eager",
        "seed": args.seed,
        "num_samples": len(samples),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "supervised_tokens_total": sum(step["supervised_tokens"] for step in fp32_epoch["steps"]),
        "fp32_anchor": {
            "chapter22_first_loss": CHAPTER07_ANCHOR_BY_LENGTH[args.max_length],
            "measured_first_loss": first_loss,
            "abs_error": anchor_abs_error,
        },
        "gradient_parity": parity,
        "one_step_parameters": one_step,
        "trajectory_loss": loss_summary,
        "trajectory_gradient_norm": norm_summary,
        "trajectory_loss_series": {
            "fp32": [step["loss"] for step in fp32_epoch["steps"]],
            "bf16_autocast": [step["loss"] for step in bf16_epoch["steps"]],
        },
        "trajectory_gradient_norm_series": {
            "fp32": [step["gradient_norm"] for step in fp32_epoch["steps"]],
            "bf16_autocast": [step["gradient_norm"] for step in bf16_epoch["steps"]],
        },
        "dtype_census": {
            "module_outputs": {name: census[name] for name in (
                "model.embed_tokens",
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn",
                "model.layers.0.mlp.down_proj",
                "model.norm",
                "<loss>",
            ) if name in census},
            "state_dtypes": states,
        },
        "memory": {
            "fp32": fp32_epoch["memory"],
            "bf16_autocast": bf16_epoch["memory"],
        },
        "status": "pass",
    })
    write_json(args.output, payload)
    print("chapter_23 correctness checks passed")


if __name__ == "__main__":
    main()
