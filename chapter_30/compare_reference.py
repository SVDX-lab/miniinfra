"""Chapter 30 shared Phase A: the big-batch reference trajectory on cuda:0.

Identical to Chapter 29's reference path: the Chapter 28 stable body over the
global batch, per-step loss / gradient norm / parameter fingerprints, with
step-0 gradients stashed for the tolerance comparison.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from ddp_core import load_tokenizer_and_model
from mini_batch import batch_loss, pad_samples
from sft_core import seed_everything


def fingerprint(model: torch.nn.Module) -> dict:
    return {name: float(parameter.detach().double().sum()) for name, parameter in model.named_parameters()}


def run_reference(args, samples, plan, device) -> dict:
    seed_everything(args.seed)
    _, model = load_tokenizer_and_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, foreach=False, fused=False)

    steps = []
    for step_index in range(args.steps):
        host_batch = pad_samples([samples[index] for index in plan[step_index]], args.pad_token_id)
        batch = type(host_batch)(
            input_ids=host_batch.input_ids.to(device),
            attention_mask=host_batch.attention_mask.to(device),
            labels=host_batch.labels.to(device),
            source_lines=host_batch.source_lines,
            lengths=host_batch.lengths,
            supervised_tokens_per_sample=host_batch.supervised_tokens_per_sample,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = batch_loss(logits, batch.labels, "token_mean")
        loss.backward()
        if step_index == 0:
            # Stash BEFORE clipping: the gradient of the objective is the
            # comparison target; clipping is a training-step concern.
            stashed = {name: parameter.grad.detach().to("cpu").clone() for name, parameter in model.named_parameters()}
            torch.save({"grads": stashed}, args.stash_path)
            del stashed
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if step_index == 0:
            stashed_norm = float(grad_norm.item())
        optimizer.step()
        steps.append({
            "step": step_index,
            "loss": float(loss.item()),
            "gradient_norm": float(grad_norm.item()),
            "fingerprint": fingerprint(model),
        })
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return {"steps": steps, "stashed_grad_norm": stashed_norm}

def gradient_report(model, stashed_path, clip_to=None):
    """Scale-aware comparison of gradients against the stashed big-batch reference.

    Raw elementwise relative error is dominated by near-zero entries, so the
    primary bar is per-tensor: max abs difference divided by the tensor's max
    absolute value. The global cosine similarity and two absolute measures
    are reported alongside. ``clip_to`` emulates the reference path's
    ``clip_grad_norm_`` for arms whose gradients were captured post-clip.
    """

    bundle = torch.load(stashed_path, map_location="cpu")
    stashed = bundle["grads"]
    if clip_to is not None:
        # Emulate the reference path's gradient clipping: scale the stashed
        # pre-clip gradients the way clip_grad_norm_ would have.
        stash_norm = sum(float(tensor.double().pow(2).sum()) for tensor in stashed.values()) ** 0.5
        factor = min(1.0, clip_to / stash_norm) if stash_norm > 0 else 1.0
        stashed = {name: tensor * factor for name, tensor in stashed.items()}
    device = next(model.parameters()).device
    sum_dot = 0.0
    sum_sq_ref = 0.0
    sum_sq_new = 0.0
    worst_rel = 0.0
    worst_abs = 0.0
    worst_entry = None
    for name, parameter in model.named_parameters():
        # Compare on the CPU: the launcher's own CUDA context (~350 MB) stays
        # resident on rank 0's device, and double-precision copies of the
        # tied-embedding gradient overflow 12 GB there.
        reference = stashed[name]
        actual = parameter.grad.detach().to("cpu")
        difference = (actual - reference).abs()
        tensor_max = float(reference.abs().max())
        scale_rel = float(difference.max()) / max(tensor_max, 1e-8)
        if scale_rel > worst_rel:
            worst_rel = scale_rel
            worst_entry = name
        worst_abs = max(worst_abs, float(difference.max()))
        sum_dot += float((actual.double() * reference.double()).sum())
        sum_sq_ref += float((reference.double() ** 2).sum())
        sum_sq_new += float((actual.double() ** 2).sum())
        del reference
    cosine = sum_dot / (sum_sq_ref ** 0.5 * sum_sq_new ** 0.5)
    return {
        "grad_cosine_similarity": cosine,
        "grad_scale_relative_error": worst_rel,
        "grad_max_absolute_error": worst_abs,
        "grad_abs_error_over_reference_norm": worst_abs / float(sum_sq_ref ** 0.5),
        "grad_scale_relative_error_parameter": worst_entry,
    }
