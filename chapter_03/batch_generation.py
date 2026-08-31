"""固定批处理的 Batch 构造、生成循环和指标统计。"""

import time

import torch

from qwen3_model import cache_size_bytes


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, percent):
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, (len(ordered) * percent + 99) // 100)
    return ordered[min(rank, len(ordered)) - 1]


def left_pad_sequences(token_sequences, pad_token_id, device):
    """把一组非空的一维 Token 序列左 Padding 为矩形 Tensor。"""

    if not token_sequences:
        raise ValueError("token_sequences 不能为空")
    lengths = []
    normalized = []
    for sequence in token_sequences:
        tensor = torch.as_tensor(sequence, dtype=torch.long, device=device)
        if tensor.ndim != 1 or tensor.numel() < 1:
            raise ValueError("每个 Prompt 必须是一维非空 Token 序列")
        normalized.append(tensor)
        lengths.append(tensor.numel())

    max_length = max(lengths)
    batch_size = len(normalized)
    input_ids = torch.full(
        (batch_size, max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.bool, device=device
    )
    for row, sequence in enumerate(normalized):
        length = sequence.numel()
        input_ids[row, max_length - length :] = sequence
        attention_mask[row, max_length - length :] = True

    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.clamp_(min=0)
    return (
        input_ids,
        attention_mask,
        position_ids,
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def _normalize_budgets(max_new_tokens, batch_size, device):
    if isinstance(max_new_tokens, int):
        budgets = [max_new_tokens] * batch_size
    else:
        budgets = list(max_new_tokens)
    if len(budgets) != batch_size or any(value < 1 for value in budgets):
        raise ValueError("每个请求必须提供一个大于 0 的输出预算")
    return torch.tensor(budgets, dtype=torch.long, device=device)


def _timed_step(
    model,
    input_ids,
    attention_mask,
    position_ids,
    past_key_values,
    device,
):
    synchronize(device)
    start_time = time.perf_counter()
    logits, present = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    last_logits = logits[:, -1, :]
    next_tokens = torch.argmax(last_logits, dim=-1)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return next_tokens, last_logits, present, elapsed_ms


@torch.inference_mode()
def generate_fixed_batch(
    model,
    token_sequences,
    max_new_tokens,
    eos_token_id,
    device,
    pad_token_id=0,
    stop_on_eos=True,
    capture_logits=False,
):
    """运行成员固定的 Batch；完成槽位不会被新请求替换。"""

    input_ids, history_mask, position_ids, prompt_lengths = left_pad_sequences(
        token_sequences, pad_token_id, device
    )
    batch_size, padded_prompt_length = input_ids.shape
    budgets = _normalize_budgets(max_new_tokens, batch_size, device)
    generated_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    outputs = [[] for _ in range(batch_size)]
    completion_steps = [None] * batch_size
    step_times_ms = []
    active_counts = []
    logits_trace = [] if capture_logits else None
    past_key_values = None
    memory_after_prefill_bytes = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(int(budgets.max().item())):
        active_before = active.clone()
        next_tokens, last_logits, past_key_values, elapsed_ms = _timed_step(
            model,
            input_ids,
            history_mask,
            position_ids,
            past_key_values,
            device,
        )
        if step == 0 and device.type == "cuda":
            memory_after_prefill_bytes = torch.cuda.memory_allocated(device)
        if capture_logits:
            logits_trace.append(last_logits.float().cpu())

        active_counts.append(int(active_before.sum().item()))
        step_times_ms.append(elapsed_ms)
        for row in range(batch_size):
            if not bool(active_before[row].item()):
                continue
            token_id = int(next_tokens[row].item())
            outputs[row].append(token_id)
            generated_counts[row] += 1
            reached_eos = stop_on_eos and token_id == eos_token_id
            reached_budget = generated_counts[row] >= budgets[row]
            if reached_eos or bool(reached_budget.item()):
                active[row] = False
                completion_steps[row] = step + 1

        del last_logits
        if not bool(active.any().item()):
            break

        # 只有仍活跃请求的最新 Token 会成为下一轮有效 K/V。
        input_ids = torch.where(
            active, next_tokens, torch.full_like(next_tokens, pad_token_id)
        ).unsqueeze(1)
        history_mask = torch.cat((history_mask, active.unsqueeze(1)), dim=1)
        next_positions = prompt_lengths + generated_counts - 1
        position_ids = torch.where(
            active, next_positions, torch.zeros_like(next_positions)
        ).unsqueeze(1)

    peak_memory_bytes = 0
    if device.type == "cuda":
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    total_ms = sum(step_times_ms)
    valid_tokens = sum(len(output) for output in outputs)
    capacity_tokens = batch_size * len(step_times_ms)
    decode_times = step_times_ms[1:]
    cumulative_times = []
    running_time = 0.0
    for value in step_times_ms:
        running_time += value
        cumulative_times.append(running_time)
    completion_times_ms = [
        cumulative_times[step - 1] if step is not None else None
        for step in completion_steps
    ]
    return {
        "new_token_ids": outputs,
        "logits_trace": logits_trace,
        "metrics": {
            "batch_size": batch_size,
            "prompt_lengths": prompt_lengths.cpu().tolist(),
            "padded_prompt_length": padded_prompt_length,
            "prompt_tokens": int(prompt_lengths.sum().item()),
            "padded_prompt_tokens": batch_size * padded_prompt_length,
            "padding_efficiency": float(
                prompt_lengths.sum().item() / (batch_size * padded_prompt_length)
            ),
            "prefill_ms": step_times_ms[0],
            "effective_prompt_token_throughput_per_second": (
                int(prompt_lengths.sum().item()) * 1000 / step_times_ms[0]
            ),
            "padded_prompt_token_throughput_per_second": (
                batch_size * padded_prompt_length * 1000 / step_times_ms[0]
            ),
            "decode_times_ms": decode_times,
            "average_decode_iteration_ms": (
                sum(decode_times) / len(decode_times) if decode_times else 0.0
            ),
            "p50_decode_iteration_ms": percentile(decode_times, 50),
            "p95_decode_iteration_ms": percentile(decode_times, 95),
            "end_to_end_ms": total_ms,
            "request_throughput_per_second": (
                batch_size * 1000 / total_ms if total_ms else 0.0
            ),
            "output_token_throughput_per_second": (
                valid_tokens * 1000 / total_ms if total_ms else 0.0
            ),
            "valid_generated_tokens": valid_tokens,
            "active_counts": active_counts,
            "slot_utilization": (
                valid_tokens / capacity_tokens if capacity_tokens else 0.0
            ),
            "completion_steps": completion_steps,
            "completion_times_ms": completion_times_ms,
            "memory_after_prefill_bytes": memory_after_prefill_bytes,
            "peak_memory_bytes": peak_memory_bytes,
            "cache_bytes": cache_size_bytes(past_key_values),
            "cache_length": past_key_values[0][0].shape[2],
        },
    }


@torch.inference_mode()
def generate_serial(
    model,
    token_sequences,
    max_new_tokens,
    eos_token_id,
    device,
    pad_token_id=0,
    stop_on_eos=True,
):
    """Baseline：相同请求逐个使用 KV Cache 完整生成。"""

    if isinstance(max_new_tokens, int):
        budgets = [max_new_tokens] * len(token_sequences)
    else:
        budgets = list(max_new_tokens)
    if len(budgets) != len(token_sequences):
        raise ValueError("输出预算数量必须与请求数量一致")

    runs = []
    wall_start = time.perf_counter()
    for sequence, budget in zip(token_sequences, budgets):
        runs.append(
            generate_fixed_batch(
                model,
                [sequence],
                budget,
                eos_token_id,
                device,
                pad_token_id=pad_token_id,
                stop_on_eos=stop_on_eos,
            )
        )
    synchronize(device)
    wall_ms = (time.perf_counter() - wall_start) * 1000
    model_ms = sum(run["metrics"]["end_to_end_ms"] for run in runs)
    output_ids = [run["new_token_ids"][0] for run in runs]
    valid_tokens = sum(len(output) for output in output_ids)
    return {
        "new_token_ids": output_ids,
        "runs": runs,
        "metrics": {
            "batch_size": len(token_sequences),
            "end_to_end_ms": model_ms,
            "wall_ms": wall_ms,
            "request_throughput_per_second": (
                len(token_sequences) * 1000 / model_ms if model_ms else 0.0
            ),
            "output_token_throughput_per_second": (
                valid_tokens * 1000 / model_ms if model_ms else 0.0
            ),
            "peak_memory_bytes": max(
                run["metrics"]["peak_memory_bytes"] for run in runs
            ),
        },
    }
