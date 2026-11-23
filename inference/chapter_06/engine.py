"""第 06 期带显式迭代级调度器的独立 Paged KV Cache 引擎。"""

import math
import time

import torch

from paged_cache import PagedKVCache, paged_decode_forward
from scheduler import IterationScheduler, RequestState, SchedulerConfig


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
    if not token_sequences:
        raise ValueError("token_sequences 不能为空")
    normalized = []
    lengths = []
    for sequence in token_sequences:
        tensor = torch.as_tensor(sequence, dtype=torch.long, device=device)
        if tensor.ndim != 1 or tensor.numel() < 1:
            raise ValueError("每个 Prompt 必须是一维非空 Token 序列")
        normalized.append(tensor)
        lengths.append(tensor.numel())
    maximum = max(lengths)
    input_ids = torch.full(
        (len(normalized), maximum), pad_token_id,
        dtype=torch.long, device=device,
    )
    attention_mask = torch.zeros(
        (len(normalized), maximum), dtype=torch.bool, device=device
    )
    for row, sequence in enumerate(normalized):
        length = sequence.numel()
        input_ids[row, maximum - length:] = sequence
        attention_mask[row, maximum - length:] = True
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.clamp_(min=0)
    return input_ids, attention_mask, position_ids, lengths


def _split_prefill_cache(past_key_values, prompt_lengths):
    request_caches = []
    for row, length in enumerate(prompt_lengths):
        request_caches.append([
            (
                key[row:row + 1, :, -length:, :].clone(),
                value[row:row + 1, :, -length:, :].clone(),
            )
            for key, value in past_key_values
        ])
    return request_caches


def _timed_prefill(model, states, pad_token_id, device):
    input_ids, mask, positions, lengths = left_pad_sequences(
        [state.spec.token_ids for state in states], pad_token_id, device
    )
    synchronize(device)
    total_start = time.perf_counter()
    logits, past = model(
        input_ids,
        attention_mask=mask,
        position_ids=positions,
        use_cache=True,
    )
    next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
    synchronize(device)
    model_ms = (time.perf_counter() - total_start) * 1000

    cache_start = time.perf_counter()
    caches = _split_prefill_cache(past, lengths)
    synchronize(device)
    cache_ms = (time.perf_counter() - cache_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    return next_tokens.cpu().tolist(), caches, total_ms, model_ms, cache_ms


def _store_prefill(cache, admitted, request_caches, device):
    synchronize(device)
    start = time.perf_counter()
    for state, request_cache in zip(admitted, request_caches):
        cache.store_prefill(state.spec.request_id, request_cache)
    synchronize(device)
    return (time.perf_counter() - start) * 1000


def _release(cache, completed_states, device):
    if not completed_states:
        return 0.0
    synchronize(device)
    start = time.perf_counter()
    for state in completed_states:
        cache.release(state.spec.request_id)
    synchronize(device)
    return (time.perf_counter() - start) * 1000


def _timed_decode(
    model, running, cache, capacity, pad_token_id, device
):
    synchronize(device)
    prepare_start = time.perf_counter()
    input_ids = torch.full(
        (capacity, 1), pad_token_id, dtype=torch.long, device=device
    )
    position_ids = torch.zeros((capacity, 1), dtype=torch.long, device=device)
    slot_request_ids = [None] * capacity
    for state in running:
        slot = state.slot_index
        input_ids[slot, 0] = state.generated[-1]
        position_ids[slot, 0] = state.cache_length
        slot_request_ids[slot] = state.spec.request_id
    synchronize(device)
    prepare_ms = (time.perf_counter() - prepare_start) * 1000

    model_start = time.perf_counter()
    logits, visited_slots = paged_decode_forward(
        model, input_ids, position_ids, slot_request_ids, cache
    )
    next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
    synchronize(device)
    model_ms = (time.perf_counter() - model_start) * 1000
    selected = [int(next_tokens[state.slot_index].item()) for state in running]
    del logits, next_tokens
    return selected, prepare_ms + model_ms, model_ms, prepare_ms, visited_slots


def _record_tokens(states, token_ids, token_time_ms, eos_token_id, stop_on_eos):
    completed = []
    for state, token_id in zip(states, token_ids):
        state.generated.append(int(token_id))
        state.token_times_ms.append(token_time_ms)
        if state.first_token_ms is None:
            state.first_token_ms = token_time_ms
        reached_eos = stop_on_eos and token_id == eos_token_id
        reached_budget = len(state.generated) >= state.spec.max_new_tokens
        if reached_eos or reached_budget:
            state.status = "finished"
            state.completion_ms = token_time_ms
            completed.append(state.spec.request_id)
    return completed


def _maximum_consecutive_phase(events, phase):
    maximum = 0
    current = 0
    for event in events:
        if event["phase"] == phase:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _finalize(
    states, events, config, peak_memory_bytes, cache, scheduler_ms_total
):
    ordered = list(states)
    first_arrival = min(state.spec.arrival_ms for state in ordered)
    last_completion = max(state.completion_ms for state in ordered)
    makespan_ms = last_completion - first_arrival
    busy_ms = sum(event["total_ms"] for event in events)
    model_ms = sum(event["model_ms"] for event in events)
    cache_ms = sum(event["cache_management_ms"] for event in events)
    output_tokens = sum(len(state.generated) for state in ordered)
    executed_rows = sum(event["executed_batch_size"] for event in events)
    itls = [
        right - left
        for state in ordered
        for left, right in zip(state.token_times_ms, state.token_times_ms[1:])
    ]
    request_metrics = [
        {
            "request_id": state.spec.request_id,
            "arrival_ms": state.spec.arrival_ms,
            "admitted_ms": state.admitted_ms,
            "first_token_ms": state.first_token_ms,
            "completion_ms": state.completion_ms,
            "queue_ms": state.admitted_ms - state.spec.arrival_ms,
            "ttft_ms": state.first_token_ms - state.spec.arrival_ms,
            "end_to_end_ms": state.completion_ms - state.spec.arrival_ms,
            "output_tokens": len(state.generated),
            "token_times_ms": state.token_times_ms,
        }
        for state in ordered
    ]
    prefill_events = [event for event in events if event["phase"] == "prefill"]
    padded_tokens = sum(event["scheduled_tokens"] for event in prefill_events)
    logical_tokens = sum(event["logical_tokens"] for event in prefill_events)
    bounded_prefills = [
        event for event in prefill_events if not event["oversize_singleton"]
    ]
    budget_capacity = (
        len(bounded_prefills) * config.token_budget
        if config.token_budget is not None else 0
    )
    metrics = {
        "request_count": len(ordered),
        "policy": config.policy,
        "token_budget": config.token_budget,
        "max_running_requests": config.max_running_requests,
        "makespan_ms": makespan_ms,
        "busy_ms": busy_ms,
        "model_ms": model_ms,
        "scheduler_ms": scheduler_ms_total,
        "scheduler_fraction": scheduler_ms_total / busy_ms if busy_ms else 0.0,
        "cache_management_ms": cache_ms,
        "request_throughput_per_second": (
            len(ordered) * 1000 / makespan_ms if makespan_ms else 0.0
        ),
        "output_token_throughput_per_second": (
            output_tokens * 1000 / makespan_ms if makespan_ms else 0.0
        ),
        "execution_slot_utilization": (
            output_tokens / executed_rows if executed_rows else 0.0
        ),
        "queue_ms_p50": percentile(
            [item["queue_ms"] for item in request_metrics], 50
        ),
        "queue_ms_p95": percentile(
            [item["queue_ms"] for item in request_metrics], 95
        ),
        "ttft_ms_p50": percentile(
            [item["ttft_ms"] for item in request_metrics], 50
        ),
        "ttft_ms_p95": percentile(
            [item["ttft_ms"] for item in request_metrics], 95
        ),
        "end_to_end_ms_p50": percentile(
            [item["end_to_end_ms"] for item in request_metrics], 50
        ),
        "end_to_end_ms_p95": percentile(
            [item["end_to_end_ms"] for item in request_metrics], 95
        ),
        "itl_ms_p50": percentile(itls, 50),
        "itl_ms_p95": percentile(itls, 95),
        "itl_ms_max": max(itls) if itls else 0.0,
        "prefill_iterations": len(prefill_events),
        "decode_iterations": sum(
            event["phase"] == "decode" for event in events
        ),
        "max_consecutive_prefill_iterations": _maximum_consecutive_phase(
            events, "prefill"
        ),
        "prefill_logical_tokens": logical_tokens,
        "prefill_padded_tokens": padded_tokens,
        "prefill_padding_fraction": (
            1 - logical_tokens / padded_tokens if padded_tokens else 0.0
        ),
        "prefill_budget_utilization": (
            sum(event["scheduled_tokens"] for event in bounded_prefills)
            / budget_capacity if budget_capacity else 0.0
        ),
        "oversize_prefill_iterations": sum(
            event["oversize_singleton"] for event in prefill_events
        ),
        "peak_memory_bytes": peak_memory_bytes,
        "peak_live_request_cache_bytes": max(
            (event["live_request_cache_bytes"] for event in events), default=0
        ),
        "peak_pool_cache_bytes": cache.pool_bytes,
        "visited_kv_token_slots": sum(
            event["visited_kv_token_slots"] for event in events
        ),
        "block_size": cache.block_size,
        "bytes_per_block": cache.bytes_per_block,
        "peak_used_blocks": cache.peak_used_blocks,
        "pool_blocks": cache.max_blocks,
        "block_allocation_count": cache.allocation_count,
        "block_reuse_count": cache.reuse_count,
        "block_release_count": cache.release_count,
    }
    return {
        "new_token_ids": [state.generated for state in ordered],
        "request_metrics": request_metrics,
        "events": events,
        "metrics": metrics,
    }


@torch.inference_mode()
def run_scheduler(
    model,
    request_specs,
    max_running_requests,
    eos_token_id,
    device,
    policy,
    token_budget=None,
    block_size=16,
    pad_token_id=0,
    stop_on_eos=True,
):
    """运行 baseline 或 budgeted 迭代级调度策略。"""

    if not request_specs:
        raise ValueError("request_specs 不能为空")
    for spec in request_specs:
        spec.validate()
    config = SchedulerConfig(policy, max_running_requests, token_budget)
    scheduler = IterationScheduler(config)
    states = [RequestState(spec) for spec in request_specs]
    waiting = sorted(states, key=lambda state: state.spec.arrival_ms)
    running = []
    events = []
    clock_ms = min(state.spec.arrival_ms for state in states)
    dtype = next(model.parameters()).dtype
    maximum_lengths = sorted(
        (
            len(spec.token_ids) + spec.max_new_tokens - 1
            for spec in request_specs
        ),
        reverse=True,
    )[:max_running_requests]
    max_blocks = sum(
        math.ceil(length / block_size) for length in maximum_lengths
    )
    cache = PagedKVCache(model.config, block_size, max_blocks, device, dtype)
    scheduler_ms_total = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while waiting or running:
        schedule_start = time.perf_counter()
        output = scheduler.schedule(waiting, running, clock_ms)
        scheduler_ms = (time.perf_counter() - schedule_start) * 1000
        scheduler_ms_total += scheduler_ms
        if output is None:
            if running:
                raise RuntimeError("调度器没有为 running 请求生成执行计划")
            clock_ms = min(state.spec.arrival_ms for state in waiting)
            continue

        if output.phase == "prefill":
            admitted = list(output.selected)
            used_slots = {state.slot_index for state in running}
            free_slot_ids = [
                slot for slot in range(max_running_requests)
                if slot not in used_slots
            ]
            for state in admitted:
                waiting.remove(state)
                state.status = "running"
                state.admitted_ms = clock_ms
                state.slot_index = free_slot_ids.pop(0)
                state.cache_length = len(state.spec.token_ids)
            token_ids, caches, compute_ms, model_ms, cache_ms = _timed_prefill(
                model, admitted, pad_token_id, device
            )
            store_ms = _store_prefill(cache, admitted, caches, device)
            del caches
            running.extend(admitted)
            compute_ms += store_ms
            cache_ms += store_ms
            clock_ms += scheduler_ms + compute_ms
            active_count = len(running)
            completed_ids = _record_tokens(
                admitted, token_ids, clock_ms, eos_token_id, stop_on_eos
            )
            completed_states = [
                state for state in admitted if state.status == "finished"
            ]
            release_ms = _release(cache, completed_states, device)
            clock_ms += release_ms
            running = [state for state in running if state.status == "running"]
            snapshot = cache.snapshot()
            events.append({
                "iteration": len(events),
                "phase": "prefill",
                "policy": policy,
                "admitted": output.request_ids,
                "scheduled_request_ids": output.request_ids,
                "completed": completed_ids,
                "scheduled_tokens": output.scheduled_tokens,
                "logical_tokens": output.logical_tokens,
                "token_budget": output.token_budget,
                "oversize_singleton": output.oversize_singleton,
                "executed_batch_size": len(admitted),
                "active_requests": active_count,
                "total_ms": scheduler_ms + compute_ms + release_ms,
                "scheduler_ms": scheduler_ms,
                "model_ms": model_ms,
                "cache_management_ms": cache_ms + release_ms,
                "live_request_cache_bytes": snapshot["live_cache_bytes"],
                "pool_cache_bytes": snapshot["pool_bytes"],
                "visited_kv_token_slots": 0,
                "cache_snapshot": snapshot,
            })
            continue

        if output.phase != "decode":
            raise RuntimeError("未知调度阶段: " + output.phase)
        scheduled = list(output.selected)
        if scheduled != running:
            raise RuntimeError("本期 Decode 必须覆盖全部 running 请求")
        active_before = len(running)
        token_ids, compute_ms, model_ms, cache_ms, visited_slots = _timed_decode(
            model, running, cache, max_running_requests, pad_token_id, device
        )
        for state in running:
            state.cache_length += 1
        clock_ms += scheduler_ms + compute_ms
        completed_ids = _record_tokens(
            running, token_ids, clock_ms, eos_token_id, stop_on_eos
        )
        completed_states = [
            state for state in running if state.status == "finished"
        ]
        release_ms = _release(cache, completed_states, device)
        clock_ms += release_ms
        running = [state for state in running if state.status == "running"]
        snapshot = cache.snapshot()
        events.append({
            "iteration": len(events),
            "phase": "decode",
            "policy": policy,
            "admitted": [],
            "scheduled_request_ids": output.request_ids,
            "completed": completed_ids,
            "scheduled_tokens": output.scheduled_tokens,
            "logical_tokens": output.logical_tokens,
            "token_budget": output.token_budget,
            "oversize_singleton": False,
            "executed_batch_size": max_running_requests,
            "active_requests": active_before,
            "total_ms": scheduler_ms + compute_ms + release_ms,
            "scheduler_ms": scheduler_ms,
            "model_ms": model_ms,
            "cache_management_ms": cache_ms + release_ms,
            "live_request_cache_bytes": snapshot["live_cache_bytes"],
            "pool_cache_bytes": snapshot["pool_bytes"],
            "visited_kv_token_slots": visited_slots,
            "cache_snapshot": snapshot,
        })

    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    return _finalize(
        states, events, config, peak_memory, cache, scheduler_ms_total
    )
