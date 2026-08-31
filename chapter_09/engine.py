"""第 09 期 Eager/FlashAttention 独立教学推理引擎。"""

import math
import time
from collections import defaultdict

import torch

from paged_cache import PagedKVCache, paged_decode_forward
from scheduler import ChunkedPrefillScheduler, RequestState, SchedulerConfig


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, percent):
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, (len(ordered) * percent + 99) // 100)
    return ordered[min(rank, len(ordered)) - 1]


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


def _combine_request_caches(cache, states):
    caches = [cache.dense_request_cache(state.spec.request_id) for state in states]
    if all(item is None for item in caches):
        return None
    if any(item is None for item in caches):
        raise RuntimeError("同组 Chunk 的历史 Cache 状态不一致")
    return [
        (
            torch.cat([item[layer_index][0] for item in caches], dim=0),
            torch.cat([item[layer_index][1] for item in caches], dim=0),
        )
        for layer_index in range(cache.config.num_hidden_layers)
    ]


def _build_chunk_batch(plans, pad_token_id, device):
    lengths = [plan.token_count for plan in plans]
    maximum = max(lengths)
    prefix_length = plans[0].start
    if any(plan.start != prefix_length for plan in plans):
        raise ValueError("同一执行组的历史前缀长度必须一致")
    input_ids = torch.full(
        (len(plans), maximum), pad_token_id,
        dtype=torch.long, device=device,
    )
    attention_mask = torch.zeros(
        (len(plans), prefix_length + maximum),
        dtype=torch.bool, device=device,
    )
    position_ids = torch.zeros(
        (len(plans), maximum), dtype=torch.long, device=device
    )
    if prefix_length:
        attention_mask[:, :prefix_length] = True
    for row, plan in enumerate(plans):
        tokens = torch.as_tensor(
            plan.state.spec.token_ids[plan.start:plan.end],
            dtype=torch.long, device=device,
        )
        length = tokens.numel()
        offset = maximum - length
        input_ids[row, offset:] = tokens
        attention_mask[row, prefix_length + offset:] = True
        position_ids[row, offset:] = torch.arange(
            plan.start, plan.end, dtype=torch.long, device=device
        )
    return input_ids, attention_mask, position_ids, lengths


def _timed_prefill(model, plans, cache, pad_token_id, device, capture_logits):
    groups = defaultdict(list)
    for plan in plans:
        groups[plan.start].append(plan)
    final_tokens = {}
    final_logits = {}
    model_ms = 0.0
    cache_ms = 0.0
    peak_memory = 0

    synchronize(device)
    total_start = time.perf_counter()
    for _, group in sorted(groups.items()):
        input_ids, mask, positions, lengths = _build_chunk_batch(
            group, pad_token_id, device
        )
        past = _combine_request_caches(cache, [plan.state for plan in group])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        model_start = time.perf_counter()
        logits, present = model(
            input_ids,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=past,
            use_cache=True,
        )
        synchronize(device)
        model_ms += (time.perf_counter() - model_start) * 1000
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))

        synchronize(device)
        cache_start = time.perf_counter()
        for row, (plan, length) in enumerate(zip(group, lengths)):
            request_cache = [
                (
                    key[row:row + 1, :, -length:, :],
                    value[row:row + 1, :, -length:, :],
                )
                for key, value in present
            ]
            cache.append_prefill(plan.state.spec.request_id, request_cache)
            if plan.end == len(plan.state.spec.token_ids):
                vector = logits[row, -1, :]
                final_tokens[plan.state.spec.request_id] = int(
                    torch.argmax(vector).item()
                )
                if capture_logits:
                    final_logits[plan.state.spec.request_id] = vector.float().cpu()
        synchronize(device)
        cache_ms += (time.perf_counter() - cache_start) * 1000
        del input_ids, mask, positions, logits, present, past

    synchronize(device)
    total_ms = (time.perf_counter() - total_start) * 1000
    return final_tokens, final_logits, total_ms, model_ms, cache_ms, peak_memory


def _timed_decode(model, running, cache, capacity, pad_token_id, device):
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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    model_start = time.perf_counter()
    logits, visited_slots = paged_decode_forward(
        model, input_ids, position_ids, slot_request_ids, cache
    )
    next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
    synchronize(device)
    model_ms = (time.perf_counter() - model_start) * 1000
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    selected = [int(next_tokens[state.slot_index].item()) for state in running]
    del logits, next_tokens
    return selected, prepare_ms + model_ms, model_ms, prepare_ms, visited_slots, peak_memory


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


def _release(cache, completed_states, device):
    if not completed_states:
        return 0.0
    synchronize(device)
    start = time.perf_counter()
    for state in completed_states:
        cache.release(state.spec.request_id)
    synchronize(device)
    return (time.perf_counter() - start) * 1000


def _finalize(
    states, events, config, peak_memory, cache, scheduler_ms_total,
    prefix_cache_enabled,
):
    ordered = list(states)
    first_arrival = min(state.spec.arrival_ms for state in ordered)
    last_completion = max(state.completion_ms for state in ordered)
    makespan_ms = last_completion - first_arrival
    busy_ms = sum(event["total_ms"] for event in events) + sum(
        state.prefix_lookup_ms for state in states
    )
    output_tokens = sum(len(state.generated) for state in ordered)
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
            "prefill_started_ms": state.prefill_started_ms,
            "prefill_completed_ms": state.prefill_completed_ms,
            "first_token_ms": state.first_token_ms,
            "completion_ms": state.completion_ms,
            "queue_ms": state.admitted_ms - state.spec.arrival_ms,
            "prefill_elapsed_ms": state.prefill_completed_ms - state.prefill_started_ms,
            "ttft_ms": state.first_token_ms - state.spec.arrival_ms,
            "service_ttft_ms": state.first_token_ms - state.admitted_ms,
            "end_to_end_ms": state.completion_ms - state.spec.arrival_ms,
            "prompt_tokens": len(state.spec.token_ids),
            "prefix_hit_tokens": state.prefix_hit_tokens,
            "prefix_lookup_ms": state.prefix_lookup_ms,
            "executed_prefill_tokens": len(state.spec.token_ids) - state.prefix_hit_tokens,
            "output_tokens": len(state.generated),
            "token_times_ms": state.token_times_ms,
        }
        for state in ordered
    ]
    prefill_events = [event for event in events if event["phase"] == "prefill"]
    padded_tokens = sum(event["scheduled_tokens"] for event in prefill_events)
    logical_tokens = sum(event["logical_tokens"] for event in prefill_events)
    budget_capacity = len(prefill_events) * config.token_budget
    metrics = {
        "request_count": len(ordered),
        "mode": "enabled" if prefix_cache_enabled else "disabled",
        "token_budget": config.token_budget,
        "max_running_requests": config.max_running_requests,
        "makespan_ms": makespan_ms,
        "busy_ms": busy_ms,
        "model_ms": sum(event["model_ms"] for event in events),
        "scheduler_ms": scheduler_ms_total,
        "scheduler_fraction": scheduler_ms_total / busy_ms if busy_ms else 0.0,
        "cache_management_ms": sum(
            event["cache_management_ms"] for event in events
        ) + sum(state.prefix_lookup_ms for state in states),
        "prefix_lookup_ms": sum(state.prefix_lookup_ms for state in states),
        "output_token_throughput_per_second": (
            output_tokens * 1000 / makespan_ms if makespan_ms else 0.0
        ),
        "queue_ms_p50": percentile([item["queue_ms"] for item in request_metrics], 50),
        "queue_ms_p95": percentile([item["queue_ms"] for item in request_metrics], 95),
        "ttft_ms_p50": percentile([item["ttft_ms"] for item in request_metrics], 50),
        "ttft_ms_p95": percentile([item["ttft_ms"] for item in request_metrics], 95),
        "end_to_end_ms_p95": percentile(
            [item["end_to_end_ms"] for item in request_metrics], 95
        ),
        "itl_ms_p50": percentile(itls, 50),
        "itl_ms_p95": percentile(itls, 95),
        "itl_ms_max": max(itls) if itls else 0.0,
        "prefill_iterations": len(prefill_events),
        "decode_iterations": sum(event["phase"] == "decode" for event in events),
        "max_consecutive_prefill_iterations": _maximum_consecutive_phase(
            events, "prefill"
        ),
        "prefill_logical_tokens": logical_tokens,
        "prefill_padded_tokens": padded_tokens,
        "prefill_padding_fraction": (
            1 - logical_tokens / padded_tokens if padded_tokens else 0.0
        ),
        "prefill_budget_utilization": (
            padded_tokens / budget_capacity if budget_capacity else 0.0
        ),
        "hard_budget_violations": sum(
            event["scheduled_tokens"] > config.token_budget
            for event in prefill_events
        ),
        "prefill_chunk_count": sum(
            len(event["chunk_ranges"]) for event in prefill_events
        ),
        "max_prefill_tokens_per_iteration": max(
            (event["scheduled_tokens"] for event in prefill_events), default=0
        ),
        "max_prefill_iteration_ms": max(
            (event["total_ms"] for event in prefill_events), default=0.0
        ),
        "peak_memory_bytes": peak_memory,
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
        "prefix_cache_enabled": prefix_cache_enabled,
        "prefix_cache_capacity_blocks": cache.prefix_cache_capacity_blocks,
        "prefix_lookup_count": cache.prefix_lookup_count,
        "prefix_hit_tokens": sum(state.prefix_hit_tokens for state in states),
        "executed_prompt_tokens": sum(
            len(state.spec.token_ids) - state.prefix_hit_tokens for state in states
        ),
        "prefix_publish_count": cache.prefix_publish_count,
        "prefix_eviction_count": cache.prefix_eviction_count,
        "cached_blocks_at_end": cache.cached_block_count,
    }
    return request_metrics, metrics


@torch.inference_mode()
def run_engine(
    model,
    request_specs,
    max_running_requests,
    eos_token_id,
    device,
    token_budget,
    block_size=16,
    pad_token_id=0,
    stop_on_eos=True,
    capture_logits=False,
    prefix_cache_enabled=False,
    prefix_cache_capacity_blocks=256,
    model_namespace="Qwen/Qwen3-0.6B",
):
    """运行完整 Chunked Prefill/Decode；Attention 后端由 model 选择。"""
    if not request_specs:
        raise ValueError("request_specs 不能为空")
    for spec in request_specs:
        spec.validate()
    if len({spec.request_id for spec in request_specs}) != len(request_specs):
        raise ValueError("request_id 必须唯一")
    config = SchedulerConfig(max_running_requests, token_budget)
    scheduler = ChunkedPrefillScheduler(config)
    states = [RequestState(spec) for spec in request_specs]
    waiting = sorted(states, key=lambda state: state.spec.arrival_ms)
    prefilling = []
    running = []
    events = []
    captured_logits = {}
    clock_ms = min(state.spec.arrival_ms for state in states)
    dtype = next(model.parameters()).dtype
    maximum_lengths = sorted(
        (len(spec.token_ids) + spec.max_new_tokens - 1 for spec in request_specs),
        reverse=True,
    )[:max_running_requests]
    live_blocks = sum(math.ceil(length / block_size) for length in maximum_lengths)
    cache_blocks = prefix_cache_capacity_blocks if prefix_cache_enabled else 0
    max_blocks = live_blocks + cache_blocks
    cache = PagedKVCache(
        model.config, block_size, max_blocks, device, dtype,
        prefix_cache_enabled=prefix_cache_enabled,
        prefix_cache_capacity_blocks=cache_blocks,
        model_namespace=model_namespace,
    )
    scheduler_ms_total = 0.0
    peak_memory = 0

    while waiting or prefilling or running:
        # 只为本轮确实有空闲运行名额的 FCFS 请求执行缓存匹配。
        # 后续请求不会在前序请求发布缓存之前提前完成冷查找。
        free_slots = max_running_requests - len(prefilling) - len(running)
        ready = [
            state for state in waiting
            if state.spec.arrival_ms <= clock_ms
            and state.spec.request_id not in cache.block_tables
        ][:free_slots]
        for state in ready:
            state.admitted_ms = clock_ms
            lookup_start = time.perf_counter()
            hit_tokens = cache.attach_prefix(
                state.spec.request_id, state.spec.token_ids
            )
            state.prefix_lookup_ms = (time.perf_counter() - lookup_start) * 1000
            clock_ms += state.prefix_lookup_ms
            state.prefix_hit_tokens = hit_tokens
            state.prefill_cursor = hit_tokens
            state.cache_length = hit_tokens
        schedule_start = time.perf_counter()
        output = scheduler.schedule(waiting, prefilling, running, clock_ms)
        scheduler_ms = (time.perf_counter() - schedule_start) * 1000
        scheduler_ms_total += scheduler_ms
        if output is None:
            if prefilling or running:
                raise RuntimeError("调度器没有为活跃请求生成执行计划")
            clock_ms = min(state.spec.arrival_ms for state in waiting)
            continue

        if output.phase == "prefill":
            plans = list(output.prefill_plans)
            existing_ids = {
                state.spec.request_id for state in prefilling + running
            }
            admitted = [plan.state for plan in plans if plan.state.status == "waiting"]
            used_slots = {state.slot_index for state in prefilling + running}
            free_slot_ids = [
                slot for slot in range(max_running_requests) if slot not in used_slots
            ]
            for state in admitted:
                if state.spec.request_id in existing_ids:
                    raise RuntimeError("请求被重复接纳")
                waiting.remove(state)
                state.status = "prefilling"
                if state.admitted_ms is None:
                    state.admitted_ms = clock_ms
                state.prefill_started_ms = clock_ms
                state.slot_index = free_slot_ids.pop(0)
                prefilling.append(state)

            final_tokens, logits, compute_ms, model_ms, cache_ms, iter_peak = (
                _timed_prefill(
                    model, plans, cache, pad_token_id, device, capture_logits
                )
            )
            peak_memory = max(peak_memory, iter_peak)
            clock_ms += scheduler_ms + compute_ms
            finalized = []
            publish_ms = 0.0
            for plan in plans:
                state = plan.state
                if state.prefill_cursor != plan.start:
                    raise RuntimeError("Prefill cursor 与调度计划不一致")
                state.prefill_cursor = plan.end
                state.cache_length = plan.end
                if cache.sequence_lengths[state.spec.request_id] != plan.end:
                    raise RuntimeError("Paged KV Cache 长度与 Prefill cursor 不一致")
                if plan.end == len(state.spec.token_ids):
                    publish_start = time.perf_counter()
                    cache.publish_prompt(
                        state.spec.request_id, state.spec.token_ids
                    )
                    publish_elapsed = (time.perf_counter() - publish_start) * 1000
                    publish_ms += publish_elapsed
                    clock_ms += publish_elapsed
                    state.prefill_completed_ms = clock_ms
                    state.status = "running"
                    finalized.append(state)
                    if state.spec.request_id in logits:
                        captured_logits[state.spec.request_id] = logits[
                            state.spec.request_id
                        ]

            token_ids = [final_tokens[state.spec.request_id] for state in finalized]
            completed_ids = _record_tokens(
                finalized, token_ids, clock_ms, eos_token_id, stop_on_eos
            )
            completed_states = [state for state in finalized if state.status == "finished"]
            release_ms = _release(cache, completed_states, device)
            clock_ms += release_ms
            prefilling = [state for state in prefilling if state.status == "prefilling"]
            running.extend(state for state in finalized if state.status == "running")
            snapshot = cache.snapshot()
            events.append({
                "iteration": len(events),
                "phase": "prefill",
                "mode": "enabled" if prefix_cache_enabled else "disabled",
                "admitted": [state.spec.request_id for state in admitted],
                "scheduled_request_ids": output.request_ids,
                "completed": completed_ids,
                "chunk_ranges": [
                    {
                        "request_id": plan.state.spec.request_id,
                        "start": plan.start,
                        "end": plan.end,
                    }
                    for plan in plans
                ],
                "scheduled_tokens": output.scheduled_tokens,
                "logical_tokens": output.logical_tokens,
                "token_budget": output.token_budget,
                "executed_batch_size": len(plans),
                "active_requests": len(prefilling) + len(running),
                "total_ms": scheduler_ms + compute_ms + publish_ms + release_ms,
                "scheduler_ms": scheduler_ms,
                "model_ms": model_ms,
                "cache_management_ms": cache_ms + publish_ms + release_ms,
                "iteration_peak_memory_bytes": iter_peak,
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
            raise RuntimeError("Decode 必须覆盖全部 running 请求")
        token_ids, compute_ms, model_ms, cache_ms, visited, iter_peak = (
            _timed_decode(
                model, running, cache, max_running_requests, pad_token_id, device
            )
        )
        peak_memory = max(peak_memory, iter_peak)
        active_before = len(running)
        for state in running:
            state.cache_length += 1
        clock_ms += scheduler_ms + compute_ms
        completed_ids = _record_tokens(
            running, token_ids, clock_ms, eos_token_id, stop_on_eos
        )
        completed_states = [state for state in running if state.status == "finished"]
        release_ms = _release(cache, completed_states, device)
        clock_ms += release_ms
        running = [state for state in running if state.status == "running"]
        snapshot = cache.snapshot()
        events.append({
            "iteration": len(events),
            "phase": "decode",
            "mode": "enabled" if prefix_cache_enabled else "disabled",
            "admitted": [],
            "scheduled_request_ids": output.request_ids,
            "completed": completed_ids,
            "chunk_ranges": [],
            "scheduled_tokens": output.scheduled_tokens,
            "logical_tokens": output.logical_tokens,
            "token_budget": output.token_budget,
            "executed_batch_size": max_running_requests,
            "active_requests": active_before,
            "total_ms": scheduler_ms + compute_ms + release_ms,
            "scheduler_ms": scheduler_ms,
            "model_ms": model_ms,
            "cache_management_ms": cache_ms + release_ms,
            "iteration_peak_memory_bytes": iter_peak,
            "live_request_cache_bytes": snapshot["live_cache_bytes"],
            "pool_cache_bytes": snapshot["pool_bytes"],
            "visited_kv_token_slots": visited,
            "cache_snapshot": snapshot,
        })

    request_metrics, metrics = _finalize(
        states, events, config, peak_memory, cache, scheduler_ms_total,
        prefix_cache_enabled,
    )
    return {
        "new_token_ids": [state.generated for state in states],
        "first_token_logits": captured_logits,
        "request_metrics": request_metrics,
        "events": events,
        "metrics": metrics,
        "final_cache_snapshot": cache.snapshot(),
    }
