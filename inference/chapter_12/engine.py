"""第 12 期 KV Cache Offloading 独立教学引擎（抢占式换出）。

底座为第 07 期形态：Paged KV Cache + Continuous Batching + Chunked Prefill +
FCFS 调度。本期在其上加入 Block 级资源管理：

- 人为受限的 GPU Block Pool 与 Block 级准入控制；
- Decode 跨块增长无法满足时触发抢占，受害者固定为最后接纳的 running 请求；
- swap 抢占：整请求 KV 同步换出到 Pinned CPU Pool，恢复时换回断点续算；
- recompute 抢占：直接丢弃 KV，恢复时对 prompt + 已生成部分重新 Prefill；
- swapped 队列按接纳顺序 FCFS，恢复优先于新请求准入。

抢占、恢复和准入都发生在迭代边界，不拆散正在执行的 batch。换出/换入使用
默认 Stream 上的同步 blocking copy；异步传输与计算重叠属于第 13 期。
"""

import time
from collections import defaultdict

import torch

from paged_cache import PagedKVCache, CPUPinnedPool, paged_decode_forward
from scheduler import ChunkedPrefillScheduler, RequestState, SchedulerConfig

PREEMPT_MODES = ("swap", "recompute")
ADMISSION_MODES = ("incremental", "conservative")


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
        source = plan.state.prefill_source_tokens
        tokens = torch.as_tensor(
            source[plan.start:plan.end],
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
            if plan.end == len(plan.state.prefill_source_tokens):
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


def _timed_decode(
    model, running, cache, capacity, pad_token_id, device, capture_logits,
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
    selected_logits = {}
    if capture_logits:
        selected_logits = {
            state.spec.request_id: logits[state.slot_index, -1, :].float().cpu()
            for state in running
        }
    del logits, next_tokens
    return (
        selected, selected_logits, prepare_ms + model_ms, model_ms,
        prepare_ms, visited_slots, peak_memory,
    )


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


def _timed_swap_out(gpu_cache, cpu_pool, request_id, device):
    synchronize(device)
    start = time.perf_counter()
    stats = cpu_pool.swap_out(gpu_cache, request_id)
    synchronize(device)
    stats["wall_ms"] = (time.perf_counter() - start) * 1000
    stats["gb_per_second"] = (
        stats["bytes"] / (stats["wall_ms"] / 1000) / 1e9
        if stats["wall_ms"] > 0 else 0.0
    )
    return stats


def _timed_swap_in(gpu_cache, cpu_pool, request_id, device):
    synchronize(device)
    start = time.perf_counter()
    stats = cpu_pool.swap_in(gpu_cache, request_id)
    synchronize(device)
    stats["wall_ms"] = (time.perf_counter() - start) * 1000
    stats["gb_per_second"] = (
        stats["bytes"] / (stats["wall_ms"] / 1000) / 1e9
        if stats["wall_ms"] > 0 else 0.0
    )
    return stats


def _growth_need(running_states, block_size):
    """本轮 Decode 中需要申请新 Block 的 running 请求数。"""
    return sum(
        1 for state in running_states if state.cache_length % block_size == 0
    )


def _free_slot_ids(prefilling, running, max_running_requests):
    used = {state.slot_index for state in prefilling + running}
    return [
        slot for slot in range(max_running_requests) if slot not in used
    ]


def _finalize(
    states, events, resource_events, config, peak_memory, cache, cpu_pool,
    scheduler_ms_total, preempt_mode, admission_mode, logical_concurrency_peak,
):
    ordered = list(states)
    first_arrival = min(state.spec.arrival_ms for state in ordered)
    last_completion = max(state.completion_ms for state in ordered)
    makespan_ms = last_completion - first_arrival
    busy_ms = sum(event["total_ms"] for event in events)
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
            "prefill_elapsed_ms": (
                state.prefill_completed_ms - state.prefill_started_ms
            ),
            "ttft_ms": state.first_token_ms - state.spec.arrival_ms,
            "end_to_end_ms": state.completion_ms - state.spec.arrival_ms,
            "prompt_tokens": len(state.spec.token_ids),
            "output_tokens": len(state.generated),
            "token_times_ms": state.token_times_ms,
            "preempt_count": state.preempt_count,
            "resume_count": state.resume_count,
            "paused_ms_total": state.paused_ms_total,
            "swap_out_count": state.swap_out_count,
            "swap_in_count": state.swap_in_count,
            "recompute_prefill_ms_total": state.recompute_prefill_ms_total,
            "recompute_elapsed_ms": (
                state.recompute_completed_ms - state.recompute_started_ms
                if state.recompute_started_ms is not None
                and state.recompute_completed_ms is not None
                else 0.0
            ),
        }
        for state in ordered
    ]
    prefill_events = [event for event in events if event["phase"] == "prefill"]
    padded_tokens = sum(event["scheduled_tokens"] for event in prefill_events)
    logical_tokens = sum(event["logical_tokens"] for event in prefill_events)
    budget_capacity = len(prefill_events) * config.token_budget

    swap_out_events = [
        event for event in resource_events if event["type"] == "swap_out"
    ]
    swap_in_events = [
        event for event in resource_events if event["type"] == "swap_in"
    ]
    drop_events = [
        event for event in resource_events if event["type"] == "preempt_drop"
    ]
    resume_recompute_events = [
        event for event in resource_events if event["type"] == "resume_recompute"
    ]
    pause_ms_values = [
        event["pause_ms"] for event in resource_events if "pause_ms" in event
    ]
    swap_out_bytes = sum(event["bytes"] for event in swap_out_events)
    swap_in_bytes = sum(event["bytes"] for event in swap_in_events)
    swap_out_logical_bytes = sum(
        event.get("logical_bytes", event["bytes"]) for event in swap_out_events
    )
    swap_in_logical_bytes = sum(
        event.get("logical_bytes", event["bytes"]) for event in swap_in_events
    )
    swap_out_tail_fragment_bytes = sum(
        event.get("tail_fragment_bytes", 0) for event in swap_out_events
    )
    swap_in_tail_fragment_bytes = sum(
        event.get("tail_fragment_bytes", 0) for event in swap_in_events
    )
    swap_out_ms = sum(event["wall_ms"] for event in swap_out_events)
    swap_in_ms = sum(event["wall_ms"] for event in swap_in_events)
    dropped_bytes = sum(event["bytes"] for event in drop_events)

    metrics = {
        "request_count": len(ordered),
        "preempt_mode": preempt_mode,
        "admission_mode": admission_mode,
        "token_budget": config.token_budget,
        "max_running_requests": config.max_running_requests,
        "makespan_ms": makespan_ms,
        "busy_ms": busy_ms,
        "model_ms": sum(event["model_ms"] for event in events),
        "scheduler_ms": scheduler_ms_total,
        "scheduler_fraction": scheduler_ms_total / busy_ms if busy_ms else 0.0,
        "cache_management_ms": sum(
            event["cache_management_ms"] for event in events
        ),
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
        "prefill_logical_tokens": logical_tokens,
        "prefill_padded_tokens": padded_tokens,
        "prefill_budget_utilization": (
            padded_tokens / budget_capacity if budget_capacity else 0.0
        ),
        "max_prefill_tokens_per_iteration": max(
            (event["scheduled_tokens"] for event in prefill_events), default=0
        ),
        "peak_memory_bytes": peak_memory,
        "peak_live_request_cache_bytes": max(
            (event["live_request_cache_bytes"] for event in events), default=0
        ),
        "pool_cache_bytes": cache.pool_bytes,
        "visited_kv_token_slots": sum(
            event["visited_kv_token_slots"] for event in events
        ),
        "block_size": cache.block_size,
        "bytes_per_block": cache.bytes_per_block,
        "pool_blocks": cache.max_blocks,
        "peak_used_blocks": cache.peak_used_blocks,
        "block_allocation_count": cache.allocation_count,
        "block_reuse_count": cache.reuse_count,
        "block_release_count": cache.release_count,
        "logical_concurrency_peak": logical_concurrency_peak,
        "preemption_count": len(swap_out_events) + len(drop_events),
        "resume_count": len(swap_in_events) + len(resume_recompute_events),
        "swap_out_events": len(swap_out_events),
        "swap_in_events": len(swap_in_events),
        "preempt_drop_events": len(drop_events),
        "resume_recompute_events": len(resume_recompute_events),
        "swap_out_bytes_total": swap_out_bytes,
        "swap_in_bytes_total": swap_in_bytes,
        "swap_out_logical_bytes_total": swap_out_logical_bytes,
        "swap_in_logical_bytes_total": swap_in_logical_bytes,
        "swap_out_tail_fragment_bytes_total": swap_out_tail_fragment_bytes,
        "swap_in_tail_fragment_bytes_total": swap_in_tail_fragment_bytes,
        "dropped_kv_bytes_total": dropped_bytes,
        "swap_out_wall_ms_total": swap_out_ms,
        "swap_in_wall_ms_total": swap_in_ms,
        "swap_out_gb_per_second": (
            swap_out_bytes / (swap_out_ms / 1000) / 1e9 if swap_out_ms else 0.0
        ),
        "swap_in_gb_per_second": (
            swap_in_bytes / (swap_in_ms / 1000) / 1e9 if swap_in_ms else 0.0
        ),
        "recompute_redo_tokens_total": sum(
            event["redo_tokens"] for event in resume_recompute_events
        ),
        "recompute_prefill_wall_ms_total": sum(
            event["total_ms"] for event in prefill_events
            if event.get("recompute_request_ids")
        ),
        "pause_ms_p50": percentile(pause_ms_values, 50),
        "pause_ms_p95": percentile(pause_ms_values, 95),
        "pause_ms_max": max(pause_ms_values) if pause_ms_values else 0.0,
        "cpu_pool": cpu_pool.snapshot(),
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
    preempt_mode="swap",
    admission_mode="incremental",
    pool_blocks=None,
    cpu_pool_blocks=None,
    block_size=16,
    pad_token_id=0,
    stop_on_eos=True,
    capture_logits=False,
):
    """运行带抢占式换出的 Chunked Prefill 引擎。

    pool_blocks=None 时自动按"全部请求同时驻留"分配 GPU Pool（无容量压力）；
    显式传入较小值即人为制造容量压力，这是本期主实验的受控手段。
    """
    if not request_specs:
        raise ValueError("request_specs 不能为空")
    if preempt_mode not in PREEMPT_MODES:
        raise ValueError("preempt_mode 必须是 %s" % (PREEMPT_MODES,))
    if admission_mode not in ADMISSION_MODES:
        raise ValueError("admission_mode 必须是 %s" % (ADMISSION_MODES,))
    for spec in request_specs:
        spec.validate()
    config = SchedulerConfig(max_running_requests, token_budget)
    scheduler = ChunkedPrefillScheduler(config)

    dtype = next(model.parameters()).dtype
    worst_case_blocks = {
        spec.request_id: (
            (len(spec.token_ids) + spec.max_new_tokens + block_size - 1)
            // block_size
        )
        for spec in request_specs
    }
    largest_blocks = max(worst_case_blocks.values())
    total_blocks = sum(worst_case_blocks.values())
    if pool_blocks is None:
        pool_blocks = total_blocks
    if pool_blocks < largest_blocks:
        raise ValueError(
            "GPU Block Pool 过小：单请求最坏需要 %d 块，当前只有 %d 块。"
            "单请求上下文上限仍由 GPU 池决定。" % (largest_blocks, pool_blocks)
        )
    if cpu_pool_blocks is None:
        # recompute 抢占不向 CPU 搬运数据，只需最小的占位池。
        cpu_pool_blocks = total_blocks if preempt_mode == "swap" else 1
    if preempt_mode == "swap" and cpu_pool_blocks < total_blocks:
        raise ValueError(
            "CPU Pinned Pool 配置不足：为保证 swap 路径不在运行期回退或死锁，"
            "本负载最坏需要 %d 块，当前配置 %d 块"
            % (total_blocks, cpu_pool_blocks)
        )

    cache = PagedKVCache(model.config, block_size, pool_blocks, device, dtype)
    pin_memory = device.type == "cuda"
    cpu_pool = CPUPinnedPool(
        model.config, block_size, cpu_pool_blocks, dtype, pin_memory
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    states = [RequestState(spec) for spec in request_specs]
    waiting = sorted(states, key=lambda state: state.spec.arrival_ms)
    prefilling = []
    running = []
    swapped = []
    events = []
    resource_events = []
    captured_logits = defaultdict(list)
    clock_ms = min(state.spec.arrival_ms for state in states)
    scheduler_ms_total = 0.0
    peak_memory = 0
    admission_counter = 0
    logical_concurrency_peak = 0

    while waiting or prefilling or running or swapped:
        logical_concurrency_peak = max(
            logical_concurrency_peak, len(prefilling) + len(running) + len(swapped)
        )
        # 1) 抢占：保守准入下 Decode 增长被预留覆盖，不应触发。
        if admission_mode == "incremental":
            need = _growth_need(running, cache.block_size)
            while cache.free_block_count < need:
                if not running:
                    break
                victim = max(running, key=lambda state: state.admission_seq)
                request_id = victim.spec.request_id
                victim_blocks = cache.blocks_needed(request_id)
                coresident_ids = [
                    state.spec.request_id for state in running if state is not victim
                ]
                if preempt_mode == "swap":
                    stats = _timed_swap_out(cache, cpu_pool, request_id, device)
                    stats["type"] = "swap_out"
                    victim.swap_out_count += 1
                else:
                    synchronize(device)
                    drop_start = time.perf_counter()
                    dropped_bytes = victim_blocks * cache.bytes_per_block
                    cache.release(request_id)
                    synchronize(device)
                    stats = {
                        "type": "preempt_drop",
                        "request_id": request_id,
                        "blocks": victim_blocks,
                        "logical_tokens": victim.cache_length,
                        "bytes": dropped_bytes,
                        "wall_ms": (time.perf_counter() - drop_start) * 1000,
                    }
                victim.status = "swapped"
                victim.slot_index = None
                victim.preempt_count += 1
                running.remove(victim)
                swapped.append(victim)
                clock_ms += stats["wall_ms"]
                stats["clock_ms"] = clock_ms
                stats["reason"] = "decode_growth"
                stats["coresident_request_ids"] = coresident_ids
                victim.preempted_clock_ms = clock_ms
                resource_events.append(stats)
                need = _growth_need(running, cache.block_size)
            remaining_need = _growth_need(running, cache.block_size)
            if cache.free_block_count < remaining_need:
                raise RuntimeError(
                    "抢占全部候选后仍无法满足 Decode 增长：GPU Block Pool 过小"
                )

        # 2) 恢复：swapped 按 FCFS 优先于 waiting 准入。
        while swapped:
            used_slots = {s.slot_index for s in prefilling + running}
            if len(used_slots) >= max_running_requests:
                break
            head = min(swapped, key=lambda state: state.admission_seq)
            request_id = head.spec.request_id
            if preempt_mode == "swap":
                resume_need = cpu_pool.blocks_needed(request_id)
            else:
                # recompute 恢复要重算 prompt + 已生成部分；此时 resuming
                # 仍为 False，必须显式取长度，不能依赖 prefill_source_tokens。
                resume_need = cache.blocks_for_tokens(
                    len(head.spec.token_ids) + len(head.generated)
                )
            growth_reserve = len(running) + 1
            if cache.free_block_count < resume_need + growth_reserve:
                break
            slot = _free_slot_ids(prefilling, running, max_running_requests)[0]
            if preempt_mode == "swap":
                head.status = "restoring"
                coresident_ids = [state.spec.request_id for state in running]
                stats = _timed_swap_in(cache, cpu_pool, request_id, device)
                stats["type"] = "swap_in"
                stats["coresident_request_ids"] = coresident_ids
                head.status = "running"
                head.slot_index = slot
                head.swap_in_count += 1
                running.append(head)
            else:
                head.resuming = True
                head.recompute_started_ms = clock_ms
                head.prefill_cursor = 0
                head.cache_length = 0
                source_length = len(head.prefill_source_tokens)
                cache.begin_request(request_id)
                cache.reserve_blocks(request_id, source_length)
                head.status = "recomputing"
                head.slot_index = slot
                prefilling.append(head)
                stats = {
                    "type": "resume_recompute",
                    "request_id": request_id,
                    "blocks": cache.blocks_for_tokens(source_length),
                    "logical_tokens": source_length,
                    "bytes": cache.blocks_for_tokens(source_length)
                    * cache.bytes_per_block,
                    "redo_tokens": source_length,
                    "wall_ms": 0.0,
                    "coresident_request_ids": [
                        state.spec.request_id for state in running
                    ],
                }
            head.resume_count += 1
            pause_ms = clock_ms - (head.preempted_clock_ms or clock_ms)
            head.paused_ms_total += pause_ms
            stats["pause_ms"] = pause_ms
            swapped.remove(head)
            clock_ms += stats["wall_ms"]
            stats["clock_ms"] = clock_ms
            resource_events.append(stats)

        # 3) 准入：严格 FCFS，队头放不下时后续请求一并等待。
        for state in list(waiting):
            if state.spec.arrival_ms > clock_ms:
                break
            used_slots = {s.slot_index for s in prefilling + running}
            if len(used_slots) >= max_running_requests:
                break
            request_id = state.spec.request_id
            if admission_mode == "conservative":
                reserved_tokens = (
                    len(state.spec.token_ids) + state.spec.max_new_tokens
                )
            else:
                reserved_tokens = len(state.spec.token_ids)
            need_blocks = cache.blocks_for_tokens(reserved_tokens)
            growth_reserve = len(running) + 1
            if cache.free_block_count < need_blocks + growth_reserve:
                break
            waiting.remove(state)
            state.status = "prefilling"
            state.admitted_ms = clock_ms
            state.prefill_started_ms = clock_ms
            state.admission_seq = admission_counter
            admission_counter += 1
            state.slot_index = _free_slot_ids(
                prefilling, running, max_running_requests
            )[0]
            cache.begin_request(request_id)
            cache.reserve_blocks(request_id, reserved_tokens)
            prefilling.append(state)
            resource_events.append({
                "type": "admit",
                "request_id": request_id,
                "clock_ms": clock_ms,
                "blocks": need_blocks,
                "reserved_tokens": reserved_tokens,
                "wall_ms": 0.0,
            })

        # 4) 没有驻留请求时推进逻辑时钟或报告配置错误。
        if not prefilling and not running:
            if waiting and all(
                state.spec.arrival_ms > clock_ms for state in waiting
            ):
                clock_ms = min(state.spec.arrival_ms for state in waiting)
                continue
            raise RuntimeError(
                "没有可推进的驻留请求：waiting=%d prefilling=%d running=%d "
                "swapped=%d free_blocks=%d"
                % (
                    len(waiting), len(prefilling), len(running), len(swapped),
                    cache.free_block_count,
                )
            )

        schedule_start = time.perf_counter()
        output = scheduler.schedule(prefilling, running)
        scheduler_ms = (time.perf_counter() - schedule_start) * 1000
        scheduler_ms_total += scheduler_ms
        if output is None:
            raise RuntimeError("调度器没有为活跃请求生成执行计划")

        if output.phase == "prefill":
            plans = list(output.prefill_plans)
            recompute_request_ids = [
                plan.state.spec.request_id for plan in plans if plan.state.resuming
            ]
            final_tokens, logits, compute_ms, model_ms, cache_ms, iter_peak = (
                _timed_prefill(
                    model, plans, cache, pad_token_id, device, capture_logits
                )
            )
            peak_memory = max(peak_memory, iter_peak)
            clock_ms += scheduler_ms + compute_ms
            finalized = []
            for plan in plans:
                state = plan.state
                if state.prefill_cursor != plan.start:
                    raise RuntimeError("Prefill cursor 与调度计划不一致")
                state.prefill_cursor = plan.end
                state.cache_length = plan.end
                if cache.sequence_lengths[state.spec.request_id] != plan.end:
                    raise RuntimeError("Paged KV Cache 长度与 Prefill cursor 不一致")
                if plan.end == len(state.prefill_source_tokens):
                    state.prefill_completed_ms = (
                        state.prefill_completed_ms
                        if state.resuming
                        else clock_ms
                    )
                    if state.resuming:
                        state.recompute_completed_ms = clock_ms
                    state.resuming = False
                    state.status = "running"
                    finalized.append(state)
                    if state.spec.request_id in logits:
                        captured_logits[state.spec.request_id].append(
                            logits[state.spec.request_id]
                        )

            for plan in plans:
                if plan.state.spec.request_id in recompute_request_ids:
                    plan.state.recompute_prefill_ms_total += compute_ms

            token_ids = [final_tokens[state.spec.request_id] for state in finalized]
            completed_ids = _record_tokens(
                finalized, token_ids, clock_ms, eos_token_id, stop_on_eos
            )
            completed_states = [
                state for state in finalized if state.status == "finished"
            ]
            release_ms = _release(cache, completed_states, device)
            clock_ms += release_ms
            prefilling = [
                state for state in prefilling
                if state.status in ("prefilling", "recomputing")
            ]
            running.extend(
                state for state in finalized if state.status == "running"
            )
            snapshot = cache.snapshot()
            events.append({
                "iteration": len(events),
                "phase": "prefill",
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
                "recompute_request_ids": recompute_request_ids,
                "scheduled_tokens": output.scheduled_tokens,
                "logical_tokens": output.logical_tokens,
                "token_budget": output.token_budget,
                "executed_batch_size": len(plans),
                "active_requests": len(prefilling) + len(running),
                "swapped_requests": len(swapped),
                "total_ms": scheduler_ms + compute_ms + release_ms,
                "scheduler_ms": scheduler_ms,
                "model_ms": model_ms,
                "cache_management_ms": cache_ms + release_ms,
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
        (
            token_ids, decode_logits, compute_ms, model_ms, cache_ms, visited,
            iter_peak,
        ) = (
            _timed_decode(
                model, running, cache, max_running_requests, pad_token_id, device,
                capture_logits,
            )
        )
        for state in running:
            if state.spec.request_id in decode_logits:
                captured_logits[state.spec.request_id].append(
                    decode_logits[state.spec.request_id]
                )
        peak_memory = max(peak_memory, iter_peak)
        active_before = len(running)
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
            "scheduled_request_ids": output.request_ids,
            "completed": completed_ids,
            "chunk_ranges": [],
            "scheduled_tokens": output.scheduled_tokens,
            "logical_tokens": output.logical_tokens,
            "token_budget": output.token_budget,
            "executed_batch_size": max_running_requests,
            "active_requests": active_before,
            "swapped_requests": len(swapped),
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
        states, events, resource_events, config, peak_memory, cache, cpu_pool,
        scheduler_ms_total, preempt_mode, admission_mode,
        logical_concurrency_peak,
    )
    return {
        "new_token_ids": [state.generated for state in states],
        "first_token_logits": {
            request_id: values[0] for request_id, values in captured_logits.items()
            if values
        },
        "token_logits": dict(captured_logits),
        "request_metrics": request_metrics,
        "events": events,
        "resource_events": resource_events,
        "metrics": metrics,
    }
