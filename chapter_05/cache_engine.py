"""第 05 期 dense 与 Paged KV Cache 的独立 Continuous Batching 引擎。

两条路径共享 FCFS、固定最大运行请求数和 Prefill 优先规则，唯一核心变量是
KV Cache 的物理存储和 Attention 读取方式。
"""

import time
import math
from dataclasses import dataclass, field

import torch

from qwen3_model import cache_size_bytes
from paged_cache import PagedKVCache, paged_decode_forward


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
        input_ids[row, maximum - length :] = sequence
        attention_mask[row, maximum - length :] = True
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.clamp_(min=0)
    return input_ids, attention_mask, position_ids, lengths


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    token_ids: tuple
    max_new_tokens: int
    arrival_ms: float = 0.0

    def validate(self):
        if not self.request_id:
            raise ValueError("request_id 不能为空")
        if not self.token_ids:
            raise ValueError("Prompt Token 不能为空")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须大于 0")
        if self.arrival_ms < 0:
            raise ValueError("arrival_ms 不能小于 0")


@dataclass
class RequestState:
    spec: RequestSpec
    status: str = "waiting"
    generated: list = field(default_factory=list)
    slot_index: int | None = None
    cache_length: int = 0
    admitted_ms: float | None = None
    first_token_ms: float | None = None
    completion_ms: float | None = None
    token_times_ms: list = field(default_factory=list)


def make_request_specs(token_sequences, max_new_tokens, arrival_times_ms=None):
    sequences = [tuple(int(token) for token in sequence) for sequence in token_sequences]
    if isinstance(max_new_tokens, int):
        budgets = [max_new_tokens] * len(sequences)
    else:
        budgets = list(max_new_tokens)
    arrivals = (
        [0.0] * len(sequences)
        if arrival_times_ms is None else list(arrival_times_ms)
    )
    if len(budgets) != len(sequences) or len(arrivals) != len(sequences):
        raise ValueError("Prompt、输出预算和到达时间的数量必须一致")
    specs = [
        RequestSpec(str(index), sequence, int(budget), float(arrival))
        for index, (sequence, budget, arrival) in enumerate(
            zip(sequences, budgets, arrivals)
        )
    ]
    for spec in specs:
        spec.validate()
    if len({spec.request_id for spec in specs}) != len(specs):
        raise ValueError("request_id 必须唯一")
    return specs


def _split_prefill_cache(past_key_values, prompt_lengths):
    request_caches = []
    for row, length in enumerate(prompt_lengths):
        request_caches.append([
            (
                key[row : row + 1, :, -length:, :].clone(),
                value[row : row + 1, :, -length:, :].clone(),
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
    model_start = total_start
    logits, past = model(
        input_ids,
        attention_mask=mask,
        position_ids=positions,
        use_cache=True,
    )
    next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
    synchronize(device)
    model_ms = (time.perf_counter() - model_start) * 1000

    cache_start = time.perf_counter()
    caches = _split_prefill_cache(past, lengths)
    synchronize(device)
    cache_ms = (time.perf_counter() - cache_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    return next_tokens.cpu().tolist(), caches, total_ms, model_ms, cache_ms


def _merge_admitted_caches(
    packed_cache,
    history_mask,
    admitted,
    request_caches,
    capacity,
    device,
):
    """把新请求 Cache 写入持久化 dense 槽位，必要时左侧扩容。"""

    synchronize(device)
    start = time.perf_counter()
    existing_length = 0 if history_mask is None else history_mask.shape[1]
    target_length = max(
        [existing_length] + [state.cache_length for state in admitted]
    )
    sample_key, _ = request_caches[0][0]
    moved_bytes = 0
    if packed_cache is None:
        packed_cache = []
        for layer_index in range(len(request_caches[0])):
            layer_key, _ = request_caches[0][layer_index]
            shape = (
                capacity,
                layer_key.shape[1],
                target_length,
                layer_key.shape[3],
            )
            key = torch.zeros(shape, dtype=sample_key.dtype, device=device)
            packed_cache.append((key, torch.zeros_like(key)))
        history_mask = torch.zeros(
            (capacity, target_length), dtype=torch.bool, device=device
        )
    elif target_length > existing_length:
        padding = target_length - existing_length
        moved_bytes = cache_size_bytes(packed_cache)
        expanded = []
        for key, value in packed_cache:
            zero_shape = (capacity, key.shape[1], padding, key.shape[3])
            key_padding = torch.zeros(zero_shape, dtype=key.dtype, device=device)
            value_padding = torch.zeros_like(key_padding)
            expanded.append((
                torch.cat((key_padding, key), dim=2),
                torch.cat((value_padding, value), dim=2),
            ))
        packed_cache = expanded
        history_mask = torch.cat((
            torch.zeros((capacity, padding), dtype=torch.bool, device=device),
            history_mask,
        ), dim=1)

    for state, request_cache in zip(admitted, request_caches):
        slot = state.slot_index
        length = state.cache_length
        history_mask[slot].zero_()
        history_mask[slot, target_length - length :] = True
        for layer_index, (request_key, request_value) in enumerate(request_cache):
            packed_key, packed_value = packed_cache[layer_index]
            packed_key[slot].zero_()
            packed_value[slot].zero_()
            packed_key[slot, :, target_length - length :, :] = request_key[0]
            packed_value[slot, :, target_length - length :, :] = request_value[0]
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return packed_cache, history_mask, elapsed_ms, moved_bytes


def _timed_dense_decode(
    model,
    running,
    packed_cache,
    history_mask,
    capacity,
    pad_token_id,
    device,
):
    synchronize(device)
    prepare_start = time.perf_counter()
    input_ids = torch.full(
        (capacity, 1), pad_token_id, dtype=torch.long, device=device
    )
    position_ids = torch.zeros((capacity, 1), dtype=torch.long, device=device)
    active = torch.zeros(capacity, dtype=torch.bool, device=device)
    for state in running:
        slot = state.slot_index
        input_ids[slot, 0] = state.generated[-1]
        position_ids[slot, 0] = state.cache_length
        active[slot] = True
    attention_mask = torch.cat((history_mask, active.unsqueeze(1)), dim=1)
    synchronize(device)
    prepare_ms = (time.perf_counter() - prepare_start) * 1000

    model_start = time.perf_counter()
    logits, present = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=packed_cache,
        use_cache=True,
    )
    next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
    synchronize(device)
    model_ms = (time.perf_counter() - model_start) * 1000
    selected_tokens = [int(next_tokens[state.slot_index].item()) for state in running]
    del logits, next_tokens
    return (
        selected_tokens,
        present,
        attention_mask,
        prepare_ms + model_ms,
        model_ms,
        prepare_ms,
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


def _finalize(
    states, events, max_running_requests, peak_memory_bytes, extra_metrics=None
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
    capacity_area = sum(
        event["active_requests"] * event["total_ms"] for event in events
    )
    itls = [
        right - left
        for state in ordered
        for left, right in zip(state.token_times_ms, state.token_times_ms[1:])
    ]
    request_metrics = []
    for state in ordered:
        request_metrics.append({
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
        })
    result = {
        "new_token_ids": [state.generated for state in ordered],
        "request_metrics": request_metrics,
        "events": events,
        "metrics": {
            "request_count": len(ordered),
            "max_running_requests": max_running_requests,
            "makespan_ms": makespan_ms,
            "busy_ms": busy_ms,
            "model_ms": model_ms,
            "cache_management_ms": cache_ms,
            "cache_management_fraction": cache_ms / busy_ms if busy_ms else 0.0,
            "request_throughput_per_second": (
                len(ordered) * 1000 / makespan_ms if makespan_ms else 0.0
            ),
            "output_token_throughput_per_second": (
                output_tokens * 1000 / makespan_ms if makespan_ms else 0.0
            ),
            "execution_slot_utilization": (
                output_tokens / executed_rows if executed_rows else 0.0
            ),
            "running_capacity_utilization": (
                capacity_area / (max_running_requests * busy_ms)
                if busy_ms else 0.0
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
            "peak_memory_bytes": peak_memory_bytes,
            "peak_live_request_cache_bytes": max(
                (event["live_request_cache_bytes"] for event in events),
                default=0,
            ),
            "peak_pool_cache_bytes": max(
                (event.get("pool_cache_bytes", 0) for event in events),
                default=0,
            ),
            "cache_resize_moved_bytes": sum(
                event.get("cache_resize_moved_bytes", 0) for event in events
            ),
            "visited_kv_token_slots": sum(
                event.get("visited_kv_token_slots", 0) for event in events
            ),
        },
    }
    if extra_metrics:
        result["metrics"].update(extra_metrics)
    return result


@torch.inference_mode()
def run_dense_cache(
    model,
    request_specs,
    max_running_requests,
    eos_token_id,
    device,
    pad_token_id=0,
    stop_on_eos=True,
):
    """FCFS Continuous Batching；有空位时优先 Prefill 已到达请求。"""

    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")
    if not request_specs:
        raise ValueError("request_specs 不能为空")
    for spec in request_specs:
        spec.validate()
    states = [RequestState(spec) for spec in request_specs]
    # Python 排序稳定；到达时间相同时保留调用方提交顺序，作为 FCFS 次序。
    by_arrival = sorted(states, key=lambda state: state.spec.arrival_ms)
    waiting = list(by_arrival)
    running = []
    packed_cache = None
    history_mask = None
    events = []
    clock_ms = min(state.spec.arrival_ms for state in states)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while waiting or running:
        free_slots = max_running_requests - len(running)
        ready = [state for state in waiting if state.spec.arrival_ms <= clock_ms]
        admitted = ready[:free_slots]
        if admitted:
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
            token_ids, caches, total_ms, model_ms, cache_ms = _timed_prefill(
                model, admitted, pad_token_id, device
            )
            packed_cache, history_mask, merge_ms, moved_bytes = _merge_admitted_caches(
                packed_cache,
                history_mask,
                admitted,
                caches,
                max_running_requests,
                device,
            )
            total_ms += merge_ms
            cache_ms += merge_ms
            del caches
            running.extend(admitted)
            clock_ms += total_ms
            active_count = len(running)
            completed = _record_tokens(
                admitted, token_ids, clock_ms, eos_token_id, stop_on_eos
            )
            for state in admitted:
                if state.status == "finished":
                    history_mask[state.slot_index].zero_()
            running = [state for state in running if state.status == "running"]
            if not running:
                packed_cache = None
                history_mask = None
            events.append({
                "phase": "prefill",
                "admitted": [state.spec.request_id for state in admitted],
                "completed": completed,
                "executed_batch_size": len(admitted),
                "active_requests": active_count,
                "total_ms": total_ms,
                "model_ms": model_ms,
                "cache_management_ms": cache_ms,
                "live_request_cache_bytes": cache_size_bytes(packed_cache),
                "pool_cache_bytes": cache_size_bytes(packed_cache),
                "cache_resize_moved_bytes": moved_bytes,
                "visited_kv_token_slots": 0,
            })
            continue

        if running:
            active_before = len(running)
            token_ids, packed_cache, history_mask, total_ms, model_ms, cache_ms = (
                _timed_dense_decode(
                    model,
                    running,
                    packed_cache,
                    history_mask,
                    max_running_requests,
                    pad_token_id,
                    device,
                )
            )
            for state in running:
                state.cache_length += 1
            visited_slots = max_running_requests * history_mask.shape[1]
            clock_ms += total_ms
            completed = _record_tokens(
                running, token_ids, clock_ms, eos_token_id, stop_on_eos
            )
            for state in running:
                if state.status == "finished":
                    history_mask[state.slot_index].zero_()
            running = [state for state in running if state.status == "running"]
            if not running:
                packed_cache = None
                history_mask = None
            events.append({
                "phase": "decode",
                "admitted": [],
                "completed": completed,
                "executed_batch_size": max_running_requests,
                "active_requests": active_before,
                "total_ms": total_ms,
                "model_ms": model_ms,
                "cache_management_ms": cache_ms,
                "live_request_cache_bytes": cache_size_bytes(packed_cache),
                "pool_cache_bytes": cache_size_bytes(packed_cache),
                "cache_resize_moved_bytes": 0,
                "visited_kv_token_slots": (
                    visited_slots
                ),
            })
            continue

        clock_ms = min(state.spec.arrival_ms for state in waiting)

    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    return _finalize(states, events, max_running_requests, peak_memory)


def _store_paged_prefill(cache, admitted, request_caches, device):
    synchronize(device)
    start = time.perf_counter()
    for state, request_cache in zip(admitted, request_caches):
        cache.store_prefill(state.spec.request_id, request_cache)
    synchronize(device)
    return (time.perf_counter() - start) * 1000


def _release_paged(cache, completed_states, device):
    if not completed_states:
        return 0.0
    synchronize(device)
    start = time.perf_counter()
    for state in completed_states:
        cache.release(state.spec.request_id)
    synchronize(device)
    return (time.perf_counter() - start) * 1000


def _timed_paged_decode(
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


@torch.inference_mode()
def run_paged_cache(
    model,
    request_specs,
    max_running_requests,
    eos_token_id,
    device,
    block_size=16,
    pad_token_id=0,
    stop_on_eos=True,
):
    """使用 Block Pool 和 Block Table 的 FCFS Continuous Batching。"""

    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")
    if not request_specs:
        raise ValueError("request_specs 不能为空")
    for spec in request_specs:
        spec.validate()
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
    cache = PagedKVCache(
        model.config, block_size, max_blocks, device, dtype
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while waiting or running:
        free_slots = max_running_requests - len(running)
        ready = [state for state in waiting if state.spec.arrival_ms <= clock_ms]
        admitted = ready[:free_slots]
        if admitted:
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
            token_ids, caches, total_ms, model_ms, cache_ms = _timed_prefill(
                model, admitted, pad_token_id, device
            )
            store_ms = _store_paged_prefill(cache, admitted, caches, device)
            del caches
            total_ms += store_ms
            cache_ms += store_ms
            running.extend(admitted)
            clock_ms += total_ms
            active_count = len(running)
            completed_ids = _record_tokens(
                admitted, token_ids, clock_ms, eos_token_id, stop_on_eos
            )
            completed_states = [
                state for state in admitted if state.status == "finished"
            ]
            release_ms = _release_paged(cache, completed_states, device)
            total_ms += release_ms
            cache_ms += release_ms
            clock_ms += release_ms
            running = [state for state in running if state.status == "running"]
            snapshot = cache.snapshot()
            events.append({
                "phase": "prefill",
                "admitted": [state.spec.request_id for state in admitted],
                "completed": completed_ids,
                "executed_batch_size": len(admitted),
                "active_requests": active_count,
                "total_ms": total_ms,
                "model_ms": model_ms,
                "cache_management_ms": cache_ms,
                "live_request_cache_bytes": snapshot["live_cache_bytes"],
                "pool_cache_bytes": snapshot["pool_bytes"],
                "cache_resize_moved_bytes": 0,
                "visited_kv_token_slots": 0,
                "cache_snapshot": snapshot,
            })
            continue

        if running:
            active_before = len(running)
            token_ids, total_ms, model_ms, cache_ms, visited_slots = (
                _timed_paged_decode(
                    model, running, cache, max_running_requests,
                    pad_token_id, device,
                )
            )
            for state in running:
                state.cache_length += 1
            clock_ms += total_ms
            completed_ids = _record_tokens(
                running, token_ids, clock_ms, eos_token_id, stop_on_eos
            )
            completed_states = [
                state for state in running if state.status == "finished"
            ]
            release_ms = _release_paged(cache, completed_states, device)
            total_ms += release_ms
            cache_ms += release_ms
            clock_ms += release_ms
            running = [state for state in running if state.status == "running"]
            snapshot = cache.snapshot()
            events.append({
                "phase": "decode",
                "admitted": [],
                "completed": completed_ids,
                "executed_batch_size": max_running_requests,
                "active_requests": active_before,
                "total_ms": total_ms,
                "model_ms": model_ms,
                "cache_management_ms": cache_ms,
                "live_request_cache_bytes": snapshot["live_cache_bytes"],
                "pool_cache_bytes": snapshot["pool_bytes"],
                "cache_resize_moved_bytes": 0,
                "visited_kv_token_slots": visited_slots,
                "cache_snapshot": snapshot,
            })
            continue

        clock_ms = min(state.spec.arrival_ms for state in waiting)

    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    return _finalize(
        states,
        events,
        max_running_requests,
        peak_memory,
        extra_metrics={
            "block_size": block_size,
            "bytes_per_block": cache.bytes_per_block,
            "peak_used_blocks": cache.peak_used_blocks,
            "pool_blocks": cache.max_blocks,
            "block_allocation_count": cache.allocation_count,
            "block_reuse_count": cache.reuse_count,
            "block_release_count": cache.release_count,
        },
    )


@torch.inference_mode()
def run_fixed_batching(
    model,
    request_specs,
    max_running_requests,
    eos_token_id,
    device,
    pad_token_id=0,
    stop_on_eos=True,
):
    """固定批次 baseline；一批全部结束前不补入新请求。"""

    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")
    if not request_specs:
        raise ValueError("request_specs 不能为空")
    states = [RequestState(spec) for spec in request_specs]
    for state in states:
        state.spec.validate()
    waiting = sorted(states, key=lambda state: state.spec.arrival_ms)
    events = []
    clock_ms = min(state.spec.arrival_ms for state in states)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while waiting:
        ready = [state for state in waiting if state.spec.arrival_ms <= clock_ms]
        if not ready:
            clock_ms = min(state.spec.arrival_ms for state in waiting)
            ready = [state for state in waiting if state.spec.arrival_ms <= clock_ms]
        wave = ready[:max_running_requests]
        for state in wave:
            waiting.remove(state)
            state.status = "running"
            state.admitted_ms = clock_ms

        input_ids, history_mask, position_ids, _ = left_pad_sequences(
            [state.spec.token_ids for state in wave], pad_token_id, device
        )
        active = torch.ones(len(wave), dtype=torch.bool, device=device)
        past = None
        maximum_budget = max(state.spec.max_new_tokens for state in wave)
        for step in range(maximum_budget):
            active_before = active.clone()
            synchronize(device)
            start = time.perf_counter()
            logits, past = model(
                input_ids,
                attention_mask=history_mask,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
            )
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            synchronize(device)
            total_ms = (time.perf_counter() - start) * 1000
            del logits
            clock_ms += total_ms

            valid_states = []
            valid_tokens = []
            for row, state in enumerate(wave):
                if bool(active_before[row].item()):
                    valid_states.append(state)
                    valid_tokens.append(int(next_tokens[row].item()))
            completed = _record_tokens(
                valid_states, valid_tokens, clock_ms, eos_token_id, stop_on_eos
            )
            for row, state in enumerate(wave):
                if state.status == "finished":
                    active[row] = False
            events.append({
                "phase": "prefill" if step == 0 else "decode",
                "admitted": (
                    [state.spec.request_id for state in wave] if step == 0 else []
                ),
                "completed": completed,
                "executed_batch_size": len(wave),
                "active_requests": int(active_before.sum().item()),
                "total_ms": total_ms,
                "model_ms": total_ms,
                "cache_management_ms": 0.0,
                "live_request_cache_bytes": cache_size_bytes(past),
            })
            if not bool(active.any().item()):
                break
            input_ids = torch.where(
                active, next_tokens, torch.full_like(next_tokens, pad_token_id)
            ).unsqueeze(1)
            history_mask = torch.cat((history_mask, active.unsqueeze(1)), dim=1)
            positions = []
            for row, state in enumerate(wave):
                if bool(active[row].item()):
                    positions.append(len(state.spec.token_ids) + len(state.generated) - 1)
                else:
                    positions.append(0)
            position_ids = torch.tensor(
                positions, dtype=torch.long, device=device
            ).unsqueeze(1)
        del past

    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    return _finalize(states, events, max_running_requests, peak_memory)
