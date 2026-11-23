"""第 13 期完整引擎：在相同抢占策略下比较同步与异步 KV 传输。"""

import time
from collections import defaultdict

import torch

from engine_base import (
    _finalize,
    _growth_need,
    _record_tokens,
    _release,
    _timed_decode,
    _timed_prefill,
)
from paged_cache import CPUPinnedPool, PagedKVCache
from scheduler import ChunkedPrefillScheduler, RequestState, SchedulerConfig
from transfer import KVTransferManager, TRANSFER_MODES


def _free_slots(prefilling, running, swapping_in, capacity):
    used = {
        state.slot_index
        for state in prefilling + running + [item[0] for item in swapping_in]
        if state.slot_index is not None
    }
    return [slot for slot in range(capacity) if slot not in used]


def _next_phase_is_decode(scheduler, prefilling, running):
    if running and scheduler.last_phase == "prefill":
        return True
    if prefilling:
        return False
    return bool(running)


def _transfer_metrics(resource_events, transfer_mode):
    transfers = [
        event for event in resource_events
        if event["type"] in ("swap_out", "swap_in")
    ]
    device_ms = sum(event["device_ms"] for event in transfers)
    exposed_ms = sum(event["exposed_wait_ms"] for event in transfers)
    return {
        "transfer_mode": transfer_mode,
        "transfer_device_ms_total": device_ms,
        "transfer_submit_wall_ms_total": sum(
            event["submit_wall_ms"] for event in transfers
        ),
        "transfer_exposed_wait_ms_total": exposed_ms,
        # 这是“未作为显式等待暴露的传输时间”，不是严格的 GPU kernel 重叠率。
        "transfer_non_exposed_ms_total": max(0.0, device_ms - exposed_ms),
        "d2h_device_ms_total": sum(
            event["device_ms"] for event in transfers
            if event["direction"] == "d2h"
        ),
        "h2d_device_ms_total": sum(
            event["device_ms"] for event in transfers
            if event["direction"] == "h2d"
        ),
        "d2h_exposed_wait_ms_total": sum(
            event["exposed_wait_ms"] for event in transfers
            if event["direction"] == "d2h"
        ),
        "h2d_exposed_wait_ms_total": sum(
            event["exposed_wait_ms"] for event in transfers
            if event["direction"] == "h2d"
        ),
        "transfer_events": len(transfers),
    }


@torch.inference_mode()
def run_engine(
    model,
    request_specs,
    max_running_requests,
    eos_token_id,
    device,
    token_budget,
    transfer_mode="sync",
    admission_mode="incremental",
    pool_blocks=None,
    cpu_pool_blocks=None,
    block_size=16,
    pad_token_id=0,
    stop_on_eos=True,
    capture_logits=False,
):
    """运行独立的 Chunked Prefill + Paged Cache + KV Offloading 引擎。

    两个实验组唯一不同的是 `transfer_mode`。异步 D2H 完成前不释放 GPU Block，
    异步 H2D 完成前不让请求回到 running；没有可执行计算时才显式等待 Event。
    """
    if not request_specs:
        raise ValueError("request_specs 不能为空")
    if transfer_mode not in TRANSFER_MODES:
        raise ValueError("transfer_mode 必须是 %s" % (TRANSFER_MODES,))
    if admission_mode not in ("incremental", "conservative"):
        raise ValueError("未知 admission_mode: " + admission_mode)
    for spec in request_specs:
        spec.validate()

    config = SchedulerConfig(max_running_requests, token_budget)
    scheduler = ChunkedPrefillScheduler(config)
    dtype = next(model.parameters()).dtype
    worst_case_blocks = {
        spec.request_id: (
            len(spec.token_ids) + spec.max_new_tokens + block_size - 1
        ) // block_size
        for spec in request_specs
    }
    largest_blocks = max(worst_case_blocks.values())
    total_blocks = sum(worst_case_blocks.values())
    pool_blocks = total_blocks if pool_blocks is None else pool_blocks
    if pool_blocks < largest_blocks:
        raise ValueError(
            "GPU Block Pool 过小：单请求最坏需要 %d 块，当前只有 %d 块"
            % (largest_blocks, pool_blocks)
        )
    cpu_pool_blocks = total_blocks if cpu_pool_blocks is None else cpu_pool_blocks
    if cpu_pool_blocks < total_blocks:
        raise ValueError(
            "CPU Pinned Pool 配置不足：本负载最坏需要 %d 块，当前配置 %d 块"
            % (total_blocks, cpu_pool_blocks)
        )

    cache = PagedKVCache(model.config, block_size, pool_blocks, device, dtype)
    cpu_pool = CPUPinnedPool(
        model.config, block_size, cpu_pool_blocks, dtype,
        pin_memory=device.type == "cuda",
    )
    transfers = KVTransferManager(device, transfer_mode)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    states = [RequestState(spec) for spec in request_specs]
    waiting = sorted(states, key=lambda state: state.spec.arrival_ms)
    prefilling = []
    running = []
    swapping_out = []
    swapped = []
    swapping_in = []
    events = []
    resource_events = []
    captured_logits = defaultdict(list)
    clock_ms = min(state.spec.arrival_ms for state in states)
    scheduler_ms_total = 0.0
    peak_memory = 0
    admission_counter = 0
    logical_concurrency_peak = 0

    while (
        waiting or prefilling or running or swapping_out or swapped or swapping_in
    ):
        logical_concurrency_peak = max(
            logical_concurrency_peak,
            len(prefilling) + len(running) + len(swapping_out)
            + len(swapped) + len(swapping_in),
        )

        # 0) 只轮询已完成 Event；query 本身不等待。
        for state, task in list(swapping_out):
            if not transfers.query(task):
                continue
            stats = transfers.finish(task, cache, cpu_pool, clock_ms)
            stats["clock_ms"] = clock_ms
            state.status = "swapped"
            state.slot_index = None
            state.swap_out_count += 1
            swapping_out.remove((state, task))
            swapped.append(state)
            resource_events.append(stats)

        for state, task in list(swapping_in):
            if not transfers.query(task):
                continue
            stats = transfers.finish(task, cache, cpu_pool, clock_ms)
            state.status = "running"
            state.swap_in_count += 1
            state.resume_count += 1
            pause_ms = clock_ms - state.preempted_clock_ms
            state.paused_ms_total += pause_ms
            stats["pause_ms"] = pause_ms
            stats["clock_ms"] = clock_ms
            swapping_in.remove((state, task))
            running.append(state)
            resource_events.append(stats)

        # 1) 抢占策略与第 12 期一致。异步路径用“待释放块数”判断需提交几个受害者，
        # 但 Event 完成前这些块仍计为 GPU used blocks。
        if admission_mode == "incremental":
            need = _growth_need(running, block_size)
            projected_free = cache.free_block_count + sum(
                task.blocks for _, task in swapping_out
            )
            while projected_free < need:
                if not running:
                    break
                victim = max(running, key=lambda state: state.admission_seq)
                request_id = victim.spec.request_id
                coresident_ids = [
                    state.spec.request_id for state in running if state is not victim
                ]
                victim.status = "swapping_out"
                victim.slot_index = None
                victim.preempt_count += 1
                victim.preempted_clock_ms = clock_ms
                running.remove(victim)
                task = transfers.submit_swap_out(
                    cache, cpu_pool, request_id, clock_ms
                )
                task.metadata.update({
                    "reason": "decode_growth",
                    "coresident_request_ids": coresident_ids,
                })
                swapping_out.append((victim, task))
                clock_ms += task.submit_wall_ms
                projected_free = cache.free_block_count + sum(
                    pending.blocks for _, pending in swapping_out
                )
                need = _growth_need(running, block_size)
                if transfers.query(task):
                    break

            # 同步任务提交后已经完成，在本轮立即兑现资源释放。
            if transfer_mode == "sync" and swapping_out:
                continue

        # 2) swapped FCFS 恢复。H2D 在提交时占用 GPU Block 和运行名额，完成后
        # 才能进入 Decode。
        while swapped:
            occupied = len(prefilling) + len(running) + len(swapping_in)
            if occupied >= max_running_requests:
                break
            head = min(swapped, key=lambda state: state.admission_seq)
            request_id = head.spec.request_id
            resume_need = cpu_pool.blocks_needed(request_id)
            growth_reserve = len(running) + 1
            if cache.free_block_count < resume_need + growth_reserve:
                break
            slot = _free_slots(
                prefilling, running, swapping_in, max_running_requests
            )[0]
            swapped.remove(head)
            head.status = "swapping_in"
            head.slot_index = slot
            task = transfers.submit_swap_in(
                cache, cpu_pool, request_id, clock_ms
            )
            task.metadata["coresident_request_ids"] = [
                state.spec.request_id for state in running
            ]
            swapping_in.append((head, task))
            clock_ms += task.submit_wall_ms
            if transfer_mode == "sync":
                break

        if transfer_mode == "sync" and swapping_in:
            continue

        # 3) 新请求准入。
        for state in list(waiting):
            if state.spec.arrival_ms > clock_ms:
                break
            occupied = len(prefilling) + len(running) + len(swapping_in)
            if occupied >= max_running_requests:
                break
            request_id = state.spec.request_id
            reserved_tokens = (
                len(state.spec.token_ids) + state.spec.max_new_tokens
                if admission_mode == "conservative"
                else len(state.spec.token_ids)
            )
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
            state.slot_index = _free_slots(
                prefilling, running, swapping_in, max_running_requests
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

        # 4) 下一轮若必须 Decode、但 D2H 尚未兑现 Block，只等待到真正的数据依赖。
        if (
            _next_phase_is_decode(scheduler, prefilling, running)
            and cache.free_block_count < _growth_need(running, block_size)
        ):
            if not swapping_out:
                raise RuntimeError("没有待完成 D2H，却无法满足 Decode Block 增长")
            wait_ms = transfers.wait(swapping_out[0][1])
            clock_ms += wait_ms
            continue

        # 没有可计算请求时，等待最早的在途传输；否则推进到下一个逻辑到达时刻。
        if not prefilling and not running:
            in_flight = swapping_out + swapping_in
            if in_flight:
                wait_ms = transfers.wait(in_flight[0][1])
                clock_ms += wait_ms
                continue
            if waiting and all(state.spec.arrival_ms > clock_ms for state in waiting):
                clock_ms = min(state.spec.arrival_ms for state in waiting)
                continue
            raise RuntimeError(
                "没有可推进请求：waiting=%d swapped=%d free_blocks=%d"
                % (len(waiting), len(swapped), cache.free_block_count)
            )

        schedule_start = time.perf_counter()
        output = scheduler.schedule(prefilling, running)
        scheduler_ms = (time.perf_counter() - schedule_start) * 1000
        scheduler_ms_total += scheduler_ms
        if output is None:
            raise RuntimeError("调度器没有为活跃请求生成执行计划")

        if output.phase == "prefill":
            plans = list(output.prefill_plans)
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
                if plan.end == len(state.spec.token_ids):
                    state.prefill_completed_ms = clock_ms
                    state.status = "running"
                    finalized.append(state)
                    if state.spec.request_id in logits:
                        captured_logits[state.spec.request_id].append(
                            logits[state.spec.request_id]
                        )

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
                state for state in prefilling if state.status == "prefilling"
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
                "recompute_request_ids": [],
                "scheduled_tokens": output.scheduled_tokens,
                "logical_tokens": output.logical_tokens,
                "token_budget": output.token_budget,
                "executed_batch_size": len(plans),
                "active_requests": len(prefilling) + len(running),
                "swapped_requests": len(swapped) + len(swapping_out),
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

        if output.phase != "decode" or list(output.selected) != running:
            raise RuntimeError("Decode 计划与 running 集合不一致")
        (
            token_ids, decode_logits, compute_ms, model_ms, cache_ms, visited,
            iter_peak,
        ) = _timed_decode(
            model, running, cache, max_running_requests, pad_token_id, device,
            capture_logits,
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
            "swapped_requests": len(swapped) + len(swapping_out),
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
        scheduler_ms_total, "swap", admission_mode, logical_concurrency_peak,
    )
    metrics.update(_transfer_metrics(resource_events, transfer_mode))
    # PCIe 有效带宽必须用 CUDA Event 的设备传输时间，不能用 enqueue 或仅剩余的
    # exposed wait。保留 wall_ms 字段用于端到端暴露成本。
    metrics["swap_out_gb_per_second"] = (
        metrics["swap_out_bytes_total"]
        / (metrics["d2h_device_ms_total"] / 1000) / 1e9
        if metrics["d2h_device_ms_total"] else 0.0
    )
    metrics["swap_in_gb_per_second"] = (
        metrics["swap_in_bytes_total"]
        / (metrics["h2d_device_ms_total"] / 1000) / 1e9
        if metrics["h2d_device_ms_total"] else 0.0
    )
    return {
        "new_token_ids": [state.generated for state in states],
        "first_token_logits": {
            request_id: values[0]
            for request_id, values in captured_logits.items() if values
        },
        "token_logits": dict(captured_logits),
        "request_metrics": request_metrics,
        "events": events,
        "resource_events": resource_events,
        "metrics": metrics,
    }
