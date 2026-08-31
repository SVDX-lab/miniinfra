"""第 02 期的两条单请求生成路径与计时工具。"""

import time

import torch

from qwen3_model import cache_size_bytes


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, percent):
    """使用 nearest-rank 定义计算小样本延迟分位数。"""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, (len(ordered) * percent + 99) // 100)
    return ordered[min(rank, len(ordered)) - 1]


def _timed_forward(model, model_input, device, past_key_values, use_cache):
    synchronize(device)
    start_time = time.perf_counter()
    logits, present_key_values = model(
        model_input,
        past_key_values=past_key_values,
        use_cache=use_cache,
    )
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    del logits
    return next_token, present_key_values, elapsed_ms


@torch.inference_mode()
def generate_no_cache(
    model,
    input_ids,
    max_new_tokens,
    eos_token_id,
    device,
    stop_on_eos=True,
):
    """Baseline：每个生成步骤都重新计算完整上下文。"""

    generated_ids = input_ids
    new_token_ids = []
    step_times_ms = []
    memory_after_prefill_bytes = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(max_new_tokens):
        next_token, _, elapsed_ms = _timed_forward(
            model,
            generated_ids,
            device,
            past_key_values=None,
            use_cache=False,
        )
        if step == 0 and device.type == "cuda":
            memory_after_prefill_bytes = torch.cuda.memory_allocated(device)

        token_id = next_token.item()
        new_token_ids.append(token_id)
        step_times_ms.append(elapsed_ms)
        generated_ids = torch.cat((generated_ids, next_token), dim=1)
        if stop_on_eos and token_id == eos_token_id:
            break

    return _build_result(
        generated_ids,
        new_token_ids,
        step_times_ms,
        memory_after_prefill_bytes,
        cache_bytes=0,
        device=device,
    )


@torch.inference_mode()
def generate_with_kv_cache(
    model,
    input_ids,
    max_new_tokens,
    eos_token_id,
    device,
    stop_on_eos=True,
):
    """优化组：Prefill 建立 Cache，Decode 每步只输入最新 Token。"""

    generated_ids = input_ids
    new_token_ids = []
    step_times_ms = []
    past_key_values = None
    model_input = input_ids
    memory_after_prefill_bytes = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(max_new_tokens):
        next_token, past_key_values, elapsed_ms = _timed_forward(
            model,
            model_input,
            device,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if step == 0 and device.type == "cuda":
            memory_after_prefill_bytes = torch.cuda.memory_allocated(device)

        token_id = next_token.item()
        new_token_ids.append(token_id)
        step_times_ms.append(elapsed_ms)
        generated_ids = torch.cat((generated_ids, next_token), dim=1)
        model_input = next_token
        if stop_on_eos and token_id == eos_token_id:
            break

    return _build_result(
        generated_ids,
        new_token_ids,
        step_times_ms,
        memory_after_prefill_bytes,
        cache_bytes=cache_size_bytes(past_key_values),
        device=device,
    )


def _build_result(
    generated_ids,
    new_token_ids,
    step_times_ms,
    memory_after_prefill_bytes,
    cache_bytes,
    device,
):
    decode_times = step_times_ms[1:]
    peak_memory_bytes = 0
    if device.type == "cuda":
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)

    metrics = {
        "prefill_ms": step_times_ms[0],
        "decode_times_ms": decode_times,
        "average_decode_ms": (
            sum(decode_times) / len(decode_times) if decode_times else 0.0
        ),
        "p50_decode_ms": percentile(decode_times, 50),
        "p95_decode_ms": percentile(decode_times, 95),
        "end_to_end_ms": sum(step_times_ms),
        "memory_after_prefill_bytes": memory_after_prefill_bytes,
        "peak_memory_bytes": peak_memory_bytes,
        "cache_bytes": cache_bytes,
    }
    return {
        "output_ids": generated_ids,
        "new_token_ids": new_token_ids,
        "metrics": metrics,
    }


def generate(
    mode,
    model,
    input_ids,
    max_new_tokens,
    eos_token_id,
    device,
    stop_on_eos=True,
):
    if mode == "no-cache":
        function = generate_no_cache
    elif mode == "kv-cache":
        function = generate_with_kv_cache
    else:
        raise ValueError("未知生成模式: " + mode)
    return function(
        model,
        input_ids,
        max_new_tokens,
        eos_token_id,
        device,
        stop_on_eos=stop_on_eos,
    )
