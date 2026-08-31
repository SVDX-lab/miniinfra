"""第 15 期 monolithic 与 Prefill/Decode 分离的独立执行核心。"""

import hashlib
import json
import math
import time

import torch

from handoff_protocol import namespace_digest, payload_digest
from paged_cache import DTYPE_NAMES, PagedKVCache


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _new_kv_only(present, count):
    return [
        (key[:, :, -count:, :], value[:, :, -count:, :])
        for key, value in present
    ]


def build_namespace(model_id, revision, model, dtype, block_size):
    config = model.config
    return {
        "format_version": 1,
        "model_id": model_id,
        "model_revision": revision,
        "dtype": DTYPE_NAMES[dtype],
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "rope_theta": config.rope_theta,
        "engine_block_size": int(block_size),
        "kv_layout": "layer,kv,head,token,dim",
    }


def synthetic_tokens(length, vocab_size):
    if length < 1:
        raise ValueError("Prompt 长度必须大于 0")
    usable = max(1, vocab_size - 1000)
    return [1000 + ((index * 7919 + 17) % usable) for index in range(length)]


def _logits_digest(logits):
    return hashlib.sha256(logits.numpy().tobytes()).hexdigest()


@torch.inference_mode()
def prefill_request(
    model, token_ids, device, block_size=16, token_budget=256,
    request_id="request-0", max_new_tokens=4,
):
    if not token_ids or token_budget < 1:
        raise ValueError("Prompt 不能为空，Token Budget 必须大于 0")
    device = torch.device(device)
    dtype = next(model.parameters()).dtype
    max_blocks = math.ceil((len(token_ids) + max_new_tokens) / block_size) + 1
    cache = PagedKVCache(model.config, block_size, max_blocks, device, dtype)
    cache.create_request(request_id)
    cursor = 0
    first_logits = None
    model_ms = 0.0
    synchronize(device)
    start_total = time.perf_counter()
    while cursor < len(token_ids):
        end = min(cursor + token_budget, len(token_ids))
        values = torch.tensor(
            [token_ids[cursor:end]], dtype=torch.long, device=device
        )
        positions = torch.arange(
            cursor, end, dtype=torch.long, device=device
        ).unsqueeze(0)
        mask = torch.ones((1, end), dtype=torch.bool, device=device)
        past = cache.dense(request_id)
        synchronize(device)
        start = time.perf_counter()
        logits, present = model(
            values, attention_mask=mask, position_ids=positions,
            past_key_values=past, use_cache=True,
        )
        synchronize(device)
        model_ms += (time.perf_counter() - start) * 1000
        count = end - cursor
        cache.append(request_id, _new_kv_only(present, count))
        cursor = end
        if cursor == len(token_ids):
            first_logits = logits[0, -1].float().cpu()
    first_token = int(torch.argmax(first_logits).item())
    prefill_ready_ms = (time.perf_counter() - start_total) * 1000
    return cache, {
        "first_token": first_token,
        "first_logits_sha256": _logits_digest(first_logits),
        "prefill_model_ms": model_ms,
        "prefill_ready_ms": prefill_ready_ms,
    }


def export_handoff(
    cache, request_id, token_ids, first_token, first_logits_sha256,
    namespace, attempt_id,
):
    device = cache.device
    synchronize(device)
    start = time.perf_counter()
    payload = cache.export_chunk(request_id, 0, len(token_ids))
    synchronize(device)
    export_ms = (time.perf_counter() - start) * 1000
    manifest = {
        "request_id": request_id,
        "attempt_id": attempt_id,
        "namespace": namespace,
        "namespace_digest": namespace_digest(namespace),
        "prompt_tokens": len(token_ids),
        "prompt_token_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "first_token": int(first_token),
        "first_logits_sha256": first_logits_sha256,
        "payload_bytes": len(payload),
        "payload_sha256": payload_digest(payload),
        "shape": list(cache.external_shape(len(token_ids))),
    }
    return manifest, payload, export_ms


@torch.inference_mode()
def decode_from_payload(
    model, manifest, payload, device, expected_namespace,
    max_new_tokens, block_size=16, request_id="request-0",
    on_imported=None,
):
    if manifest["request_id"] != request_id:
        raise ValueError("Request ID 不匹配")
    expected_digest = namespace_digest(expected_namespace)
    if manifest["namespace_digest"] != expected_digest:
        raise ValueError("模型或 KV Namespace 不匹配")
    if payload_digest(payload) != manifest["payload_sha256"]:
        raise ValueError("接收端 Payload SHA-256 校验失败")
    device = torch.device(device)
    dtype = next(model.parameters()).dtype
    prompt_tokens = int(manifest["prompt_tokens"])
    max_blocks = math.ceil((prompt_tokens + max_new_tokens) / block_size) + 1
    cache = PagedKVCache(model.config, block_size, max_blocks, device, dtype)
    cache.create_request(request_id)
    synchronize(device)
    import_start = time.perf_counter()
    cache.import_chunk(request_id, payload, prompt_tokens)
    synchronize(device)
    import_ms = (time.perf_counter() - import_start) * 1000
    if on_imported is not None:
        on_imported()
    generated = [int(manifest["first_token"])]
    decode_ms = 0.0
    decode_step_wall_ms = []
    while len(generated) < max_new_tokens:
        step_start = time.perf_counter()
        position = cache.sequence_lengths[request_id]
        values = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        positions = torch.tensor([[position]], dtype=torch.long, device=device)
        mask = torch.ones((1, position + 1), dtype=torch.bool, device=device)
        past = cache.dense(request_id)
        synchronize(device)
        start = time.perf_counter()
        logits, present = model(
            values, attention_mask=mask, position_ids=positions,
            past_key_values=past, use_cache=True,
        )
        synchronize(device)
        decode_ms += (time.perf_counter() - start) * 1000
        cache.append(request_id, _new_kv_only(present, 1))
        generated.append(int(torch.argmax(logits[0, -1]).item()))
        decode_step_wall_ms.append((time.perf_counter() - step_start) * 1000)
    before_release = cache.snapshot()
    cache.release(request_id)
    return {
        "new_token_ids": generated,
        "metrics": {
            "import_h2d_ms": import_ms,
            "decode_model_ms": decode_ms,
            "decode_step_wall_ms": decode_step_wall_ms,
            "cache_snapshot_before_release": before_release,
            "final_cache_snapshot": cache.snapshot(),
        },
    }


@torch.inference_mode()
def run_monolithic(
    model, token_ids, max_new_tokens, device, block_size=16,
    token_budget=256, request_id="request-0",
):
    synchronize(device)
    total_start = time.perf_counter()
    cache, prefill = prefill_request(
        model, token_ids, device, block_size, token_budget,
        request_id, max_new_tokens,
    )
    generated = [prefill["first_token"]]
    first_token_ready_ms = (time.perf_counter() - total_start) * 1000
    decode_ms = 0.0
    decode_step_wall_ms = []
    while len(generated) < max_new_tokens:
        step_start = time.perf_counter()
        position = cache.sequence_lengths[request_id]
        values = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        positions = torch.tensor([[position]], dtype=torch.long, device=device)
        mask = torch.ones((1, position + 1), dtype=torch.bool, device=device)
        past = cache.dense(request_id)
        synchronize(device)
        start = time.perf_counter()
        logits, present = model(
            values, attention_mask=mask, position_ids=positions,
            past_key_values=past, use_cache=True,
        )
        synchronize(device)
        decode_ms += (time.perf_counter() - start) * 1000
        cache.append(request_id, _new_kv_only(present, 1))
        generated.append(int(torch.argmax(logits[0, -1]).item()))
        decode_step_wall_ms.append((time.perf_counter() - step_start) * 1000)
    total_ms = (time.perf_counter() - total_start) * 1000
    before_release = cache.snapshot()
    cache.release(request_id)
    return {
        "new_token_ids": generated,
        "first_logits_sha256": prefill["first_logits_sha256"],
        "metrics": {
            **prefill, "ttft_ms": first_token_ready_ms,
            "decode_model_ms": decode_ms,
            "decode_step_wall_ms": decode_step_wall_ms,
            "end_to_end_ms": total_ms,
            "cache_snapshot_before_release": before_release,
            "final_cache_snapshot": cache.snapshot(),
        },
    }
