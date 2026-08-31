"""外部 KV Cache recompute/external 两条路径的独立单请求教学引擎。"""

import math
import time

import torch

from cache_protocol import TokenChunker
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


@torch.inference_mode()
def run_request(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id,
    device,
    mode="recompute",
    external_client=None,
    model_id="Qwen/Qwen3-0.6B",
    revision="unknown",
    block_size=16,
    external_chunk_size=256,
    token_budget=256,
    stop_on_eos=False,
    capture_logits=False,
    request_id="request-0",
):
    if mode not in ("recompute", "external"):
        raise ValueError("mode 必须是 recompute 或 external")
    if mode == "external" and external_client is None:
        raise ValueError("external 模式必须提供 external_client")
    if not token_ids or max_new_tokens < 1 or token_budget < 1:
        raise ValueError("Prompt、输出预算和 Token Budget 必须非空/大于 0")
    device = torch.device(device)
    dtype = next(model.parameters()).dtype
    maximum_tokens = len(token_ids) + max_new_tokens
    max_blocks = math.ceil(maximum_tokens / block_size) + 1
    cache = PagedKVCache(model.config, block_size, max_blocks, device, dtype)
    cache.create_request(request_id)
    namespace = build_namespace(
        model_id, revision, model, dtype, block_size
    )
    chunker = TokenChunker(namespace, external_chunk_size)
    identities = chunker.identities(token_ids, leave_last_token=True)

    metrics = {
        "mode": mode,
        "prompt_tokens": len(token_ids),
        "external_chunk_size": external_chunk_size,
        "candidate_chunks": len(identities),
        "lookup_ms": 0.0,
        "load_ms": 0.0,
        "network_load_ms": 0.0,
        "import_h2d_ms": 0.0,
        "load_fallback": False,
        "load_error": None,
        "loaded_bytes": 0,
        "hit_chunks": 0,
        "hit_tokens": 0,
        "prefill_model_ms": 0.0,
        "executed_prefill_tokens": 0,
        "store_ms": 0.0,
        "export_d2h_ms": 0.0,
        "store_request_ms": 0.0,
        "store_error": None,
        "stored_bytes": 0,
        "stored_chunks": 0,
        "store_rejected_chunks": 0,
        "decode_model_ms": 0.0,
    }

    synchronize(device)
    service_start = time.perf_counter()
    if mode == "external" and identities:
        start = time.perf_counter()
        try:
            hit_chunks = external_client.lookup(identities)
        except (ConnectionError, OSError, RuntimeError, ValueError) as error:
            hit_chunks = 0
            metrics["load_fallback"] = True
            metrics["load_error"] = "%s: %s" % (type(error).__name__, error)
        metrics["lookup_ms"] = (time.perf_counter() - start) * 1000
        if hit_chunks:
            load_start = time.perf_counter()
            try:
                network_start = time.perf_counter()
                payloads = external_client.load(identities[:hit_chunks])
                metrics["network_load_ms"] = (
                    time.perf_counter() - network_start
                ) * 1000
                import_start = time.perf_counter()
                for identity, payload in zip(identities[:hit_chunks], payloads):
                    cache.import_chunk(request_id, payload, identity.token_count)
                    metrics["loaded_bytes"] += len(payload)
                synchronize(device)
                metrics["import_h2d_ms"] = (
                    time.perf_counter() - import_start
                ) * 1000
            except (ConnectionError, OSError, RuntimeError, ValueError) as error:
                cache.release(request_id)
                cache.create_request(request_id)
                hit_chunks = 0
                metrics["loaded_bytes"] = 0
                metrics["load_fallback"] = True
                metrics["load_error"] = "%s: %s" % (type(error).__name__, error)
            metrics["load_ms"] = (time.perf_counter() - load_start) * 1000
        metrics["hit_chunks"] = hit_chunks
        metrics["hit_tokens"] = hit_chunks * external_chunk_size

    cursor = metrics["hit_tokens"]
    first_logits = None
    while cursor < len(token_ids):
        end = min(cursor + token_budget, len(token_ids))
        values = torch.as_tensor(
            token_ids[cursor:end], dtype=torch.long, device=device
        ).unsqueeze(0)
        positions = torch.arange(
            cursor, end, dtype=torch.long, device=device
        ).unsqueeze(0)
        past = cache.dense(request_id)
        mask = torch.ones((1, end), dtype=torch.bool, device=device)
        synchronize(device)
        model_start = time.perf_counter()
        logits, present = model(
            values,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=past,
            use_cache=True,
        )
        synchronize(device)
        metrics["prefill_model_ms"] += (
            time.perf_counter() - model_start
        ) * 1000
        count = end - cursor
        cache.append(request_id, _new_kv_only(present, count))
        metrics["executed_prefill_tokens"] += count
        cursor = end
        if cursor == len(token_ids):
            first_logits = logits[0, -1].float().cpu()

    first_token = int(torch.argmax(first_logits).item())
    synchronize(device)
    metrics["service_ttft_ms"] = (time.perf_counter() - service_start) * 1000

    if mode == "external" and identities:
        ancestors = []
        for identity in identities:
            # 命中 Chunk 无需重复通过网络 Store；首次 miss 后依次发布完整链。
            if len(ancestors) < metrics["hit_chunks"]:
                ancestors.append(identity.digest)
                continue
            export_start = time.perf_counter()
            payload = cache.export_chunk(
                request_id, identity.token_start, identity.token_count
            )
            synchronize(device)
            export_ms = (time.perf_counter() - export_start) * 1000
            metrics["export_d2h_ms"] += export_ms
            start = time.perf_counter()
            try:
                response = external_client.store(
                    identity=identity,
                    payload=payload,
                    namespace_digest=chunker.namespace_digest,
                    dtype=DTYPE_NAMES[dtype],
                    shape=cache.external_shape(identity.token_count),
                    ancestors=ancestors,
                )
            except (ConnectionError, OSError, RuntimeError, ValueError) as error:
                metrics["store_error"] = "%s: %s" % (type(error).__name__, error)
                metrics["store_ms"] += export_ms + (
                    time.perf_counter() - start
                ) * 1000
                break
            request_ms = (time.perf_counter() - start) * 1000
            metrics["store_request_ms"] += request_ms
            metrics["store_ms"] += export_ms + request_ms
            if response["status"] in ("stored", "exists"):
                metrics["stored_chunks"] += response["status"] == "stored"
                metrics["stored_bytes"] += (
                    len(payload) if response["status"] == "stored" else 0
                )
                ancestors.append(identity.digest)
            else:
                metrics["store_rejected_chunks"] += 1
                break

    generated = [first_token]
    while len(generated) < max_new_tokens:
        if stop_on_eos and generated[-1] == eos_token_id:
            break
        position = cache.sequence_lengths[request_id]
        values = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        positions = torch.tensor([[position]], dtype=torch.long, device=device)
        mask = torch.ones((1, position + 1), dtype=torch.bool, device=device)
        past = cache.dense(request_id)
        synchronize(device)
        start = time.perf_counter()
        logits, present = model(
            values,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=past,
            use_cache=True,
        )
        synchronize(device)
        metrics["decode_model_ms"] += (time.perf_counter() - start) * 1000
        cache.append(request_id, _new_kv_only(present, 1))
        generated.append(int(torch.argmax(logits[0, -1]).item()))

    metrics["end_to_end_ms"] = (
        metrics["service_ttft_ms"]
        + metrics["store_ms"]
        + metrics["decode_model_ms"]
    )
    metrics["cache_snapshot_before_release"] = cache.snapshot()
    metrics["namespace_digest"] = chunker.namespace_digest
    result = {
        "new_token_ids": generated,
        "first_token_logits": first_logits if capture_logits else None,
        "metrics": metrics,
    }
    cache.release(request_id)
    result["final_cache_snapshot"] = cache.snapshot()
    return result
