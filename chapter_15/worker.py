"""Controller 启动的独立 monolithic、Prefill 与 Decode Worker。"""

import argparse
import json
import os
import time

import torch

from engine import (
    build_namespace, decode_from_payload, export_handoff, prefill_request,
    run_monolithic, synthetic_tokens,
)
from experiment_utils import (
    add_model_arguments, environment_record, load_model, parse_dtype, write_json,
)
from handoff_protocol import HandoffClient


def common_record(args, device, dtype):
    return {
        "role": args.role,
        "environment": environment_record(args.model, args.revision, device, dtype),
        "config": {
            "prompt_length": args.prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "block_size": args.block_size,
            "token_budget": args.token_budget,
            "worker_warmup": args.worker_warmup,
            "request_id": args.request_id,
            "attempt_id": args.attempt_id,
        },
    }


def run_producer(args, model, token_ids, device, dtype):
    client = HandoffClient(port=args.port, timeout=args.timeout)
    namespace = build_namespace(
        args.model, args.revision, model, dtype, args.block_size
    )
    cache, prefill = prefill_request(
        model, token_ids, device, args.block_size, args.token_budget,
        args.request_id, args.max_new_tokens,
    )
    manifest, payload, export_ms = export_handoff(
        cache, args.request_id, token_ids, prefill["first_token"],
        prefill["first_logits_sha256"], namespace, args.attempt_id,
    )
    start = time.perf_counter()
    client.publish(manifest, payload)
    publish_ms = (time.perf_counter() - start) * 1000
    deadline = time.monotonic() + args.timeout
    state = None
    reason = None
    while time.monotonic() < deadline:
        response = client.status(args.request_id, args.attempt_id)
        state, reason = response["state"], response.get("reason")
        if state in ("acknowledged", "fallback"):
            break
        time.sleep(0.01)
    else:
        reason = "等待 Decode ACK 超时"
        client.abort(args.request_id, args.attempt_id, reason)
        state = "fallback"
    before_release = cache.snapshot()
    cache.release(args.request_id)
    service_released = False
    if state in ("acknowledged", "fallback"):
        service_released = client.release(
            args.request_id, args.attempt_id
        )["released"]
    return {
        "first_token": prefill["first_token"],
        "first_logits_sha256": prefill["first_logits_sha256"],
        "manifest": manifest,
        "metrics": {
            **prefill, "export_d2h_ms": export_ms,
            "publish_ms": publish_ms, "ack_state": state,
            "ack_reason": reason,
            "cache_snapshot_before_release": before_release,
            "final_cache_snapshot": cache.snapshot(),
            "service_released": service_released,
        },
    }


def run_decoder(args, model, token_ids, device, dtype):
    client = HandoffClient(port=args.port, timeout=args.timeout)
    expected_namespace = build_namespace(
        args.model, args.revision, model, dtype, args.block_size
    )
    if args.inject_failure == "namespace":
        expected_namespace = dict(expected_namespace)
        expected_namespace["format_version"] = 999
    start = time.perf_counter()
    response, payload = client.receive(args.request_id, args.attempt_id)
    receive_ms = (time.perf_counter() - start) * 1000
    manifest = response["manifest"]
    if args.inject_failure == "checksum" and payload:
        changed = bytearray(payload)
        changed[len(changed) // 2] ^= 1
        payload = bytes(changed)
    ack_ms = 0.0
    acknowledged = False

    def acknowledge():
        nonlocal ack_ms, acknowledged
        start_ack = time.perf_counter()
        client.acknowledge(args.request_id, args.attempt_id, True)
        ack_ms = (time.perf_counter() - start_ack) * 1000
        acknowledged = True

    try:
        result = decode_from_payload(
            model, manifest, payload, device, expected_namespace,
            args.max_new_tokens, args.block_size, args.request_id,
            on_imported=acknowledge,
        )
        fallback, error = False, None
    except (KeyError, RuntimeError, ValueError) as caught:
        error = "%s: %s" % (type(caught).__name__, caught)
        if not acknowledged:
            start_ack = time.perf_counter()
            client.acknowledge(
                args.request_id, args.attempt_id, False, reason=error
            )
            ack_ms = (time.perf_counter() - start_ack) * 1000
        result = run_monolithic(
            model, token_ids, args.max_new_tokens, device,
            args.block_size, args.token_budget, args.request_id,
        )
        fallback = True
    result["metrics"].update({
        "receive_ms": receive_ms, "ack_ms": ack_ms,
        "fallback": fallback, "handoff_error": error,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument(
        "--role", choices=("monolithic", "prefill", "decode"), required=True
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--prompt-length", type=int, default=513)
    parser.add_argument("--token-ids-file")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--request-id", default="request-0")
    parser.add_argument("--attempt-id", default="attempt-0")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--worker-warmup", type=int, default=1,
        help="模型加载后、正式计时前在同一 CUDA 进程内执行的 warm-up 次数",
    )
    parser.add_argument(
        "--inject-failure", choices=("none", "checksum", "namespace"),
        default="none",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.role != "monolithic" and args.port < 1:
        parser.error("prefill/decode Worker 必须提供 --port")
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    _, config, model = load_model(args.model, args.revision, device, dtype)
    if args.token_ids_file:
        with open(args.token_ids_file, "r", encoding="utf-8") as file:
            token_ids = [int(value) for value in json.load(file)]
        args.prompt_length = len(token_ids)
    else:
        token_ids = synthetic_tokens(args.prompt_length, config.vocab_size)
    for warmup_index in range(args.worker_warmup):
        run_monolithic(
            model, token_ids, args.max_new_tokens, device,
            args.block_size, args.token_budget,
            request_id="warmup-%d" % warmup_index,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    record = common_record(args, device, dtype)
    if args.role == "monolithic":
        result = run_monolithic(
            model, token_ids, args.max_new_tokens, device,
            args.block_size, args.token_budget, args.request_id,
        )
    elif args.role == "prefill":
        result = run_producer(args, model, token_ids, device, dtype)
    else:
        result = run_decoder(args, model, token_ids, device, dtype)
    record.update(result)
    if device.type == "cuda":
        record["environment"]["peak_gpu_allocated_bytes"] = (
            torch.cuda.max_memory_allocated(device)
        )
        record["environment"]["peak_gpu_reserved_bytes"] = (
            torch.cuda.max_memory_reserved(device)
        )
    write_json(args.output, record)
    print("role=%s pid=%d done" % (args.role, os.getpid()))


if __name__ == "__main__":
    main()
