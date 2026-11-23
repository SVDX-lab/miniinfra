"""真实权重 monolithic、跨进程 Handoff 与故障回退验证。"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from experiment_utils import add_model_arguments, handoff_service, write_json


def worker_base(args, role, output, port=0, inject_failure="none"):
    worker = Path(__file__).with_name("worker.py")
    command = [
        sys.executable, str(worker), "--role", role,
        "--model", args.model, "--revision", args.revision,
        "--device", args.device, "--dtype", args.dtype,
        "--prompt-length", str(args.prompt_length),
        "--max-new-tokens", str(args.max_new_tokens),
        "--block-size", str(args.block_size),
        "--token-budget", str(args.token_budget),
        "--request-id", args.request_id, "--attempt-id", args.attempt_id,
        "--timeout", str(args.timeout), "--port", str(port),
        "--worker-warmup", str(args.worker_warmup),
        "--inject-failure", inject_failure, "--output", str(output),
    ]
    if getattr(args, "token_ids_file", None):
        command.extend(("--token-ids-file", args.token_ids_file))
    return command


def checked_wait(process, name):
    stdout, stderr = process.communicate(timeout=300)
    if process.returncode:
        raise RuntimeError(
            "%s Worker 失败(code=%d)\nstdout:\n%s\nstderr:\n%s"
            % (name, process.returncode, stdout, stderr)
        )
    return stdout.strip()


def run_split_case(args, directory, client, port, inject_failure="none"):
    suffix = inject_failure
    producer_path = directory / ("prefill-%s.json" % suffix)
    decoder_path = directory / ("decode-%s.json" % suffix)
    producer = subprocess.Popen(
        worker_base(args, "prefill", producer_path, port),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if producer.poll() is not None:
            checked_wait(producer, "Prefill")
            raise RuntimeError("Prefill Worker 在发布后等待 ACK 前意外退出")
        status = client.status(args.request_id, args.attempt_id)
        if status["state"] == "kv_ready":
            break
        time.sleep(0.02)
    else:
        producer.terminate()
        raise RuntimeError("等待 Prefill Worker 发布 KV 超时")
    decoder = subprocess.Popen(
        worker_base(
            args, "decode", decoder_path, port,
            inject_failure=inject_failure,
        ),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    logs = {
        "decode": checked_wait(decoder, "Decode"),
        "prefill": checked_wait(producer, "Prefill"),
    }
    records = {
        "prefill": json.loads(producer_path.read_text(encoding="utf-8")),
        "decode": json.loads(decoder_path.read_text(encoding="utf-8")),
    }
    pmetrics = records["prefill"]["metrics"]
    dmetrics = records["decode"]["metrics"]
    if not dmetrics["fallback"]:
        ttft = sum((
            pmetrics["prefill_ready_ms"], pmetrics["export_d2h_ms"],
            pmetrics["publish_ms"], dmetrics["receive_ms"],
            dmetrics["import_h2d_ms"], dmetrics["ack_ms"],
        ))
        end_to_end = ttft + sum(dmetrics["decode_step_wall_ms"])
    else:
        ttft = None
        end_to_end = None
    return {
        "inject_failure": inject_failure, "workers": records,
        "worker_logs": logs,
        "derived_metrics": {
            "ack_gated_ttft_ms": ttft,
            "end_to_end_ms": end_to_end,
            "handoff_payload_bytes": records["prefill"]["manifest"]["payload_bytes"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="第 15 期跨进程正确性验证")
    add_model_arguments(parser)
    parser.add_argument("--prompt-length", type=int, default=513)
    parser.add_argument("--token-ids-file")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--request-id", default="request-0")
    parser.add_argument("--attempt-id", default="attempt-0")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--worker-warmup", type=int, default=1)
    parser.add_argument(
        "--failure", choices=("none", "checksum", "namespace"), default="none"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="chapter15-") as temp:
        directory = Path(temp)
        baseline_path = directory / "monolithic.json"
        baseline_run = subprocess.run(
            worker_base(args, "monolithic", baseline_path),
            check=True, text=True, capture_output=True,
        )
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        with handoff_service() as (client, port):
            split = run_split_case(
                args, directory, client, port, inject_failure=args.failure
            )
            server_stats = client.stats()
    producer = split["workers"]["prefill"]
    decoder = split["workers"]["decode"]
    checks = {
        "three_different_processes": len({
            baseline["environment"]["pid"],
            producer["environment"]["pid"], decoder["environment"]["pid"],
        }) == 3,
        "first_logits_equal": (
            baseline["first_logits_sha256"]
            == producer["first_logits_sha256"]
        ),
        "output_tokens_equal": (
            baseline["new_token_ids"] == decoder["new_token_ids"]
        ),
        "fallback_matches_injection": (
            decoder["metrics"]["fallback"] == (args.failure != "none")
        ),
        "producer_released_local_blocks": (
            producer["metrics"]["final_cache_snapshot"]["used_blocks"] == 0
        ),
        "decoder_released_local_blocks": (
            decoder["metrics"]["final_cache_snapshot"]["used_blocks"] == 0
        ),
        "service_released_payload": server_stats["entries"] == 0,
    }
    result = {
        "config": vars(args), "checks": checks,
        "baseline": baseline, "split": split,
        "server_stats": server_stats,
        "baseline_log": baseline_run.stdout.strip(),
    }
    write_json(args.output, result)
    print(checks)
    if not all(checks.values()):
        raise SystemExit("跨进程正确性验证失败")


if __name__ == "__main__":
    main()
