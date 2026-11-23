"""Producer 退出后由全新 Consumer 进程命中同一外部 KV。"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from experiment_utils import add_model_arguments, cache_service, write_json


def main():
    parser = argparse.ArgumentParser(description="第 14 期真实跨进程验证")
    add_model_arguments(parser)
    parser.add_argument("--prompt-length", type=int, default=513)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--external-chunk-size", type=int, default=256)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--capacity-mib", type=float, default=1024.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    worker = Path(__file__).with_name("cross_process_worker.py")
    with tempfile.TemporaryDirectory(prefix="chapter14-") as directory:
        directory = Path(directory)
        outputs = {
            "producer": directory / "producer.json",
            "consumer": directory / "consumer.json",
        }
        with cache_service(args.capacity_mib) as (client, port):
            logs = {}
            for role in ("producer", "consumer"):
                command = [
                    sys.executable, str(worker),
                    "--role", role,
                    "--port", str(port),
                    "--model", args.model,
                    "--revision", args.revision,
                    "--device", args.device,
                    "--dtype", args.dtype,
                    "--prompt-length", str(args.prompt_length),
                    "--max-new-tokens", str(args.max_new_tokens),
                    "--block-size", str(args.block_size),
                    "--external-chunk-size", str(args.external_chunk_size),
                    "--token-budget", str(args.token_budget),
                    "--output", str(outputs[role]),
                ]
                completed = subprocess.run(
                    command, check=True, text=True, capture_output=True
                )
                logs[role] = completed.stdout.strip()
            server_stats = client.stats()
        records = {
            role: json.loads(path.read_text(encoding="utf-8"))
            for role, path in outputs.items()
        }
    expected_hit = (
        (args.prompt_length - 1) // args.external_chunk_size
    ) * args.external_chunk_size
    checks = {
        "different_processes": (
            records["producer"]["environment"]["pid"]
            != records["consumer"]["environment"]["pid"]
        ),
        "producer_cold": records["producer"]["metrics"]["hit_tokens"] == 0,
        "consumer_hit_tokens": records["consumer"]["metrics"]["hit_tokens"],
        "consumer_expected_hit": (
            records["consumer"]["metrics"]["hit_tokens"] == expected_hit
        ),
        "output_tokens_equal": (
            records["producer"]["new_token_ids"]
            == records["consumer"]["new_token_ids"]
        ),
    }
    result = {
        "config": vars(args),
        "checks": checks,
        "workers": records,
        "worker_logs": logs,
        "server_stats": server_stats,
    }
    write_json(args.output, result)
    print(checks)
    if not all((
        checks["different_processes"], checks["producer_cold"],
        checks["consumer_expected_hit"], checks["output_tokens_equal"],
    )):
        raise SystemExit("跨进程验证失败")


if __name__ == "__main__":
    main()
