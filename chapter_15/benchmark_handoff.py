"""扫描 Prompt 长度，比较 monolithic 与同步跨进程 KV Handoff。"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from experiment_utils import add_model_arguments, write_json


def median(values):
    return statistics.median(float(value) for value in values)


def main():
    parser = argparse.ArgumentParser(description="第 15 期 KV Handoff 基准")
    add_model_arguments(parser)
    parser.add_argument("--prompt-lengths", default="256,512,1024,2048")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument(
        "--warmup", type=int, default=0,
        help="额外丢弃的整组进程样本；每个 Worker 内部另有固定 warm-up",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker-warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    lengths = [int(value) for value in args.prompt_lengths.split(",")]
    validator = Path(__file__).with_name("validate_disaggregated.py")
    records = []
    with tempfile.TemporaryDirectory(prefix="chapter15-bench-") as temp:
        temp = Path(temp)
        for length in lengths:
            samples = []
            total = args.warmup + args.repeats
            for repeat in range(total):
                output = temp / ("%d-%d.json" % (length, repeat))
                command = [
                    sys.executable, str(validator),
                    "--model", args.model, "--revision", args.revision,
                    "--device", args.device, "--dtype", args.dtype,
                    "--prompt-length", str(length),
                    "--max-new-tokens", str(args.max_new_tokens),
                    "--block-size", str(args.block_size),
                    "--token-budget", str(args.token_budget),
                    "--timeout", str(args.timeout),
                    "--worker-warmup", str(args.worker_warmup),
                    "--attempt-id", "attempt-%d-%d" % (length, repeat),
                    "--output", str(output),
                ]
                subprocess.run(command, check=True, capture_output=True, text=True)
                record = json.loads(output.read_text(encoding="utf-8"))
                record["config"]["output"] = "<temporary-result-file>"
                if repeat >= args.warmup:
                    samples.append(record)
            baseline_metrics = [sample["baseline"]["metrics"] for sample in samples]
            split_metrics = [
                sample["split"]["derived_metrics"] for sample in samples
            ]
            producer_metrics = [
                sample["split"]["workers"]["prefill"]["metrics"]
                for sample in samples
            ]
            decoder_metrics = [
                sample["split"]["workers"]["decode"]["metrics"]
                for sample in samples
            ]
            records.append({
                "prompt_tokens": length,
                "payload_bytes": split_metrics[0]["handoff_payload_bytes"],
                "monolithic_ttft_ms_median": median(
                    item["ttft_ms"] for item in baseline_metrics
                ),
                "split_ack_gated_ttft_ms_median": median(
                    item["ack_gated_ttft_ms"] for item in split_metrics
                ),
                "monolithic_end_to_end_ms_median": median(
                    item["end_to_end_ms"] for item in baseline_metrics
                ),
                "split_end_to_end_ms_median": median(
                    item["end_to_end_ms"] for item in split_metrics
                ),
                "export_d2h_ms_median": median(
                    item["export_d2h_ms"] for item in producer_metrics
                ),
                "publish_ms_median": median(
                    item["publish_ms"] for item in producer_metrics
                ),
                "receive_ms_median": median(
                    item["receive_ms"] for item in decoder_metrics
                ),
                "import_h2d_ms_median": median(
                    item["import_h2d_ms"] for item in decoder_metrics
                ),
                "ack_ms_median": median(
                    item["ack_ms"] for item in decoder_metrics
                ),
                "samples": samples,
            })
    result = {"config": vars(args), "results": records}
    write_json(args.output, result)
    for record in records:
        print(
            "prompt=%d payload=%.1fMiB mono_ttft=%.2fms split_ttft=%.2fms"
            % (
                record["prompt_tokens"], record["payload_bytes"] / 2**20,
                record["monolithic_ttft_ms_median"],
                record["split_ack_gated_ttft_ms_median"],
            )
        )


if __name__ == "__main__":
    main()
