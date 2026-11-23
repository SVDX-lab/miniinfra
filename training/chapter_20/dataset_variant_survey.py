#!/usr/bin/env python3
"""One-off survey: length distributions of dataset variants under the Qwen3 tokenizer.

Candidates:
  A. sft_t2t_mini.jsonl @ 312afb4f (current main line), uniform random sample
     over the WHOLE file instead of the first-512 prefix;
  B. sft_t2t.jsonl   @ 312afb4f (14GB full version), range sample;
  C. sft_512.jsonl   @ 9b06bee2 (2025-02 revision, file since deleted), range sample;
  D. sft_1024.jsonl  @ 9b06bee2, range sample;
  E. sft_2048.jsonl  @ 9b06bee2, range sample.

Lengths are natural Qwen3 token counts of the rendered chat template, no truncation.
This script writes a JSON report; it changes nothing in the chapter code.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from assistant_loss import render_qwen3_segments
from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import DATASET_FILE, DATASET_REPO, conversation_filter_reason

MIRROR = "https://hf-mirror.com"
OLD_REV = "9b06bee29c0d5685647ac11b0dcea9f572c5064a"
SAMPLES_PER_CANDIDATE = 300
WINDOW_BYTES = 1_500_000
WINDOWS = 40


def render_length(tokenizer, record):
    try:
        segments = render_qwen3_segments(record.get("conversations", []))
    except ValueError:
        return None
    text = "".join(segment.text for segment in segments)
    return len(tokenizer.encode(text))


def summarize(lengths):
    ordered = sorted(lengths)
    n = len(ordered)

    def pct(fraction):
        return ordered[min(n - 1, int(fraction * n))]

    return {
        "n": n,
        "min": ordered[0],
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": ordered[-1],
        "frac_lt_256": round(sum(1 for v in ordered if v < 256) / n, 4),
        "frac_lt_512": round(sum(1 for v in ordered if v < 512) / n, 4),
        "frac_ge_512": round(sum(1 for v in ordered if v >= 512) / n, 4),
        "frac_at_1024cap": round(sum(1 for v in ordered if v >= 1024) / n, 4),
    }


def tokenize_candidate(tokenizer, lines, seed):
    rng = random.Random(seed)
    if len(lines) > SAMPLES_PER_CANDIDATE:
        lines = rng.sample(lines, SAMPLES_PER_CANDIDATE)
    passing, rejected, lengths = 0, 0, []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        reason = conversation_filter_reason(record)
        if reason is not None:
            rejected += 1
            continue
        length = render_length(tokenizer, record)
        if length is None:
            rejected += 1
            continue
        passing += 1
        lengths.append(length)
    return {
        "raw_lines_seen": len(lines),
        "plain_filter_passed": passing,
        "plain_filter_rejected": rejected,
        "lengths": summarize(lengths) if lengths else None,
    }


def remote_size(url):
    result = subprocess.run(
        ["curl", "-sIL", url], capture_output=True, text=True, check=True
    )
    size = None
    for block in result.stdout.split("\r\n\r\n"):
        for line in block.splitlines():
            if line.lower().startswith("content-length:"):
                size = int(line.split(":", 1)[1].strip())
    return size


def range_lines(url, seed):
    size = remote_size(url)
    if not size or size <= WINDOW_BYTES:
        raise RuntimeError(f"cannot range-sample {url} (size={size})")
    rng = random.Random(seed)
    lines = []
    for _ in range(WINDOWS):
        start = rng.randint(0, size - WINDOW_BYTES - 1)
        result = subprocess.run(
            ["curl", "-sL", "-r", f"{start}-{start + WINDOW_BYTES - 1}", url],
            capture_output=True, check=True,
        )
        data = result.stdout
        newline = data.find(b"\n")
        if newline == -1:
            continue
        chunk = data[newline + 1 :].decode("utf-8", errors="ignore")
        lines.extend(line for line in chunk.split("\n") if line.strip())
    return lines


def reservoir_lines(path, count, seed):
    rng = random.Random(seed)
    held: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            if len(held) < count:
                held.append(line)
            else:
                draw = rng.randint(0, index)
                if draw < count:
                    held[draw] = line
    return held


def main() -> None:
    tokenizer = Qwen3Tokenizer(resolve_model_directory("Qwen/Qwen3-0.6B-Base"))
    report = {}

    local_mini = Path(
        __import__("huggingface_hub").hf_hub_download(
            repo_id=DATASET_REPO, repo_type="dataset",
            filename=DATASET_FILE, revision="312afb4f76391145c6902f765bb51691c09a12f5",
        )
    )
    candidates = [
        ("A_t2t_mini_uniform_random", "reservoir", str(local_mini), None),
        ("B_t2t_full_range", "range", f"{MIRROR}/datasets/{DATASET_REPO}/resolve/312afb4f76391145c6902f765bb51691c09a12f5/sft_t2t.jsonl", None),
        ("C_sft_512_old_rev", "range", f"{MIRROR}/datasets/{DATASET_REPO}/resolve/{OLD_REV}/sft_512.jsonl", None),
        ("D_sft_1024_old_rev", "range", f"{MIRROR}/datasets/{DATASET_REPO}/resolve/{OLD_REV}/sft_1024.jsonl", None),
        ("E_sft_2048_old_rev", "range", f"{MIRROR}/datasets/{DATASET_REPO}/resolve/{OLD_REV}/sft_2048.jsonl", None),
    ]
    for name, mode, source, _ in candidates:
        print(f"surveying {name} ...", flush=True)
        try:
            if mode == "reservoir":
                lines = reservoir_lines(source, SAMPLES_PER_CANDIDATE, seed=20260912)
            else:
                lines = range_lines(source, seed=20260912)
            report[name] = tokenize_candidate(tokenizer, lines, seed=20260913)
            print(name, json.dumps(report[name]["lengths"], ensure_ascii=False), flush=True)
        except Exception as error:  # noqa: BLE001 - survey must keep going
            report[name] = {"error": repr(error)}
            print(name, "ERROR", repr(error), flush=True)

    Path("dataset-variant-survey.local.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("saved dataset-variant-survey.local.json")


if __name__ == "__main__":
    main()
