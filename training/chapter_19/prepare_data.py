#!/usr/bin/env python3
"""Download the pinned dataset and build Chapter 19's fixed 512-record shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from sft_core import (
    DATASET_FILE,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SHA256,
    DATASET_SIZE,
    conversation_filter_reason,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/chapter19_train.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data-manifest.local.json"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    downloaded = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=DATASET_FILE,
            revision=DATASET_REVISION,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
    )
    actual_size = downloaded.stat().st_size
    actual_sha256 = sha256(downloaded)
    if actual_size != DATASET_SIZE or actual_sha256 != DATASET_SHA256:
        raise RuntimeError(f"dataset verification failed: size={actual_size}, sha256={actual_sha256}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_lines: list[int] = []
    filtered_reasons: Counter[str] = Counter()
    scanned_lines = 0
    non_empty_records = 0
    with downloaded.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            scanned_lines = line_number
            if not line.strip():
                continue
            non_empty_records += 1
            record = json.loads(line)
            reason = conversation_filter_reason(record)
            if reason is not None:
                filtered_reasons[reason] += 1
                continue
            target.write(
                json.dumps(
                    {"source_line": line_number, "conversations": record["conversations"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            selected_lines.append(line_number)
            if len(selected_lines) == args.samples:
                break
    if len(selected_lines) != args.samples:
        raise RuntimeError(f"only found {len(selected_lines)} eligible records")

    manifest = {
        "repo": DATASET_REPO,
        "revision": DATASET_REVISION,
        "file": DATASET_FILE,
        "size": DATASET_SIZE,
        "sha256": DATASET_SHA256,
        "selection": "first N plain system/user/assistant records of sft_mini_512@9b06bee2 (data-pipeline block corpus)",
        "scanned_lines": scanned_lines,
        "non_empty_records": non_empty_records,
        "selected_records": len(selected_lines),
        "filtered_records": sum(filtered_reasons.values()),
        "filtered_by_reason": dict(sorted(filtered_reasons.items())),
        "source_lines": selected_lines,
        "loss_definition": "Assistant-only labels inherited unchanged from Chapter 17",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(selected_lines)} records and verified {DATASET_FILE}")


if __name__ == "__main__":
    main()
