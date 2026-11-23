#!/usr/bin/env python3
"""Quick length check of sft_mini_512.jsonl @ 9b06bee2 (one-off decision support)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset_variant_survey import MIRROR, range_lines, tokenize_candidate
from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer


def main() -> None:
    tokenizer = Qwen3Tokenizer(resolve_model_directory("Qwen/Qwen3-0.6B-Base"))
    url = (
        f"{MIRROR}/datasets/jingyaogong/minimind_dataset/resolve/"
        "9b06bee29c0d5685647ac11b0dcea9f572c5064a/sft_mini_512.jsonl"
    )
    lines = range_lines(url, seed=20260912)
    print("lines:", len(lines), flush=True)
    report = tokenize_candidate(tokenizer, lines, seed=20260913)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
