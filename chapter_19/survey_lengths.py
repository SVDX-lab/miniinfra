#!/usr/bin/env python3
"""Measure the natural (cap-1024) token lengths of the fixed Chapter 19 shard.

The formal benchmark truncates to 512, which leaves the corpus nearly uniform
(96.9% of samples hit the cap).  This survey records each sample's length under
a 1024 cap so the padding projection on the natural length spread can be
reproduced from real data.  The output feeds the projection table in the
README; it contains no training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen3_model import resolve_model_directory
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import load_jsonl_records
from training import MODEL_ID, MODEL_REVISION, encode_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chapter19_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("corpus-lengths-cap1024.json"))
    args = parser.parse_args()

    tokenizer = Qwen3Tokenizer(resolve_model_directory(MODEL_ID, MODEL_REVISION))
    records = load_jsonl_records(args.data)
    samples = encode_samples(tokenizer, records, args.max_length)
    lengths = sorted(sample.non_padding_tokens for sample in samples)
    count = len(lengths)
    payload = {
        "schema_version": "chapter19.length-survey.v1",
        "data": str(args.data),
        "max_length": args.max_length,
        "sample_count": count,
        "lengths_sorted": lengths,
        "summary": {
            "min": lengths[0],
            "p25": lengths[count // 4],
            "p50": lengths[count // 2],
            "p75": lengths[3 * count // 4],
            "max": lengths[-1],
            "fraction_at_cap": sum(1 for value in lengths if value >= args.max_length) / count,
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {count} lengths to {args.output}; p50={payload['summary']['p50']} at_cap={payload['summary']['fraction_at_cap']:.3f}")


if __name__ == "__main__":
    main()
