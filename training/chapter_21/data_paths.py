"""Chapter 21 data paths: online JSONL tokenization versus the offline shard.

Both paths expose the same tiny interface - ``__len__`` and
``materialize(index) -> (source_line, input_ids[1, L], labels[1, L])`` - and
share one collate function, so a training step cannot tell them apart except by
timing.  The online path renders the chat template and runs BPE on every
materialization; the shard path seeks, reads and checksum-verifies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from assistant_loss import IGNORE_INDEX, encode_conversation
from mini_batch import PaddedBatch, trim_sample
from qwen3_tokenizer import Qwen3Tokenizer
from sft_core import load_jsonl_records
from token_shard import TokenShardReader


class OnlineCorpus:
    """Baseline: JSONL records materialized through render + BPE on demand."""

    def __init__(self, jsonl_path: Path, tokenizer: Qwen3Tokenizer, max_length: int) -> None:
        self.records = load_jsonl_records(Path(jsonl_path))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def materialize(self, index: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        sample = encode_conversation(
            self.tokenizer, self.records[index], self.max_length, "assistant_only"
        )
        sample = trim_sample(sample)
        return sample.source_line, sample.input_ids, sample.labels


class ShardCorpus:
    """Variant: materialize by seeking into the binary token shard."""

    def __init__(self, shard_path: Path, model_revision: str, dataset_revision: str) -> None:
        self.reader = TokenShardReader(
            Path(shard_path), expect_model_revision=model_revision, expect_dataset_revision=dataset_revision
        )

    def __len__(self) -> int:
        return len(self.reader)

    def materialize(self, index: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        return self.reader.read_record(index)


def pad_pairs(
    pairs: Sequence[tuple[int, torch.Tensor, torch.Tensor]],
    pad_token_id: int,
) -> PaddedBatch:
    """Right-pad materialized (source_line, ids, labels) tuples into one batch.

    Same defaults as ``mini_batch.pad_samples``: pad tokens, zero attention and
    ``-100`` labels first, then copy each real prefix, so both data paths
    produce bitwise-identical batches for identical members.
    """

    if not pairs:
        raise ValueError("pairs must not be empty")
    lengths = [ids.shape[1] for _, ids, _ in pairs]
    maximum = max(lengths)
    batch_size = len(pairs)
    input_ids = torch.full((batch_size, maximum), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, maximum), dtype=torch.long)
    labels = torch.full((batch_size, maximum), IGNORE_INDEX, dtype=torch.long)
    supervised: list[int] = []
    for row, (source_line, ids, pair_labels) in enumerate(pairs):
        length = ids.shape[1]
        input_ids[row, :length] = ids[0]
        attention_mask[row, :length] = 1
        labels[row, :length] = pair_labels[0]
        supervised.append(int((pair_labels[0, 1:] != IGNORE_INDEX).sum().item()))
    return PaddedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        source_lines=tuple(int(pair[0]) for pair in pairs),
        lengths=tuple(lengths),
        supervised_tokens_per_sample=tuple(supervised),
    )
