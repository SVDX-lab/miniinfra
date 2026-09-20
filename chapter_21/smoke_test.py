#!/usr/bin/env python3
"""CPU-only Chapter 21 checks: shard roundtrip, corruption detection, collate parity."""

from __future__ import annotations

import struct
from pathlib import Path

import torch

from assistant_loss import EncodedConversation, IGNORE_INDEX
from data_paths import pad_pairs
from mini_batch import pad_samples
from token_shard import (
    MAGIC,
    TokenShardReader,
    TokenShardWriter,
    fnv1a64,
)


MODEL_REV = "a" * 40
DATASET_REV = "b" * 40


def synthetic_sample(ids: list[int], supervised_from: int, source_line: int) -> EncodedConversation:
    labels = [IGNORE_INDEX if index < supervised_from else token for index, token in enumerate(ids)]
    return EncodedConversation(
        input_ids=torch.tensor([ids]),
        attention_mask=torch.tensor([[1] * len(ids)]),
        labels=torch.tensor([labels]),
        assistant_loss_mask=torch.tensor([[label != IGNORE_INDEX for label in labels]]),
        source_line=source_line,
        rendered_text="",
        token_spans=(),
        loss_mode="assistant_only",
    )


SAMPLES = [
    synthetic_sample(list(range(1, 12)), 4, 11),
    synthetic_sample(list(range(20, 32)), 6, 12),
    synthetic_sample(list(range(40, 45)), 2, 13),
    synthetic_sample(list(range(50, 57)), 3, 14),
]


def build(tmp: Path) -> Path:
    shard = tmp / "roundtrip.bin"
    writer = TokenShardWriter(shard, MODEL_REV, DATASET_REV, max_length=32)
    for sample in SAMPLES:
        writer.add(sample)
    writer.close()
    return shard


def check_roundtrip(tmp: Path) -> None:
    shard = build(tmp)
    reader = TokenShardReader(shard, MODEL_REV, DATASET_REV)
    assert len(reader) == len(SAMPLES)
    assert reader.meta.max_length == 32
    for index, sample in enumerate(SAMPLES):
        source_line, ids, labels = reader.read_record(index)
        length = sample.non_padding_tokens
        assert source_line == sample.source_line
        assert ids.shape == (1, length) and labels.shape == (1, length)
        assert torch.equal(ids, sample.input_ids[:, :length]), f"ids differ at {index}"
        assert torch.equal(labels, sample.labels[:, :length]), f"labels differ at {index}"
    # 随机访问与顺序访问逐位一致
    a = reader.read_record(3)
    b = reader.read_record(3)
    assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) and a[0] == b[0]
    reader.close()


def check_header_validation(tmp: Path) -> None:
    shard = build(tmp)
    try:
        TokenShardReader(shard, MODEL_REV, "c" * 40)
    except ValueError as error:
        assert "dataset revision mismatch" in str(error)
    else:
        raise AssertionError("wrong dataset revision must raise")
    bad_magic = tmp / "badmagic.bin"
    data = bytearray(shard.read_bytes())
    data[0:8] = b"XXXXXXXX"
    bad_magic.write_bytes(bytes(data))
    try:
        TokenShardReader(bad_magic, MODEL_REV, DATASET_REV)
    except ValueError as error:
        assert "bad magic" in str(error)
    else:
        raise AssertionError("bad magic must raise")


def check_corruption_detection(tmp: Path) -> None:
    shard = build(tmp)
    raw = shard.read_bytes()
    reader = TokenShardReader(shard, MODEL_REV, DATASET_REV)
    offsets = []
    handle = open(shard, "rb")
    handle.seek(0, 2)
    file_size = handle.tell()
    handle.seek(file_size - 8)
    (index_offset,) = struct.unpack("<Q", handle.read(8))
    handle.seek(index_offset)
    for _ in range(len(SAMPLES)):
        offsets.append(struct.unpack("<Q", handle.read(8))[0])
    handle.close()

    # 翻转 record 1 载荷中的一个字节 → 校验和必须报错
    flipped = bytearray(raw)
    flipped[offsets[1] + 12] ^= 0xFF
    corrupt = tmp / "corrupt.bin"
    corrupt.write_bytes(bytes(flipped))
    bad_reader = TokenShardReader(corrupt, MODEL_REV, DATASET_REV)
    bad_reader.read_record(0)
    try:
        bad_reader.read_record(1)
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("payload corruption must raise")

    # 截断索引区 → 头部/索引校验必须报错
    truncated = tmp / "truncated.bin"
    truncated.write_bytes(raw[: file_size - 20])
    try:
        TokenShardReader(truncated, MODEL_REV, DATASET_REV)
    except ValueError:
        pass
    else:
        raise AssertionError("truncated index must raise")


def check_checksum_function() -> None:
    assert fnv1a64(b"") == 0xCBF29CE484222325
    assert fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    data = bytes(range(256))
    assert fnv1a64(data) == fnv1a64(data)


def check_pad_pairs_matches_pad_samples(tmp: Path) -> None:
    shard = build(tmp)
    reader = TokenShardReader(shard, MODEL_REV, DATASET_REV)
    members = (0, 2)
    pairs = [reader.read_record(index) for index in members]
    from_pairs = pad_pairs(pairs, pad_token_id=0)
    from_samples = pad_samples([SAMPLES[index] for index in members], pad_token_id=0)
    assert torch.equal(from_pairs.input_ids, from_samples.input_ids)
    assert torch.equal(from_pairs.attention_mask, from_samples.attention_mask)
    assert torch.equal(from_pairs.labels, from_samples.labels)
    assert from_pairs.supervised_tokens == from_samples.supervised_tokens
    assert from_pairs.source_lines == from_samples.source_lines
    try:
        pad_pairs([], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("empty pairs must raise")


def main() -> None:
    tmp = Path("data/smoke_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for stale in tmp.glob("*.bin"):
        stale.unlink()
    check_checksum_function()
    check_roundtrip(tmp)
    check_header_validation(tmp)
    check_corruption_detection(tmp)
    check_pad_pairs_matches_pad_samples(tmp)
    print("chapter_21 smoke tests passed")


if __name__ == "__main__":
    main()
