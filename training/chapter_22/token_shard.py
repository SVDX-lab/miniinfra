"""Chapter 21 core: the offline binary token shard and its online counterpart.

Shard layout (all integers little-endian):

    header:
        magic            8 bytes   b"MISFTSKD"
        format_version   u32
        model_revision   40 bytes  (sha1 hex, ascii; tokenizer uses the same)
        dataset_revision 40 bytes  (sha1 hex, ascii)
        max_length       u32
        num_records      u32
    records (concatenated, variable length):
        source_line      u64
        length           u32       (non-padding token count of the record)
        input_ids        length * u32
        labels           length * i32 (-100 padding of the loss mask)
        checksum         u64       (FNV-1a over source_line..labels bytes)
    index:
        num_records * u64  record payload offsets

The reader validates the header against the expected revisions, verifies every
record checksum on read, and supports O(1) random access through the index.
Any mismatch raises instead of falling back silently.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from assistant_loss import EncodedConversation


MAGIC = b"MISFTSKD"
FORMAT_VERSION = 1
HEADER_STRUCT = struct.Struct("<8sI40s40sII")
RECORD_PREFIX_STRUCT = struct.Struct("<QI")
CHECKSUM_STRUCT = struct.Struct("<Q")


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit checksum: fast, non-cryptographic corruption detector."""

    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


@dataclass(frozen=True)
class ShardMeta:
    model_revision: str
    dataset_revision: str
    max_length: int
    num_records: int


class TokenShardWriter:
    """Streams encoded records to disk and appends the index at the end."""

    def __init__(
        self,
        path: Path,
        model_revision: str,
        dataset_revision: str,
        max_length: int,
    ) -> None:
        self.path = Path(path)
        self.meta = ShardMeta(model_revision, dataset_revision, max_length, 0)
        self._offsets: list[int] = []
        self._handle = open(self.path, "wb")
        self._handle.write(
            HEADER_STRUCT.pack(
                MAGIC,
                FORMAT_VERSION,
                model_revision.encode("ascii"),
                dataset_revision.encode("ascii"),
                max_length,
                0,  # patched on close
            )
        )
        self._payload_start = self._handle.tell()

    def add(self, sample: EncodedConversation) -> None:
        length = sample.non_padding_tokens
        if length < 2:
            raise ValueError("a sample needs at least two non-padding tokens")
        if length > self.meta.max_length:
            raise ValueError(f"sample length {length} exceeds shard max_length {self.meta.max_length}")
        record = self._handle.tell()
        input_ids = sample.input_ids[0, :length].to(torch.uint32).numpy().tobytes()
        labels = sample.labels[0, :length].to(torch.int32).numpy().tobytes()
        body = (
            RECORD_PREFIX_STRUCT.pack(int(sample.source_line), length)
            + input_ids
            + labels
        )
        self._handle.write(body)
        self._handle.write(CHECKSUM_STRUCT.pack(fnv1a64(body)))
        self._offsets.append(record)
        self.meta = ShardMeta(
            self.meta.model_revision,
            self.meta.dataset_revision,
            self.meta.max_length,
            self.meta.num_records + 1,
        )

    def close(self) -> None:
        index_offset = self._handle.tell()
        for offset in self._offsets:
            self._handle.write(struct.pack("<Q", offset))
        # 尾部自描述指针：索引区起始位置，供读取端校验索引完整性。
        self._handle.write(struct.pack("<Q", index_offset))
        # 头部的 num_records 位于固定偏移 96（8+4+40+40+4），在此补写真实值。
        self._handle.seek(96)
        self._handle.write(struct.pack("<I", self.meta.num_records))
        self._handle.close()


def build_shard(
    path: Path,
    samples: list[EncodedConversation],
    model_revision: str,
    dataset_revision: str,
    max_length: int,
) -> dict[str, float]:
    """Encode-free helper: write ``samples`` (already encoded) into a shard."""

    import time

    start = time.perf_counter()
    writer = TokenShardWriter(path, model_revision, dataset_revision, max_length)
    for sample in samples:
        writer.add(sample)
    writer.close()
    elapsed = time.perf_counter() - start
    return {"write_seconds": elapsed, "file_bytes": Path(path).stat().st_size}


class TokenShardReader:
    """Random-access reader with header validation and checksum verification."""

    def __init__(
        self,
        path: Path,
        expect_model_revision: str,
        expect_dataset_revision: str,
    ) -> None:
        self.path = Path(path)
        self._handle = open(self.path, "rb")
        header = self._handle.read(HEADER_STRUCT.size)
        if len(header) != HEADER_STRUCT.size:
            raise ValueError(f"{self.path}: truncated header")
        magic, version, model_rev, dataset_rev, max_length, num_records = HEADER_STRUCT.unpack(header)
        if magic != MAGIC:
            raise ValueError(f"{self.path}: bad magic {magic!r}")
        if version != FORMAT_VERSION:
            raise ValueError(f"{self.path}: unsupported format version {version}")
        if model_rev.decode("ascii").rstrip("\x00") != expect_model_revision:
            raise ValueError(f"{self.path}: model/tokenizer revision mismatch")
        if dataset_rev.decode("ascii").rstrip("\x00") != expect_dataset_revision:
            raise ValueError(f"{self.path}: dataset revision mismatch")
        self.meta = ShardMeta(
            model_rev.decode("ascii").rstrip("\x00"),
            dataset_rev.decode("ascii").rstrip("\x00"),
            max_length,
            num_records,
        )
        self._handle.seek(0, 2)
        file_size = self._handle.tell()
        index_region = file_size - (num_records + 1) * 8
        if index_region <= HEADER_STRUCT.size:
            raise ValueError(f"{self.path}: index region missing or truncated")
        # 尾部 8 字节是索引区起始位置的自描述指针。
        self._handle.seek(file_size - 8)
        stored_offset = struct.unpack("<Q", self._handle.read(8))[0]
        if stored_offset != index_region:
            raise ValueError(f"{self.path}: index pointer mismatch ({stored_offset} != {index_region})")
        self._handle.seek(stored_offset)
        offsets = []
        for _ in range(num_records):
            offsets.append(struct.unpack("<Q", self._handle.read(8))[0])
        self._offsets = offsets
        if stored_offset < HEADER_STRUCT.size or stored_offset >= file_size:
            raise ValueError(f"{self.path}: index offset out of range")

    def __len__(self) -> int:
        return self.meta.num_records

    def read_record(self, index: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Return (source_line, input_ids[1, L] int64, labels[1, L] int64)."""

        if not 0 <= index < self.meta.num_records:
            raise IndexError(f"record index {index} out of range")
        self._handle.seek(self._offsets[index])
        prefix = self._handle.read(RECORD_PREFIX_STRUCT.size)
        if len(prefix) != RECORD_PREFIX_STRUCT.size:
            raise ValueError(f"{self.path}: truncated record {index}")
        source_line, length = RECORD_PREFIX_STRUCT.unpack(prefix)
        payload = self._handle.read(length * 8)
        checksum_bytes = self._handle.read(8)
        if len(payload) != length * 8 or len(checksum_bytes) != 8:
            raise ValueError(f"{self.path}: truncated payload of record {index}")
        (stored_checksum,) = CHECKSUM_STRUCT.unpack(checksum_bytes)
        body = prefix + payload
        if fnv1a64(body) != stored_checksum:
            raise ValueError(f"{self.path}: checksum mismatch on record {index}")
        input_ids = torch.frombuffer(bytearray(payload[: length * 4]), dtype=torch.uint32).to(torch.int64)
        labels = torch.frombuffer(bytearray(payload[length * 4 :]), dtype=torch.int32).to(torch.int64)
        return source_line, input_ids.unsqueeze(0), labels.unsqueeze(0)

    def close(self) -> None:
        self._handle.close()


def shard_manifest_entry(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": Path(path).stat().st_size}


def write_jsonl_copy(records: list[dict], path: Path) -> None:
    """Helper for tests: persist records in the online baseline's format."""

    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
