"""Chapter 18 data, loss, and deterministic experiment primitives."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from assistant_loss import EncodedConversation, IGNORE_INDEX


DATASET_REPO = "jingyaogong/minimind_dataset"
DATASET_REVISION = "312afb4f76391145c6902f765bb51691c09a12f5"
DATASET_FILE = "sft_t2t_mini.jsonl"
DATASET_SIZE = 1_739_201_170
DATASET_SHA256 = "abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518"
ALLOWED_PLAIN_ROLES = {"system", "user", "assistant"}


def conversation_filter_reason(record: dict[str, Any]) -> str | None:
    """Select the same plain-text training shard as Chapter 17."""

    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return "missing_or_empty_conversations"
    for message in conversations:
        if not isinstance(message, dict):
            return "message_is_not_an_object"
        if message.get("role") not in ALLOWED_PLAIN_ROLES:
            return "unsupported_role"
        if not isinstance(message.get("content"), str):
            return "content_is_not_text"
        if set(message) - {"role", "content"}:
            return "extra_message_fields"
    if not any(message["role"] == "assistant" for message in conversations):
        return "no_assistant_message"
    return None


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("source_line", line_number)
            if not isinstance(record.get("conversations"), list):
                raise ValueError(f"source line {record['source_line']} has no conversation list")
            records.append(record)
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def select_record(records: list[dict[str, Any]], cursor: int) -> dict[str, Any]:
    if cursor < 0:
        raise ValueError("cursor must be non-negative")
    return records[cursor % len(records)]


def shifted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [batch, sequence, vocab] and labels [batch, sequence]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels batch/sequence shapes must match")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if not bool((shift_labels != IGNORE_INDEX).any()):
        raise ValueError("no supervised next-token targets")
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def tensor_checksum(tensors: Iterable[torch.Tensor]) -> float:
    checksum = torch.zeros((), dtype=torch.float64)
    for tensor in tensors:
        checksum += tensor.detach().double().sum().cpu()
    return float(checksum.item())
