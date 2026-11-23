"""Chapter 16 data and loss primitives.

The chapter deliberately keeps a single-sample, fixed-length, full-conversation
objective. Later chapters change those choices one at a time.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100
ALLOWED_ROLES = {"system", "user", "assistant"}
DATASET_REPO = "jingyaogong/minimind_dataset"
DATASET_REVISION = "312afb4f76391145c6902f765bb51691c09a12f5"
DATASET_FILE = "sft_t2t_mini.jsonl"
DATASET_SIZE = 1_739_201_170
DATASET_SHA256 = "abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518"


@dataclass(frozen=True)
class EncodedConversation:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    source_line: int
    rendered_text: str

    @property
    def physical_tokens(self) -> int:
        return int(self.input_ids.numel())

    @property
    def non_padding_tokens(self) -> int:
        return int(self.attention_mask.sum().item())

    @property
    def supervised_tokens(self) -> int:
        return int((self.labels[..., 1:] != IGNORE_INDEX).sum().item())


def conversation_filter_reason(record: dict[str, Any]) -> str | None:
    """Return why a record is outside Chapter 16, or ``None`` when accepted."""

    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return "missing_or_empty_conversations"
    for message in conversations:
        if not isinstance(message, dict):
            return "message_is_not_an_object"
        if message.get("role") not in ALLOWED_ROLES:
            return "unsupported_role"
        if not isinstance(message.get("content"), str):
            return "content_is_not_text"
        extra = set(message) - {"role", "content"}
        if extra:
            return "extra_message_fields"
    if not any(message["role"] == "assistant" for message in conversations):
        return "no_assistant_message"
    return None


def is_plain_conversation(record: dict[str, Any]) -> bool:
    """Return whether Chapter 16 can represent a MiniMind record losslessly."""

    return conversation_filter_reason(record) is None


def validate_conversation(conversations: Any) -> list[dict[str, str]]:
    record = {"conversations": conversations}
    if not is_plain_conversation(record):
        raise ValueError(
            "Chapter 16 only accepts non-empty system/user/assistant messages "
            "with role and content fields; tool and reasoning records are deferred."
        )
    return [
        {"role": message["role"], "content": message["content"]}
        for message in conversations
    ]


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("source_line", line_number)
            records.append(record)
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def select_record(records: list[dict[str, Any]], cursor: int) -> dict[str, Any]:
    if cursor < 0:
        raise ValueError("cursor must be non-negative")
    return records[cursor % len(records)]


def encode_conversation(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int,
) -> EncodedConversation:
    if max_length < 4:
        raise ValueError("max_length must be at least 4")
    conversations = validate_conversation(record.get("conversations"))
    rendered = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=False,
    )
    token_ids = tokenizer.encode(rendered)
    if not token_ids:
        raise ValueError("chat template produced no tokens")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    token_ids = token_ids[: max_length - 1]
    if not token_ids or token_ids[-1] != eos_id:
        token_ids.append(eos_id)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id
    non_padding = len(token_ids)
    padded_ids = token_ids + [pad_id] * (max_length - non_padding)
    attention_mask = [1] * non_padding + [0] * (max_length - non_padding)
    labels = [token if keep else IGNORE_INDEX for token, keep in zip(padded_ids, attention_mask)]

    return EncodedConversation(
        input_ids=torch.tensor([padded_ids], dtype=torch.long),
        attention_mask=torch.tensor([attention_mask], dtype=torch.long),
        labels=torch.tensor([labels], dtype=torch.long),
        source_line=int(record.get("source_line", -1)),
        rendered_text=rendered,
    )


def shifted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [batch, sequence, vocab] and labels [batch, sequence]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels batch/sequence shapes must match")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
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
