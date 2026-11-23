"""Qwen3 conversation rendering and Assistant-only label construction.

The model input is rendered once.  Baseline and variant labels are then derived
from the same token IDs; no second template invocation is allowed to change the
input.  The supervised span starts after ``<|im_start|>assistant\n`` and ends
at the assistant message's real ``<|im_end|>`` token, inclusive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch


IGNORE_INDEX = -100
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
LossMode = Literal["full_conversation", "assistant_only"]


@dataclass(frozen=True)
class TextSegment:
    text: str
    supervised: bool
    role: str
    kind: str


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    supervised: bool
    role: str
    kind: str


@dataclass(frozen=True)
class EncodedConversation:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    assistant_loss_mask: torch.Tensor
    source_line: int
    rendered_text: str
    token_spans: tuple[TokenSpan, ...]
    loss_mode: LossMode

    @property
    def physical_tokens(self) -> int:
        return int(self.input_ids.numel())

    @property
    def non_padding_tokens(self) -> int:
        return int(self.attention_mask.sum().item())

    @property
    def supervised_tokens(self) -> int:
        return int((self.labels[..., 1:] != IGNORE_INDEX).sum().item())

    @property
    def supervision_ratio(self) -> float:
        denominator = max(1, self.non_padding_tokens - 1)
        return self.supervised_tokens / denominator


def _json(value: Any) -> str:
    """Match the compact JSON form used inside Qwen3 tool-call blocks."""

    rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ": "))
    return (
        rendered.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )


def _assistant_body(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ValueError("assistant content must be text or null")

    body = content
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise ValueError("assistant tool_calls must be a list")
    for index, raw_call in enumerate(tool_calls):
        if not isinstance(raw_call, dict):
            raise ValueError("each tool call must be an object")
        call = raw_call.get("function", raw_call)
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError("tool call must define a function name")
        arguments = call.get("arguments", {})
        arguments_text = arguments if isinstance(arguments, str) else _json(arguments)
        if (index == 0 and body) or index > 0:
            body += "\n"
        body += (
            '<tool_call>\n{"name": "'
            + call["name"]
            + '", "arguments": '
            + arguments_text
            + "}\n</tool_call>"
        )
    return body


def render_qwen3_segments(messages: Iterable[dict[str, Any]]) -> list[TextSegment]:
    """Render the fixed Qwen3 message boundary subset used by this chapter.

    It follows the pinned Qwen3 template's message wrappers and tool-call/tool-
    response serialization.  Thinking-content policy is deliberately excluded:
    Chapter 17 established a non-thinking training template and this chapter may
    change only labels, not the text presented to the model.
    """

    materialized = list(messages)
    if not materialized:
        raise ValueError("conversation must contain at least one message")

    segments: list[TextSegment] = []
    index = 0
    while index < len(materialized):
        message = materialized[index]
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        if role in {"system", "user"}:
            extra = set(message) - {"role", "content"}
            if extra:
                raise ValueError(f"unsupported {role} message fields: {sorted(extra)}")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"{role} content must be text")
            segments.append(TextSegment(IM_START + role + "\n", False, role, "header"))
            segments.append(TextSegment(content, False, role, "content"))
            segments.append(TextSegment(IM_END, False, role, "end"))
            segments.append(TextSegment("\n", False, role, "separator"))
            index += 1
            continue

        if role == "assistant":
            extra = set(message) - {"role", "content", "tool_calls"}
            if extra:
                raise ValueError(
                    "unsupported assistant message fields would be lossy: " + str(sorted(extra))
                )
            segments.append(TextSegment(IM_START + "assistant\n", False, role, "header"))
            body = _assistant_body(message)
            if body:
                segments.append(TextSegment(body, True, role, "content"))
            segments.append(TextSegment(IM_END, True, role, "end"))
            segments.append(TextSegment("\n", False, role, "separator"))
            index += 1
            continue

        if role == "tool":
            extra = set(message) - {"role", "content"}
            if extra:
                raise ValueError(f"unsupported tool message fields: {sorted(extra)}")
            segments.append(TextSegment(IM_START + "user", False, "tool", "header"))
            while index < len(materialized) and materialized[index].get("role") == "tool":
                content = materialized[index].get("content")
                if not isinstance(content, str):
                    raise ValueError("tool content must be text")
                segments.append(
                    TextSegment(
                        "\n<tool_response>\n" + content + "\n</tool_response>",
                        False,
                        "tool",
                        "response",
                    )
                )
                index += 1
            segments.append(TextSegment(IM_END, False, "tool", "end"))
            segments.append(TextSegment("\n", False, "tool", "separator"))
            continue

        raise ValueError(f"unsupported message role: {role!r}")

    if not any(segment.role == "assistant" for segment in segments):
        raise ValueError("conversation must contain at least one assistant message")
    return segments


def tokenize_segments(tokenizer: Any, segments: Iterable[TextSegment]) -> tuple[list[int], list[bool], list[TokenSpan], str]:
    token_ids: list[int] = []
    assistant_mask: list[bool] = []
    spans: list[TokenSpan] = []
    text_parts: list[str] = []
    for segment in segments:
        start = len(token_ids)
        ids = tokenizer.encode(segment.text)
        token_ids.extend(ids)
        assistant_mask.extend([segment.supervised] * len(ids))
        spans.append(TokenSpan(start, len(token_ids), segment.supervised, segment.role, segment.kind))
        text_parts.append(segment.text)
    rendered = "".join(text_parts)
    whole_ids = tokenizer.encode(rendered)
    if token_ids != whole_ids:
        raise ValueError(
            "tokenization changed across a template segment boundary; "
            "the Assistant-only mask would be ambiguous"
        )
    return token_ids, assistant_mask, spans, rendered


def encode_conversation(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int,
    loss_mode: LossMode,
) -> EncodedConversation:
    if max_length < 4:
        raise ValueError("max_length must be at least 4")
    if loss_mode not in {"full_conversation", "assistant_only"}:
        raise ValueError(f"unsupported loss mode: {loss_mode}")

    segments = render_qwen3_segments(record.get("conversations", []))
    token_ids, assistant_flags, spans, rendered = tokenize_segments(tokenizer, segments)
    if not token_ids:
        raise ValueError("chat template produced no tokens")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    truncated = len(token_ids) > max_length - 1
    token_ids = token_ids[: max_length - 1]
    assistant_flags = assistant_flags[: max_length - 1]
    if not token_ids or token_ids[-1] != eos_id:
        token_ids.append(eos_id)
        # This is a length sentinel added by the encoder, not a real assistant
        # message boundary from the pinned chat template.
        assistant_flags.append(False)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    non_padding = len(token_ids)
    padded_ids = token_ids + [pad_id] * (max_length - non_padding)
    attention = [1] * non_padding + [0] * (max_length - non_padding)
    assistant_flags += [False] * (max_length - len(assistant_flags))
    if loss_mode == "full_conversation":
        labels = [token if keep else IGNORE_INDEX for token, keep in zip(padded_ids, attention)]
    else:
        labels = [token if keep else IGNORE_INDEX for token, keep in zip(padded_ids, assistant_flags)]

    supervised_after_shift = sum(label != IGNORE_INDEX for label in labels[1:])
    if supervised_after_shift == 0:
        reason = " after truncation" if truncated else ""
        raise ValueError(f"conversation has no supervised next-token targets{reason}")

    return EncodedConversation(
        input_ids=torch.tensor([padded_ids], dtype=torch.long),
        attention_mask=torch.tensor([attention], dtype=torch.long),
        labels=torch.tensor([labels], dtype=torch.long),
        assistant_loss_mask=torch.tensor([assistant_flags], dtype=torch.bool),
        source_line=int(record.get("source_line", -1)),
        rendered_text=rendered,
        token_spans=tuple(spans),
        loss_mode=loss_mode,
    )


def assert_only_labels_differ(baseline: EncodedConversation, variant: EncodedConversation) -> None:
    torch.testing.assert_close(baseline.input_ids, variant.input_ids, rtol=0, atol=0)
    torch.testing.assert_close(baseline.attention_mask, variant.attention_mask, rtol=0, atol=0)
    baseline_active = baseline.labels != IGNORE_INDEX
    variant_active = variant.labels != IGNORE_INDEX
    if bool((variant_active & ~baseline_active).any()):
        raise AssertionError("variant supervises a position excluded by baseline")
    if bool((variant.labels[variant_active] != baseline.labels[variant_active]).any()):
        raise AssertionError("baseline and variant targets differ at an active variant position")
    if not bool((baseline_active & ~variant_active).any()):
        raise AssertionError("the two objectives have identical labels")


def supervised_token_indices(sample: EncodedConversation) -> list[int]:
    return torch.nonzero(sample.labels[0] != IGNORE_INDEX, as_tuple=False).flatten().tolist()
