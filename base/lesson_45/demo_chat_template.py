import argparse
import hashlib
import json
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
import torch

from qwen_tokenizer import load_tokenizer


CONFIG_SIZE = 9_678
CONFIG_HASH = "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"


def load_template(path):
    data = path.read_bytes()
    if len(data) != CONFIG_SIZE or hashlib.sha256(data).hexdigest() != CONFIG_HASH:
        raise ValueError("tokenizer_config.json does not match the pinned Qwen3 revision")
    config = json.loads(data)
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return environment.from_string(config["chat_template"]), config["added_tokens_decoder"]


def positions(tokens, token_id):
    return (tokens == token_id).nonzero(as_tuple=True)[0].tolist()


def main():
    parser = argparse.ArgumentParser(description="Render the fixed Qwen3 chat template")
    parser.add_argument("tokenizer_json", type=Path)
    parser.add_argument("tokenizer_config", type=Path)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer_json)
    template, registry = load_template(args.tokenizer_config)
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello", "reasoning_content": "Greet"},
        {"role": "user", "content": "Next"},
    ]
    context = dict(messages=messages, tools=None)
    history = template.render(**context, add_generation_prompt=False)
    enabled = template.render(**context, add_generation_prompt=True, enable_thinking=True)
    disabled = template.render(**context, add_generation_prompt=True, enable_thinking=False)

    header = "<|im_start|>assistant\n"
    empty_think = "<think>\n\n</think>\n\n"
    assert enabled == history + header and disabled == enabled + empty_think
    assert "Greet" not in history and "<think>" not in history
    print(f"history={history!r}")
    print(f"generation suffix (thinking on): {enabled[len(history):]!r}")
    print(f"generation suffix (thinking off): {disabled[len(history):]!r}")

    raw = tokenizer.encode(disabled)
    print(f"tokenizer.json only: count={len(raw)}, isolated <think>={tokenizer.encode('<think>')}")
    extra = []
    for token_id, marker in ((151667, "<think>"), (151668, "</think>")):
        entry = registry[str(token_id)]
        assert entry["content"] == marker and not any(
            entry[key] for key in ("lstrip", "rstrip", "single_word", "normalized", "special")
        )
        assert marker not in tokenizer.added_to_id and token_id not in raw
        extra.append(dict(entry, id=token_id))
    extended = load_tokenizer(args.tokenizer_json, additional_tokens=extra)
    tokens = torch.tensor(extended.encode(disabled), dtype=torch.int64, device="cpu")
    expected = {
        "<|im_start|>": (151644, True), "<|im_end|>": (151645, True),
        "<think>": (151667, False), "</think>": (151668, False),
    }
    print(f"with config markers: shape={tuple(tokens.shape)}, dtype={tokens.dtype}, device={tokens.device}")
    for marker, (expected_id, expected_special) in expected.items():
        token_id = extended.added_to_id[marker]
        special = token_id in extended.special_ids
        where = positions(tokens, token_id)
        assert (token_id, special) == (expected_id, expected_special)
        assert len(where) == {"<|im_start|>": 5, "<|im_end|>": 4}.get(marker, 1)
        print(f"{marker}: id={token_id}, special={special}, positions={where}")


if __name__ == "__main__":
    main()
