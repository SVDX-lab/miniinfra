"""第 11 期独立使用的 Qwen3 Byte-level BPE Tokenizer。"""

import json
import unicodedata
from pathlib import Path

import regex


PRE_TOKEN_PATTERN = regex.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def build_byte_to_unicode():
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    all_bytes = list(visible)
    unicode_numbers = list(visible)
    extra = 0
    for byte_value in range(256):
        if byte_value not in visible:
            all_bytes.append(byte_value)
            unicode_numbers.append(256 + extra)
            extra += 1
    return dict(zip(all_bytes, [chr(number) for number in unicode_numbers]))


class Qwen3Tokenizer:
    def __init__(self, model_directory):
        tokenizer_path = Path(model_directory) / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError("没有找到 tokenizer.json: " + str(tokenizer_path))
        with tokenizer_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        model_data = data["model"]
        if model_data["type"] != "BPE":
            raise ValueError("本期 Tokenizer 只支持 BPE")
        self.vocab = model_data["vocab"]
        self.id_to_token = {token_id: token for token, token_id in self.vocab.items()}
        self.merge_ranks = {
            tuple(pair): rank for rank, pair in enumerate(model_data["merges"])
        }
        self.bpe_cache = {}
        self.added_token_to_id = {}
        self.added_id_to_token = {}
        self.special_token_ids = set()
        for token_data in data.get("added_tokens", []):
            content = token_data["content"]
            token_id = token_data["id"]
            self.added_token_to_id[content] = token_id
            self.added_id_to_token[token_id] = content
            if token_data.get("special", False):
                self.special_token_ids.add(token_id)
        added_tokens = sorted(self.added_token_to_id, key=len, reverse=True)
        escaped = [regex.escape(token) for token in added_tokens]
        self.added_token_pattern = regex.compile("(" + "|".join(escaped) + ")")
        self.byte_to_unicode = build_byte_to_unicode()
        self.unicode_to_byte = {
            character: byte_value
            for byte_value, character in self.byte_to_unicode.items()
        }
        self.eos_token_id = self.added_token_to_id["<|im_end|>"]

    def bpe(self, byte_text):
        cached = self.bpe_cache.get(byte_text)
        if cached is not None:
            return cached
        pieces = list(byte_text)
        while len(pieces) > 1:
            best_pair = None
            best_rank = None
            for pair in set(zip(pieces[:-1], pieces[1:])):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break
            first, second = best_pair
            merged = []
            index = 0
            while index < len(pieces):
                if (
                    index + 1 < len(pieces)
                    and pieces[index] == first
                    and pieces[index + 1] == second
                ):
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(pieces[index])
                    index += 1
            pieces = merged
        self.bpe_cache[byte_text] = pieces
        return pieces

    def encode_ordinary_text(self, text):
        token_ids = []
        for text_piece in PRE_TOKEN_PATTERN.findall(unicodedata.normalize("NFC", text)):
            byte_text = "".join(
                self.byte_to_unicode[value] for value in text_piece.encode("utf-8")
            )
            for bpe_token in self.bpe(byte_text):
                token_id = self.vocab.get(bpe_token)
                if token_id is None:
                    raise ValueError("BPE 结果不在词表中: " + repr(bpe_token))
                token_ids.append(token_id)
        return token_ids

    def encode(self, text):
        token_ids = []
        for part in self.added_token_pattern.split(text):
            if not part:
                continue
            if part in self.added_token_to_id:
                token_ids.append(self.added_token_to_id[part])
            else:
                token_ids.extend(self.encode_ordinary_text(part))
        return token_ids

    def decode(self, token_ids, skip_special_tokens=False):
        output = []
        ordinary = []

        def flush():
            if not ordinary:
                return
            byte_text = "".join(ordinary)
            values = bytes(self.unicode_to_byte[character] for character in byte_text)
            output.append(values.decode("utf-8", errors="replace"))
            ordinary.clear()

        for token_id in token_ids:
            if token_id in self.added_id_to_token:
                flush()
                if not (skip_special_tokens and token_id in self.special_token_ids):
                    output.append(self.added_id_to_token[token_id])
                continue
            token = self.id_to_token.get(token_id)
            if token is None:
                raise ValueError("Token ID 不在词表中: " + str(token_id))
            ordinary.append(token)
        flush()
        return "".join(output)

    def apply_chat_template(self, messages, add_generation_prompt=True):
        parts = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in ("system", "user") or not isinstance(content, str):
                raise ValueError("聊天模板只支持 system/user 纯文本消息")
            parts.append("<|im_start|>" + role + "\n" + content + "<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(parts)

    def encode_chat_prompt(self, prompt):
        return self.encode(
            self.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
        )
