"""面向教学的 Qwen3 Byte-level BPE Tokenizer。

本实现直接读取模型目录中的 tokenizer.json，不依赖 Transformers 或
tokenizers 库。它只覆盖第 01 期需要的文本编码、文本解码和单轮聊天模板。
"""

import json
import unicodedata
from pathlib import Path

import regex


# 这个正则表达式来自固定模型版本的 tokenizer.json。它先把文本切成
# 较小片段，之后再对每个片段分别执行 Byte-level BPE。
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
    """建立可逆的 byte 到 Unicode 字符映射。"""

    visible_bytes = list(range(ord("!"), ord("~") + 1))
    visible_bytes += list(range(ord("¡"), ord("¬") + 1))
    visible_bytes += list(range(ord("®"), ord("ÿ") + 1))

    all_bytes = list(visible_bytes)
    unicode_numbers = list(visible_bytes)
    extra_number = 0

    for byte_value in range(256):
        if byte_value not in visible_bytes:
            all_bytes.append(byte_value)
            unicode_numbers.append(256 + extra_number)
            extra_number += 1

    unicode_characters = [chr(number) for number in unicode_numbers]
    return dict(zip(all_bytes, unicode_characters))


class Qwen3Tokenizer:
    """从 tokenizer.json 加载词表和 BPE 合并规则。"""

    def __init__(self, model_directory):
        tokenizer_path = Path(model_directory) / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError("没有找到 tokenizer 配置: " + str(tokenizer_path))

        with tokenizer_path.open("r", encoding="utf-8") as file:
            tokenizer_data = json.load(file)

        model_data = tokenizer_data["model"]
        if model_data["type"] != "BPE":
            raise ValueError("本期手写 Tokenizer 只支持 BPE 模型")

        self.vocab = model_data["vocab"]
        self.id_to_token = {token_id: token for token, token_id in self.vocab.items()}

        # 数字越小的 merge 优先级越高。
        self.merge_ranks = {
            tuple(token_pair): rank
            for rank, token_pair in enumerate(model_data["merges"])
        }
        self.bpe_cache = {}

        self.added_token_to_id = {}
        self.added_id_to_token = {}
        self.special_token_ids = set()
        for token_data in tokenizer_data.get("added_tokens", []):
            content = token_data["content"]
            token_id = token_data["id"]
            self.added_token_to_id[content] = token_id
            self.added_id_to_token[token_id] = content
            if token_data.get("special", False):
                self.special_token_ids.add(token_id)

        # added token 必须在普通文本切分前单独识别。
        added_tokens = sorted(self.added_token_to_id, key=len, reverse=True)
        escaped_tokens = [regex.escape(token) for token in added_tokens]
        self.added_token_pattern = regex.compile("(" + "|".join(escaped_tokens) + ")")

        self.byte_to_unicode = build_byte_to_unicode()
        self.unicode_to_byte = {
            character: byte_value
            for byte_value, character in self.byte_to_unicode.items()
        }
        self.eos_token_id = self.added_token_to_id["<|im_end|>"]

    def bpe(self, byte_text):
        """按照 merge 优先级，把字符序列逐步合并成词表 Token。"""

        if byte_text in self.bpe_cache:
            return self.bpe_cache[byte_text]

        pieces = list(byte_text)
        while len(pieces) > 1:
            adjacent_pairs = set(zip(pieces[:-1], pieces[1:]))

            best_pair = None
            best_rank = None
            for pair in adjacent_pairs:
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            # 当前相邻 Token 已经没有可用的 merge，BPE 到此结束。
            if best_pair is None:
                break

            first, second = best_pair
            merged_pieces = []
            index = 0
            while index < len(pieces):
                if (
                    index + 1 < len(pieces)
                    and pieces[index] == first
                    and pieces[index + 1] == second
                ):
                    merged_pieces.append(first + second)
                    index += 2
                else:
                    merged_pieces.append(pieces[index])
                    index += 1
            pieces = merged_pieces

        self.bpe_cache[byte_text] = pieces
        return pieces

    def encode_ordinary_text(self, text):
        """编码一段不包含 added token 的普通文本。"""

        token_ids = []
        normalized_text = unicodedata.normalize("NFC", text)

        for text_piece in PRE_TOKEN_PATTERN.findall(normalized_text):
            utf8_bytes = text_piece.encode("utf-8")
            byte_text = "".join(self.byte_to_unicode[value] for value in utf8_bytes)

            for bpe_token in self.bpe(byte_text):
                token_id = self.vocab.get(bpe_token)
                if token_id is None:
                    raise ValueError("BPE 结果不在词表中: " + repr(bpe_token))
                token_ids.append(token_id)

        return token_ids

    def encode(self, text):
        """把文本编码为 Token IDs，并正确处理 Qwen3 added tokens。"""

        token_ids = []
        text_parts = self.added_token_pattern.split(text)

        for text_part in text_parts:
            if not text_part:
                continue
            if text_part in self.added_token_to_id:
                token_ids.append(self.added_token_to_id[text_part])
            else:
                token_ids.extend(self.encode_ordinary_text(text_part))

        return token_ids

    def decode(self, token_ids, skip_special_tokens=False):
        """把 Token IDs 还原为 UTF-8 文本。"""

        output_parts = []
        ordinary_tokens = []

        def flush_ordinary_tokens():
            if not ordinary_tokens:
                return

            byte_text = "".join(ordinary_tokens)
            byte_values = bytes(self.unicode_to_byte[character] for character in byte_text)
            output_parts.append(byte_values.decode("utf-8", errors="replace"))
            ordinary_tokens.clear()

        for token_id in token_ids:
            if token_id in self.added_id_to_token:
                flush_ordinary_tokens()
                if not (skip_special_tokens and token_id in self.special_token_ids):
                    output_parts.append(self.added_id_to_token[token_id])
                continue

            token = self.id_to_token.get(token_id)
            if token is None:
                raise ValueError("Token ID 不在词表中: " + str(token_id))
            ordinary_tokens.append(token)

        flush_ordinary_tokens()
        return "".join(output_parts)

    def apply_chat_template(self, messages, add_generation_prompt=True):
        """构造本期使用的 Qwen3 non-thinking 文本模板。

        第 01 期只需要 system/user 文本消息，不覆盖工具调用、多模态内容和
        assistant 历史 reasoning。这些更复杂的模板分支不属于本期主题。
        """

        prompt_parts = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in ("system", "user") or not isinstance(content, str):
                raise ValueError("本期聊天模板只支持 system/user 纯文本消息")

            prompt_parts.append(
                "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
            )

        if add_generation_prompt:
            # enable_thinking=False 时，Qwen3 模板会插入一个空 think 块。
            prompt_parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")

        return "".join(prompt_parts)

    def encode_chat_prompt(self, prompt):
        """把单轮用户问题直接转换为模型输入 Token IDs。"""

        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.apply_chat_template(messages, add_generation_prompt=True)
        return self.encode(prompt_text)

