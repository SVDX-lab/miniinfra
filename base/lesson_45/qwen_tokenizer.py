"""Pinned Qwen3 encoding, adapted from the independent lesson 07 example."""

import hashlib
import json
import unicodedata

import regex


TOKENIZER_SIZE = 7_031_645
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"


def load_tokenizer(path, *, additional_tokens=()):
    data = path.read_bytes()
    if len(data) != TOKENIZER_SIZE or hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("tokenizer.json does not match the pinned Qwen3 revision")
    payload = json.loads(data)
    payload["added_tokens"].extend(additional_tokens)
    return Qwen3ByteBPE(payload)


def byte_alphabet():
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    extra = 256
    mapping = {}
    for value in range(256):
        if value in visible:
            mapping[value] = chr(value)
        else:
            mapping[value] = chr(extra)
            extra += 1
    return mapping


def merge_pair(symbols, pair):
    merged = []
    index = 0
    while index < len(symbols):
        if tuple(symbols[index:index + 2]) == pair:
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


class Qwen3ByteBPE:
    def __init__(self, data):
        model = data["model"]
        split, _ = data["pre_tokenizer"]["pretokenizers"]
        self.pattern = regex.compile(split["pattern"]["Regex"])
        self.vocab = model["vocab"]
        self.merge_ranks = {
            tuple(rule.split(" ")): rank for rank, rule in enumerate(model["merges"])
        }
        self.byte_to_symbol = byte_alphabet()
        self.symbol_to_byte = {
            symbol: byte for byte, symbol in self.byte_to_symbol.items()
        }
        if not all(symbol in self.vocab for symbol in self.symbol_to_byte):
            raise ValueError("Tokenizer is missing one or more base byte symbols")

        added = data["added_tokens"]
        self.added_to_id = {entry["content"]: entry["id"] for entry in added}
        self.special_ids = {entry["id"] for entry in added if entry["special"]}
        names = sorted(self.added_to_id, key=len, reverse=True)
        self.added_pattern = regex.compile("(" + "|".join(map(regex.escape, names)) + ")")

    def encode(self, text):
        token_ids = []
        for part in self.added_pattern.split(text):
            if part in self.added_to_id:
                token_ids.append(self.added_to_id[part])
            else:
                token_ids.extend(self.encode_ordinary(part))
        return token_ids

    def encode_ordinary(self, text):
        normalized = unicodedata.normalize("NFC", text)
        segments = self.pattern.findall(normalized)
        if "".join(segments) != normalized:
            raise ValueError("Pre-tokenization lost input characters")

        token_ids = []
        for segment in segments:
            symbols = [self.byte_to_symbol[byte] for byte in segment.encode("utf-8")]
            while True:
                pairs = zip(symbols, symbols[1:])
                candidates = (pair for pair in pairs if pair in self.merge_ranks)
                best = min(candidates, key=self.merge_ranks.__getitem__, default=None)
                if best is None:
                    break
                symbols = merge_pair(symbols, best)
            token_ids.extend(self.vocab[symbol] for symbol in symbols)
        return token_ids
