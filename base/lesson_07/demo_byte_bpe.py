import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import regex
import torch


TOKENIZER_SIZE = 7_031_645
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"


def load_tokenizer(path):
    data = path.read_bytes()
    if len(data) != TOKENIZER_SIZE or hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("tokenizer.json does not match the pinned Qwen3 revision")
    return Qwen3ByteBPE(json.loads(data))


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
        self.id_to_token = {token_id: token for token, token_id in self.vocab.items()}
        self.merge_ranks = {
            tuple(rule.split(" ")): rank for rank, rule in enumerate(model["merges"])
        }
        self.byte_fallback = model["byte_fallback"]
        self.unk_token = model["unk_token"]
        self.byte_to_symbol = byte_alphabet()
        self.symbol_to_byte = {
            symbol: byte for byte, symbol in self.byte_to_symbol.items()
        }
        if not all(symbol in self.vocab for symbol in self.symbol_to_byte):
            raise ValueError("Tokenizer is missing one or more base byte symbols")

        added = data["added_tokens"]
        self.added_to_id = {entry["content"]: entry["id"] for entry in added}
        self.id_to_added = {entry["id"]: entry["content"] for entry in added}
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

    def decode(self, token_ids, *, skip_special=False):
        buffer = bytearray()
        for token_id in token_ids:
            if skip_special and token_id in self.special_ids:
                continue
            if token_id in self.id_to_added:
                buffer.extend(self.id_to_added[token_id].encode("utf-8"))
                continue
            if token_id not in self.id_to_token:
                raise ValueError(f"Token ID outside vocabulary: {token_id}")
            token = self.id_to_token[token_id]
            buffer.extend(self.symbol_to_byte[symbol] for symbol in token)
        return buffer.decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="Read the pinned Qwen3 tokenizer")
    parser.add_argument("tokenizer_json", type=Path)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer_json)
    print(f"file: {TOKENIZER_SIZE} bytes, SHA-256 verified")
    print(f"BPE: vocab={len(tokenizer.vocab)}, merges={len(tokenizer.merge_ranks)}")
    print(f"added: {len(tokenizer.added_to_id)}, special: {len(tokenizer.special_ids)}")
    print(f"byte alphabet: {len(tokenizer.byte_to_symbol)}/256")
    print(f"byte_fallback={tokenizer.byte_fallback}, unk_token={tokenizer.unk_token}")

    sample = "\u4f60"
    raw = sample.encode("utf-8")
    base_ids = [tokenizer.vocab[tokenizer.byte_to_symbol[byte]] for byte in raw]
    sample_ids = tokenizer.encode(sample)
    print(f"{ascii(sample)}: bytes={raw.hex(' ')}, base={base_ids}, BPE={sample_ids}")
    if tokenizer.decode(base_ids) != sample:
        raise AssertionError("Base bytes did not round-trip")

    samples = ("hello world", "\u4f60\u597d\uff0c\u4e16\u754c", "\U0001f642")
    total_bytes = total_tokens = 0
    for text in samples:
        byte_count = len(text.encode("utf-8"))
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.int64, device="cpu")
        values = ids.tolist()
        if tokenizer.decode(values) != text:
            raise AssertionError(f"Round-trip failed for {text!r}")
        token_count = ids.numel()
        total_bytes += byte_count
        total_tokens += token_count
        print(f"{ascii(text)}: bytes={byte_count}, tokens={token_count}")
        print(f"  ids={values}")
    print(f"total: {total_bytes}/{total_tokens} = {total_bytes / total_tokens:.2f} bytes/token")

    text = "hello<|im_end|><tool_call>"
    ids = tokenizer.encode(text)
    print(f"added probe: {ids}")
    print(f"skip special: {tokenizer.decode(ids, skip_special=True)!r}")
    normalized = tokenizer.decode(tokenizer.encode("e\u0301"))
    print(f"NFC: {ascii('e' + chr(0x0301))} -> {ascii(normalized)}")


if __name__ == "__main__":
    main()
