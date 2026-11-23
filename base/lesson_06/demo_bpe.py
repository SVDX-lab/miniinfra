from collections import Counter

import torch


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


def train_bpe(text, max_merges):
    if max_merges < 0:
        raise ValueError("max_merges must be nonnegative")
    word_counts = Counter(word for word in text.split(" ") if word)
    words = {word: list(word) for word in word_counts}
    vocabulary = sorted({char for word in words for char in word}) + [" "]
    print(f"base vocab: {vocabulary!r}")
    rules = []

    for step in range(1, max_merges + 1):
        pairs = Counter()
        for word, symbols in words.items():
            for pair in zip(symbols, symbols[1:]):
                pairs[pair] += word_counts[word]
        if not pairs:
            print("stop: no adjacent pairs")
            break

        # Largest count first, then lexicographically smallest pair.
        best = min(pairs, key=lambda pair: (-pairs[pair], pair))
        rules.append(best)
        piece = best[0] + best[1]
        if piece not in vocabulary:
            vocabulary.append(piece)
        for word, symbols in words.items():
            words[word] = merge_pair(symbols, best)

        state = ", ".join(
            f"{'|'.join(symbols)}:{word_counts[word]}"
            for word, symbols in words.items()
        )
        print(f"{step}: {best!r} count={pairs[best]} -> {state}")
    return vocabulary, rules


def encode(text, vocabulary, rules):
    ranks = {pair: rank for rank, pair in enumerate(rules)}
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    pieces = []
    for index, word in enumerate(text.split(" ")):
        if index > 0:
            pieces.append(" ")
        symbols = list(word)
        for symbol in symbols:
            if symbol not in token_to_id:
                raise ValueError(f"Character outside vocabulary: {symbol!r}")
        while True:
            candidates = (pair for pair in zip(symbols, symbols[1:]) if pair in ranks)
            best = min(candidates, key=ranks.__getitem__, default=None)
            if best is None:
                break
            symbols = merge_pair(symbols, best)
        pieces.extend(symbols)
    ids = torch.tensor(
        [token_to_id[piece] for piece in pieces], dtype=torch.int64, device="cpu"
    )
    return pieces, ids


def decode(ids, vocabulary):
    if ids.ndim != 1 or ids.dtype != torch.int64 or ids.device.type != "cpu":
        raise ValueError("Expected a one-dimensional CPU torch.int64 tensor")
    pieces = []
    for token_id in ids.tolist():
        if not 0 <= token_id < len(vocabulary):
            raise ValueError(f"Token ID outside vocabulary: {token_id}")
        pieces.append(vocabulary[token_id])
    return "".join(pieces)


def main():
    corpus = "aba aba ab ba ba"
    print(f"corpus: {corpus}")
    vocabulary, rules = train_bpe(corpus, max_merges=4)
    print(f"vocab: {vocabulary!r}")
    text = "abab ba"
    print(f"text: {text!r}")
    pieces, ids = encode(text, vocabulary, rules)
    print(f"pieces: {pieces!r}")
    print(f"ids: {ids.tolist()}")
    decoded = decode(ids, vocabulary)
    if decoded != text:
        raise AssertionError("Decoded text differs from the input")
    print(f"decoded: {decoded!r}")


if __name__ == "__main__":
    main()
