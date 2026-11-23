import torch


UNKNOWN = "<unk>"


def show_encoding(name, units, vocabulary):
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    if UNKNOWN not in token_to_id:
        raise ValueError("Vocabulary must contain <unk>")
    tokens = [unit if unit in token_to_id else UNKNOWN for unit in units]
    ids = torch.tensor(
        [token_to_id[token] for token in tokens], dtype=torch.int64, device="cpu"
    )
    print(
        f"{name}: vocab={len(vocabulary)}, tokens={ids.numel()}, "
        f"unknown={tokens.count(UNKNOWN)}"
    )
    print("  pieces: " + " | ".join(tokens))
    print(f"  ids: {ids.tolist()}")


def main():
    text = "player replay"
    words = text.split()
    print(f"text: {text}")

    word_vocab = (UNKNOWN, "play", "player")
    alphabet = (UNKNOWN, "a", "e", "l", "p", "r", "y")
    char_units = [char for word in words for char in word]

    subword_vocab = alphabet + ("play", "er", "re")
    # Boundaries for this example are supplied by hand.
    subword_groups = [("play", "er"), ("re", "play")]
    for word, pieces in zip(words, subword_groups, strict=True):
        if "".join(pieces) != word:
            raise ValueError(f"Pieces do not reconstruct {word!r}")

    show_encoding("word", words, word_vocab)
    show_encoding("char", char_units, alphabet)
    subword_units = [piece for pieces in subword_groups for piece in pieces]
    show_encoding("subword", subword_units, subword_vocab)


if __name__ == "__main__":
    main()
