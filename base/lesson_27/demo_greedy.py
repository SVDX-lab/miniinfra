import torch


def generate(model, prompt, *, eos, limit, window):
    if not prompt or len(prompt) > window or limit < 0:
        raise ValueError("need a nonempty prompt within the window and a nonnegative limit")
    tokens = list(prompt)
    trace = []
    for _ in range(limit):
        if len(tokens) == window:
            return tokens, trace, "context"
        logits = model(torch.tensor(tokens, dtype=torch.int64, device="cpu"))
        if (logits.ndim != 2 or logits.shape[0] != len(tokens)
                or not logits.is_floating_point() or logits.device.type != "cpu"
                or not 0 <= eos < logits.shape[1] or not torch.isfinite(logits).all()):
            raise ValueError("expected finite CPU logits [positions, vocabulary] and valid EOS")
        chosen = int(logits[-1].argmax())
        trace.append((tokens.copy(), chosen))
        tokens.append(chosen)
        if chosen == eos:
            return tokens, trace, "eos"
    return tokens, trace, "limit"


def main():
    table = torch.tensor([[0, 3, 0, 0], [0, 0, 3, 0],
                          [0, 0, 0, 3], [3, 0, 0, 0]], dtype=torch.float32)
    prompt = [0, 1]

    def toy_model(prefix):
        return table[prefix]

    tokens, trace, reason = generate(toy_model, prompt, eos=3, limit=4, window=8)
    for prefix, choice in trace:
        print(f"forward prefix={prefix}, next={choice}")
    print(f"tokens={tokens}, stop={reason}, forwarded positions={sum(len(p) for p, _ in trace)}")
    assert tokens == [0, 1, 2, 3] and reason == "eos" and prompt == [0, 1]
    _, _, window_stop = generate(toy_model, prompt, eos=3, limit=4, window=2)
    _, _, limit_stop = generate(toy_model, prompt, eos=3, limit=1, window=8)
    assert (window_stop, limit_stop) == ("context", "limit")
    print(f"boundary checks: {window_stop}, {limit_stop}")


if __name__ == "__main__":
    main()
