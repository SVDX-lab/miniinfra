import math

import torch


def distribution(logits, temperature, top_k=None, top_p=1.0):
    if logits.ndim != 1 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("expected a nonempty floating logits vector")
    if not torch.isfinite(logits).all() or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("logits must be finite and temperature positive")
    size = logits.numel()
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= size):
        raise ValueError("top_k must be between one and vocabulary size")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    adjusted = (logits - logits.max()) / temperature
    candidates = adjusted.argsort(descending=True, stable=True)
    if top_k is not None:
        candidates = candidates[:top_k]
    ranked = adjusted[candidates].softmax(dim=0)
    if top_p < 1:
        cumulative_before = torch.cat((ranked.new_zeros(1), ranked.cumsum(0)[:-1]))
        keep = cumulative_before < top_p
        candidates, ranked = candidates[keep], ranked[keep]
    result = torch.zeros_like(logits)
    result[candidates] = ranked / ranked.sum()
    return result


def draw(probs, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.multinomial(probs, num_samples=8, replacement=True, generator=generator)


def main():
    logits = torch.tensor([6, 3, 1], dtype=torch.float64).log()
    base = distribution(logits, 1)
    colder = distribution(logits, 0.5)
    two = distribution(logits, 1, top_k=2)
    nucleus = distribution(logits, 1, top_p=0.8)
    combined = distribution(logits, 1, top_k=2, top_p=0.65)
    references = ([0.6, 0.3, 0.1], [36/46, 9/46, 1/46],
                  [2/3, 1/3, 0], [2/3, 1/3, 0], [1, 0, 0])
    for label, result, expected in zip(
        ("T=1", "T=0.5", "top-k=2", "top-p=0.8", "top-k=2 then top-p=0.65"),
        (base, colder, two, nucleus, combined),
        references,
        strict=True,
    ):
        torch.testing.assert_close(result, torch.tensor(expected, dtype=logits.dtype), rtol=0, atol=1e-12)
        print(f"{label}: {[round(p, 6) for p in result.tolist()]}")
    draws = draw(nucleus, 17)
    assert torch.equal(draws, draw(nucleus, 17)) and (draws < 2).all()
    print(f"seed=17, eight draws={draws.tolist()}; replay=pass")


if __name__ == "__main__":
    main()
