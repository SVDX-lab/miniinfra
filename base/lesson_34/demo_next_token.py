import torch


def shifted_pairs(input_ids, logits):
    if input_ids.ndim != 1 or input_ids.numel() < 2 or input_ids.dtype != torch.int64:
        raise ValueError("expected int64 input IDs [sequence] with at least two tokens")
    if (logits.ndim != 2 or logits.shape[0] != input_ids.numel()
            or not logits.is_floating_point() or logits.device != input_ids.device):
        raise ValueError("expected floating logits [sequence, vocabulary] on the input device")
    if (input_ids < 0).any() or (input_ids >= logits.shape[1]).any() or not torch.isfinite(logits).all():
        raise ValueError("token IDs and logits must be valid")
    return logits[:-1], input_ids[1:]


def main():
    ids = torch.tensor([0, 2, 1], dtype=torch.int64)
    weights = torch.tensor([[1, 2, 4], [4, 1, 2], [2, 4, 1]], dtype=torch.float64)
    logits = weights.log()
    scores, targets = shifted_pairs(ids, logits)
    losses = scores.logsumexp(dim=-1) - scores.gather(1, targets[:, None]).squeeze(1)
    print(f"inputs={ids.tolist()}, logits={tuple(logits.shape)}")
    print(f"scored rows={tuple(scores.shape)}, labels={targets.tolist()}")
    for pos, (prefix_id, label, greedy, loss) in enumerate(zip(
        ids[:-1].tolist(), targets.tolist(), scores.argmax(-1).tolist(), losses.tolist(),
        strict=True,
    )):
        print(f"position={pos}: observed={prefix_id}, target={label}, greedy={greedy}, loss={loss:.6f}")
    result = losses.mean()
    expected = torch.tensor([7/4, 7], dtype=logits.dtype).log()
    torch.testing.assert_close(losses, expected, atol=1e-12, rtol=0)
    print(f"mean loss={result.item():.6f}; final logits row has no target")


if __name__ == "__main__":
    main()
