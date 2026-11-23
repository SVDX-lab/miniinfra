#!/usr/bin/env python3
"""CPU-only Chapter 22 checks: loader parity, ordering, bounds, error propagation."""

from __future__ import annotations

import queue
from pathlib import Path

import torch

from assistant_loss import EncodedConversation, IGNORE_INDEX
from data_paths import pad_pairs
from prefetch_pipeline import PrefetchBatchLoader, PrefetchWorker, SyncBatchLoader, materialize_host_batch
from token_shard import TokenShardWriter


MODEL_REV = "a" * 40
DATASET_REV = "b" * 40


def synthetic_sample(ids: list[int], supervised_from: int, source_line: int) -> EncodedConversation:
    labels = [IGNORE_INDEX if index < supervised_from else token for index, token in enumerate(ids)]
    return EncodedConversation(
        input_ids=torch.tensor([ids]),
        attention_mask=torch.tensor([[1] * len(ids)]),
        labels=torch.tensor([labels]),
        assistant_loss_mask=torch.tensor([[label != IGNORE_INDEX for label in labels]]),
        source_line=source_line,
        rendered_text="",
        token_spans=(),
        loss_mode="assistant_only",
    )


SAMPLES = [
    synthetic_sample(list(range(1, 12)), 4, 11),
    synthetic_sample(list(range(20, 32)), 6, 12),
    synthetic_sample(list(range(40, 45)), 2, 13),
    synthetic_sample(list(range(50, 57)), 3, 14),
    synthetic_sample(list(range(30, 36)), 3, 15),
    synthetic_sample(list(range(60, 68)), 5, 16),
]
BATCHES = [(0, 2), (3, 4), (1, 5)]


def build_corpus(tmp: Path):
    shard = tmp / "loader.bin"
    writer = TokenShardWriter(shard, MODEL_REV, DATASET_REV, max_length=32)
    for sample in SAMPLES:
        writer.add(sample)
    writer.close()
    from data_paths import ShardCorpus

    return ShardCorpus(shard, MODEL_REV, DATASET_REV)


def check_loader_parity_and_order(corpus) -> None:
    device = torch.device("cpu")
    sync = SyncBatchLoader(corpus, BATCHES, pad_token_id=0, pin=False)
    sync.start(device)
    prefetch = PrefetchBatchLoader(corpus, BATCHES, pad_token_id=0, queue_depth=2, pin=False)
    prefetch.start(device)
    sync_batches = []
    while True:
        try:
            gpu_batch, host, _, _ = sync.next_batch()
        except StopIteration:
            break
        sync_batches.append((gpu_batch, host))
    prefetch_batches = []
    while True:
        try:
            gpu_batch, host, exposed, supply = prefetch.next_batch()
        except StopIteration:
            break
        assert exposed >= 0 and supply >= 0
        prefetch_batches.append((gpu_batch, host))
    assert len(sync_batches) == len(prefetch_batches) == len(BATCHES)
    for (sync_gpu, sync_host), (pref_gpu, pref_host) in zip(sync_batches, prefetch_batches):
        assert sync_host.members == pref_host.members
        for name in ("input_ids", "attention_mask", "labels"):
            assert torch.equal(sync_gpu[name], pref_gpu[name]), f"{name} differs between loaders"
    prefetch.shutdown()
    sync.shutdown()


def check_queue_bound(corpus) -> None:
    worker = PrefetchWorker(corpus, BATCHES, pad_token_id=0, queue_depth=1, pin=False)
    worker.start()
    seen = []
    while True:
        item = worker.queue.get()
        if item is None:
            break
        seen.append(item)
        assert worker.queue.qsize() <= 1, "bounded queue exceeded its depth"
    assert len(seen) == len(BATCHES)


def check_error_propagation(tmp: Path, corpus) -> None:
    class FailingCorpus:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def materialize(self, index):
            self.calls += 1
            if self.calls >= 3:
                raise RuntimeError("boom")
            return self.inner.materialize(index)

    failing = FailingCorpus(corpus)
    worker = PrefetchWorker(failing, BATCHES, pad_token_id=0, queue_depth=4, pin=False)
    worker.start()
    worker.join(timeout=5)
    assert worker.error is not None, "worker error must surface"
    try:
        worker.check_error()
    except RuntimeError as error:
        assert "boom" in str(error.__cause__)
    else:
        raise AssertionError("check_error must raise")


def check_batch_accounting(corpus) -> None:
    batch = materialize_host_batch(corpus, BATCHES[0], pad_token_id=0)
    direct = pad_pairs([corpus.materialize(index) for index in BATCHES[0]], pad_token_id=0)
    assert torch.equal(batch.input_ids, direct.input_ids)
    assert batch.supervised_tokens == direct.supervised_tokens
    assert batch.tensor_bytes > 0


def main() -> None:
    tmp = Path("data/smoke_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for stale in tmp.glob("*.bin"):
        stale.unlink()
    corpus = build_corpus(tmp)
    check_loader_parity_and_order(corpus)
    check_queue_bound(corpus)
    check_error_propagation(tmp, corpus)
    check_batch_accounting(corpus)
    print("chapter_22 smoke tests passed")


if __name__ == "__main__":
    main()
