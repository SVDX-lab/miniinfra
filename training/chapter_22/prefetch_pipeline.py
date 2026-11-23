"""Chapter 22 core: overlapped data supply with prefetch, pinned memory, async H2D.

Both loaders read the Chapter 21 token shard with one worker and return device
batches for the identical step body.  The only difference is scheduling:

- ``SyncBatchLoader`` (baseline) materializes batch k, blocks on its H2D copy
  on the default stream, then the step runs.  Materialization and H2D are fully
  exposed between steps.
- ``PrefetchBatchLoader`` (variant) runs a worker thread that materializes
  future batches into pinned host buffers behind a bounded queue, and uses
  double-buffered H2D: before computing step k it issues batch k+1's copy on a
  side stream with ``non_blocking=True`` and records an event; the compute
  stream waits on the previous event, so the copy of the next batch overlaps
  the compute of the current one.

The queue is bounded to cap host memory; the worker signals failure through a
sentinel so the main loop raises instead of hanging.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from data_paths import ShardCorpus, pad_pairs


BATCH_FIELDS = ("input_ids", "attention_mask", "labels")


@dataclass
class HostBatch:
    """Materialized CPU batch plus its bookkeeping."""

    members: tuple[int, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    supervised_tokens: int

    @property
    def tensor_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.cpu_tensors())

    def cpu_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.input_ids, self.attention_mask, self.labels)


def materialize_host_batch(corpus: ShardCorpus, members: Sequence[int], pad_token_id: int) -> HostBatch:
    pairs = [corpus.materialize(index) for index in members]
    batch = pad_pairs(pairs, pad_token_id)
    return HostBatch(
        members=tuple(members),
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        supervised_tokens=batch.supervised_tokens,
    )


def pin_host_batch(batch: HostBatch, pin: bool) -> HostBatch:
    """Copy the host batch into page-locked memory when CUDA is available."""

    if not pin or not torch.cuda.is_available():
        return batch
    return HostBatch(
        members=batch.members,
        input_ids=batch.input_ids.pin_memory(),
        attention_mask=batch.attention_mask.pin_memory(),
        labels=batch.labels.pin_memory(),
        supervised_tokens=batch.supervised_tokens,
    )


def to_device_batch(batch: HostBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch.input_ids.to(device),
        "attention_mask": batch.attention_mask.to(device),
        "labels": batch.labels.to(device),
    }


class PrefetchWorker(threading.Thread):
    """Materializes and pins future batches behind a bounded queue."""

    def __init__(
        self,
        corpus: ShardCorpus,
        batches: Sequence[Sequence[int]],
        pad_token_id: int,
        queue_depth: int,
        pin: bool,
    ) -> None:
        super().__init__(daemon=True)
        if queue_depth < 1:
            raise ValueError("queue_depth must be at least one")
        self.corpus = corpus
        self.batches = list(batches)
        self.pad_token_id = pad_token_id
        self.queue: "queue.Queue[Optional[HostBatch]]" = queue.Queue(maxsize=queue_depth)
        self.pin = pin
        self.error: Optional[BaseException] = None

    def run(self) -> None:
        try:
            for members in self.batches:
                batch = pin_host_batch(materialize_host_batch(self.corpus, members, self.pad_token_id), self.pin)
                self.queue.put(batch)
        except BaseException as error:  # noqa: BLE001 - must reach the main thread
            self.error = error
        finally:
            self.queue.put(None)

    def check_error(self) -> None:
        if self.error is not None:
            raise RuntimeError("prefetch worker failed") from self.error


class SyncBatchLoader:
    """Baseline: materialize, pad, and block on H2D inside the step path."""

    def __init__(self, corpus: ShardCorpus, batches: Sequence[Sequence[int]], pad_token_id: int, pin: bool = False) -> None:
        self.corpus = corpus
        self.batches = list(batches)
        self.pad_token_id = pad_token_id
        self.pin = pin
        self.position = 0
        self.device: Optional[torch.device] = None

    def start(self, device) -> None:
        self.device = torch.device(device)

    def next_batch(self) -> tuple[dict[str, torch.Tensor], HostBatch, float, float]:
        """Return (gpu_batch, host_batch, materialize_ms, blocking_h2d_ms)."""

        if self.position >= len(self.batches):
            raise StopIteration
        start = time.perf_counter()
        batch = materialize_host_batch(self.corpus, self.batches[self.position], self.pad_token_id)
        batch = pin_host_batch(batch, self.pin)
        materialize_ms = (time.perf_counter() - start) * 1000
        copy_start = time.perf_counter()
        gpu_batch = to_device_batch(batch, self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        h2d_ms = (time.perf_counter() - copy_start) * 1000
        self.position += 1
        return gpu_batch, batch, materialize_ms, h2d_ms

    def shutdown(self) -> None:
        pass


class PrefetchBatchLoader:
    """Variant: worker thread plus side-stream double-buffered async H2D."""

    def __init__(
        self,
        corpus: ShardCorpus,
        batches: Sequence[Sequence[int]],
        pad_token_id: int,
        queue_depth: int = 2,
        pin: bool = True,
    ) -> None:
        self.worker = PrefetchWorker(corpus, batches, pad_token_id, queue_depth, pin)
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.current: Optional[HostBatch] = None
        self.current_event: Optional[torch.cuda.Event] = None
        self._gpu: Optional[dict[str, torch.Tensor]] = None
        self.position = 0
        self.queue_depth = queue_depth
        self.pin = pin
        self.device: Optional[torch.device] = None

    @property
    def batches(self) -> Sequence[Sequence[int]]:
        return self.worker.batches

    def start(self, device) -> None:
        self.device = torch.device(device)
        if device.type == "cuda":
            self.copy_stream = torch.cuda.Stream(device)
        self.worker.start()
        self._issue(self._next_host())

    def _next_host(self) -> HostBatch:
        item = self.worker.queue.get()
        if item is None:
            self.worker.check_error()
            raise StopIteration
        self.worker.check_error()
        return item

    def _issue(self, batch: HostBatch) -> None:
        """Copy the batch on the side stream without blocking the CPU."""

        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                gpu_batch = {
                    name: getattr(batch, name).to(self.device, non_blocking=True)
                    for name in BATCH_FIELDS
                }
            event = torch.cuda.Event()
            event.record(self.copy_stream)
        else:
            gpu_batch = to_device_batch(batch, self.device)
            event = None
        self.current = batch
        self._gpu = gpu_batch
        self.current_event = event

    def next_batch(self) -> tuple[dict[str, torch.Tensor], HostBatch, float, float]:
        """Return the pre-issued batch; issue the copy for the next one.

        Returns (gpu_batch, host_batch, exposed_copy_ms, supply_ms).  The
        exposed copy is the residual time of the H2D issued one step earlier
        (event synchronize); the supply time is the queue wait for the batch
        after next.  Both are expected to be near zero when the overlap works.
        """

        if self.current is None or self.position >= len(self.worker.batches):
            raise StopIteration
        copy_start = time.perf_counter()
        if self.current_event is not None:
            self.current_event.synchronize()
        exposed_ms = (time.perf_counter() - copy_start) * 1000
        supply_ms = 0.0
        returned_gpu = self._gpu
        returned_host = self.current
        if self.position + 1 < len(self.worker.batches):
            supply_start = time.perf_counter()
            nxt = self._next_host()
            supply_ms = (time.perf_counter() - supply_start) * 1000
            self._issue(nxt)
        self.position += 1
        return returned_gpu, returned_host, exposed_ms, supply_ms

    def shutdown(self) -> None:
        while True:
            try:
                if self.worker.queue.get_nowait() is None:
                    break
            except queue.Empty:
                break
