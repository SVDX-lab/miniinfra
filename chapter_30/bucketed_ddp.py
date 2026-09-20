"""Chapter 30 core: gradient buckets that start AllReduce inside Backward.

Baseline (Chapter 29 body): the whole backward pass finishes, then one
AllReduce moves every FP32 gradient — the communication is fully exposed.
This module changes ONLY when the AllReduce starts: parameters are grouped
into buckets in reverse ``model.parameters()`` order (gradients become ready
roughly in that order), and the moment every parameter of a bucket has its
gradient, the bucket's flat buffer is AllReduced with ``async_op=True`` while
the backward continues.

Mechanics per step:

1. Buckets and their flat FP32 buffers are built once (param granularity, no
   parameter is split across buckets — real DDP does split; noted honestly).
2. A ``register_post_accumulate_grad_hook`` on each parameter copies the fresh
   gradient into its bucket slice, drops the original storage, and counts the
   bucket member as ready.
3. When a bucket is complete its flat buffer is AllReduced asynchronously;
   a CUDA event pair records the per-bucket communication duration.
4. After ``backward()`` returns, the tail waits for the outstanding handles in
   firing order and re-points ``param.grad`` at the reduced bucket slice (a
   view — no copy). Clip and the optimizer then see the same SUM-reduced
   gradients as the flat path.

Timing scope: ``backward_ms`` now contains overlapped communication,
``tail_wait_ms`` is the exposed remainder after backward ends,
``total_comm_ms`` is the sum of per-bucket AllReduce durations. The inferred
hidden part ``total_comm - tail_wait`` is an upper bound on successfully
overlapped communication. Buckets are reused across steps; their buffers are
the extra temporary memory this mechanism costs.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from mini_batch import batch_loss
from ddp_core import to_device_batch


@dataclass
class Bucket:
    index: int
    flat: torch.Tensor
    members: list  # list of (parameter, offset, shape)
    handles: list = field(default_factory=list)
    start_events: list = field(default_factory=list)
    end_events: list = field(default_factory=list)
    ready_count: int = 0
    fired: bool = False

    @property
    def size_bytes(self) -> int:
        return self.flat.numel() * self.flat.element_size()


def build_buckets(model: torch.nn.Module, bucket_bytes: float) -> list[Bucket]:
    """Group parameters in reverse order into buckets of at most bucket_bytes.

    Reverse order matches the direction in which gradients become ready
    during Backward, so the first bucket is typically the first that can
    fire. Parameters are never split across buckets; a single parameter
    larger than the target forms its own bucket.
    """

    if bucket_bytes <= 0:
        raise ValueError("bucket_bytes must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    device = parameters[0].device
    buckets: list[Bucket] = []
    current: list[tuple[torch.nn.Parameter, int, torch.Size]] = []
    offset = 0
    for parameter in reversed(parameters):
        if current and (offset + parameter.numel()) * parameter.element_size() > bucket_bytes:
            flat = torch.zeros(sum(p.numel() for p, _, _ in current), dtype=torch.float32, device=device)
            buckets.append(Bucket(index=len(buckets), flat=flat, members=list(current)))
            current = []
            offset = 0
        current.append((parameter, offset, parameter.shape))
        offset += parameter.numel()
    if current:
        flat = torch.zeros(offset, dtype=torch.float32, device=device)
        buckets.append(Bucket(index=len(buckets), flat=flat, members=list(current)))

    bucket_of: dict[int, tuple[Bucket, int, torch.Size]] = {}
    total_members = 0
    for bucket in buckets:
        for parameter, bucket_offset, shape in bucket.members:
            key = id(parameter)
            if key in bucket_of:
                raise AssertionError("a parameter must live in exactly one bucket")
            bucket_of[key] = (bucket, bucket_offset, shape)
            total_members += 1
    if total_members != len(parameters):
        raise AssertionError("bucket coverage must include every trainable parameter")
    return buckets


class BucketCoordinator:
    """Owns the hooks and the tail wait for one model across many steps."""

    def __init__(self, model: torch.nn.Module, bucket_bytes: float, world_size: int):
        self.model = model
        self.world_size = world_size
        self.buckets = build_buckets(model, bucket_bytes)
        self.hooks = []
        self.install_hooks()

    def install_hooks(self) -> None:
        if self.hooks:
            return
        for bucket in self.buckets:
            for parameter, bucket_offset, shape in bucket.members:
                self.hooks.append(parameter.register_post_accumulate_grad_hook(self._make_hook(bucket, parameter, bucket_offset, shape)))

    def _make_hook(self, bucket: Bucket, parameter: torch.nn.Parameter, bucket_offset: int, shape):
        slice_view = bucket.flat[bucket_offset : bucket_offset + parameter.numel()].view(shape)
        use_events = bucket.flat.is_cuda

        def hook(parameter: torch.nn.Parameter) -> None:
            if bucket.fired:
                raise AssertionError("a bucket filled twice in one backward")
            slice_view.copy_(parameter.grad)
            parameter.grad = None
            bucket.ready_count += 1
            if bucket.ready_count == len(bucket.members):
                bucket.ready_count = 0
                bucket.fired = True
                start = torch.cuda.Event(enable_timing=True) if use_events else None
                end = torch.cuda.Event(enable_timing=True) if use_events else None
                if use_events:
                    start.record()
                bucket.handles.append(dist.all_reduce(bucket.flat, op=dist.ReduceOp.SUM, async_op=True))
                if use_events:
                    bucket.start_events.append(start)
                    bucket.end_events.append(end)
                    end.record()

        return hook

    def wait_and_attach(self) -> dict:
        """Wait for outstanding AllReduces and re-attach gradients as views.

        Returns the segment timing record for this backward pass. On CPU the
        per-bucket durations stay at zero — the smoke path only checks values.
        """

        use_events = self.buckets[0].flat.is_cuda
        if use_events:
            tail_start = torch.cuda.Event(enable_timing=True)
            tail_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            tail_start.record()
        else:
            wall_start = time.perf_counter()
        fired = 0
        total_comm_ms = 0.0
        per_bucket_ms = []
        for bucket in self.buckets:
            if not bucket.fired:
                raise AssertionError(f"bucket {bucket.index} never fired; hooks missed gradients")
            handle = bucket.handles.pop(0)
            handle.wait()
            if use_events:
                start_event, end_event = bucket.start_events.pop(0), bucket.end_events.pop(0)
                duration = float(start_event.elapsed_time(end_event))
            else:
                duration = 0.0
            total_comm_ms += duration
            per_bucket_ms.append(duration)
            for parameter, bucket_offset, shape in bucket.members:
                parameter.grad = bucket.flat[bucket_offset : bucket_offset + parameter.numel()].view(shape)
            bucket.fired = False
            fired += 1
        if use_events:
            tail_end.record()
            torch.cuda.synchronize()
            tail_ms = float(tail_start.elapsed_time(tail_end))
        else:
            tail_ms = (time.perf_counter() - wall_start) * 1000
        return {
            "tail_wait_ms": tail_ms,
            "total_comm_ms": total_comm_ms,
            "per_bucket_comm_ms": per_bucket_ms,
            "buckets_fired": fired,
        }

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def _telemetry(rank: int) -> dict[str, float]:
    import os

    local_id = int(os.environ.get("LOCAL_RANK", rank))
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm", "--format=csv,noheader,nounits", f"--id={local_id}"],
        check=True, capture_output=True, text=True,
    )
    power, temperature, clock = result.stdout.strip().split(",")
    return {"power_watts": float(power), "temperature_celsius": float(temperature), "sm_clock_mhz": float(clock)}


def bucketed_train_step(
    model: torch.nn.Module,
    coordinator: BucketCoordinator,
    optimizer: torch.optim.Optimizer,
    host_batch,
    loss_scale: float,
    device: torch.device,
    max_grad_norm: float,
    world_size: int,
    rank: int,
    collect_telemetry: bool = True,
) -> dict[str, float]:
    """One optimizer step with bucketed in-backward AllReduce.

    Segment boundaries match the Chapter 29 body so arms stay comparable:
    ``grad_allreduce_ms`` here reports the EXPOSED part only (tail wait), and
    ``total_comm_ms`` the sum of all bucket AllReduce durations.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    events = {name: torch.cuda.Event(enable_timing=True) for name in ("h2d_start", "h2d_end", "forward_end", "backward_end", "optimizer_end")}
    events["h2d_start"].record()
    batch = to_device_batch(host_batch, device)
    events["h2d_end"].record()
    logits = model(batch.input_ids, attention_mask=batch.attention_mask)
    loss = batch_loss(logits, batch.labels, "token_mean") * loss_scale
    events["forward_end"].record()
    loss.backward()
    events["backward_end"].record()
    tail = coordinator.wait_and_attach()
    from ddp_core import all_reduce_scalars

    global_loss = all_reduce_scalars(float(loss.detach().item()), world_size, device)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    events["optimizer_end"].record()
    torch.cuda.synchronize(device)
    step_ms = (time.perf_counter() - wall_start) * 1000
    loss_value = float(loss.detach().item())
    metrics = {
        "h2d_time_ms": float(events["h2d_end"].elapsed_time(events["forward_end"])),
        "forward_time_ms": float(events["forward_end"].elapsed_time(events["backward_end"])) - 0.0,
        "backward_time_ms": float(events["backward_end"].elapsed_time(events["backward_end"])) if False else None,
    }
    # forward/backward split: h2d_end->forward_end covers transfer+forward,
    # forward_end->backward_end covers backward WITH overlapped communication.
    metrics = {
        "h2d_time_ms": float(events["h2d_end"].elapsed_time(events["forward_end"])),
        "forward_time_ms": float(events["h2d_end"].elapsed_time(events["forward_end"])),
        "backward_time_ms": float(events["forward_end"].elapsed_time(events["backward_end"])),
        "grad_allreduce_ms": tail["tail_wait_ms"],
        "total_comm_ms": tail["total_comm_ms"],
        "buckets_fired": float(tail["buckets_fired"]),
        "scalar_sync_ms": 0.0,
        "optimizer_time_ms": float(events["backward_end"].elapsed_time(events["optimizer_end"])) - tail["tail_wait_ms"],
        "step_time_ms": step_ms,
    }
    metrics.update({
        "local_loss": loss_value,
        "global_loss": global_loss,
        "gradient_norm_before_clip": float(grad_norm.item()),
        "physical_tokens": float(batch.physical_tokens),
        "non_padding_tokens": float(batch.non_padding_tokens),
        "supervised_tokens": float(batch.supervised_tokens),
        "padding_tokens": float(batch.padding_tokens),
        "padding_ratio": batch.padding_ratio,
        "batch_max_length": float(max(batch.lengths)),
        "samples": float(batch.batch_size),
    })
    seconds = metrics["step_time_ms"] / 1000
    metrics.update({
        "physical_tokens_per_second": batch.physical_tokens / seconds,
        "non_padding_tokens_per_second": batch.non_padding_tokens / seconds,
        "supervised_tokens_per_second": batch.supervised_tokens / seconds,
    })
    if device.type == "cuda" and collect_telemetry:
        metrics.update(_telemetry(rank))
    return metrics
