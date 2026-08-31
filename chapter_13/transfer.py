"""第 13 期核心：同步/异步 KV Block 传输与显式完成语义。"""

import time
from dataclasses import dataclass, field

import torch


TRANSFER_MODES = ("sync", "async")


@dataclass
class TransferTask:
    direction: str
    request_id: str
    mapping: dict
    submit_clock_ms: float
    start_event: object | None = None
    end_event: object | None = None
    submit_wall_ms: float = 0.0
    exposed_wait_ms: float = 0.0
    completed: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def blocks(self):
        return self.mapping["blocks"]


class KVTransferManager:
    """在一条独立 Copy Stream 上提交整请求的 Block 序列。

    `sync` 与 `async` 使用相同的资源预留和完成回调；唯一差别是同步路径在提交后
    立即等待，异步路径由调度器稍后 query/wait。这样主实验不会混入池布局或
    Block 所有权差异。
    """

    def __init__(self, device, mode):
        if mode not in TRANSFER_MODES:
            raise ValueError("transfer mode 必须是 %s" % (TRANSFER_MODES,))
        self.device = device
        self.mode = mode
        self.is_cuda = device.type == "cuda"
        self.copy_stream = None
        self.epoch_event = None
        self.tasks = []
        if self.is_cuda:
            current = torch.cuda.current_stream(device)
            self.copy_stream = torch.cuda.Stream(device=device)
            self.epoch_event = torch.cuda.Event(enable_timing=True)
            self.epoch_event.record(current)
            current.synchronize()

    def _submit(self, direction, gpu_cache, cpu_pool, request_id, clock_ms):
        if direction == "d2h":
            mapping = cpu_pool.prepare_swap_out(gpu_cache, request_id)
        elif direction == "h2d":
            mapping = cpu_pool.prepare_swap_in(gpu_cache, request_id)
        else:
            raise ValueError("未知传输方向: " + direction)

        task = TransferTask(direction, request_id, mapping, clock_ms)
        host_start = time.perf_counter()
        if not self.is_cuda:
            self._copy_blocks(task, gpu_cache, cpu_pool, non_blocking=False)
            task.completed = True
        elif self.mode == "sync":
            stream = torch.cuda.current_stream(self.device)
            task.start_event = torch.cuda.Event(enable_timing=True)
            task.end_event = torch.cuda.Event(enable_timing=True)
            task.start_event.record(stream)
            self._copy_blocks(task, gpu_cache, cpu_pool, non_blocking=False)
            task.end_event.record(stream)
            stream.synchronize()
            task.completed = True
        else:
            compute_stream = torch.cuda.current_stream(self.device)
            source_ready = torch.cuda.Event()
            source_ready.record(compute_stream)
            self.copy_stream.wait_event(source_ready)
            task.start_event = torch.cuda.Event(enable_timing=True)
            task.end_event = torch.cuda.Event(enable_timing=True)
            task.start_event.record(self.copy_stream)
            with torch.cuda.stream(self.copy_stream):
                self._copy_blocks(task, gpu_cache, cpu_pool, non_blocking=True)
            task.end_event.record(self.copy_stream)
        task.submit_wall_ms = (time.perf_counter() - host_start) * 1000
        self.tasks.append(task)
        return task

    @staticmethod
    def _copy_blocks(task, gpu_cache, cpu_pool, non_blocking):
        for gpu_id, cpu_id in zip(
            task.mapping["gpu_ids"], task.mapping["cpu_ids"]
        ):
            if task.direction == "d2h":
                cpu_pool.blocks[cpu_id].copy_(
                    gpu_cache.blocks[gpu_id], non_blocking=non_blocking
                )
            else:
                gpu_cache.blocks[gpu_id].copy_(
                    cpu_pool.blocks[cpu_id], non_blocking=non_blocking
                )

    def submit_swap_out(self, gpu_cache, cpu_pool, request_id, clock_ms):
        return self._submit(
            "d2h", gpu_cache, cpu_pool, request_id, clock_ms
        )

    def submit_swap_in(self, gpu_cache, cpu_pool, request_id, clock_ms):
        return self._submit(
            "h2d", gpu_cache, cpu_pool, request_id, clock_ms
        )

    def query(self, task):
        if task.completed:
            return True
        if not self.is_cuda:
            task.completed = True
        elif task.end_event.query():
            task.completed = True
        return task.completed

    def wait(self, task):
        if self.query(task):
            return 0.0
        start = time.perf_counter()
        task.end_event.synchronize()
        waited = (time.perf_counter() - start) * 1000
        task.exposed_wait_ms += waited
        task.completed = True
        return waited

    def finish(self, task, gpu_cache, cpu_pool, complete_clock_ms):
        if not self.query(task):
            raise RuntimeError("传输尚未完成，不能转移 Block 所有权")
        if task.direction == "d2h":
            cpu_pool.complete_swap_out(gpu_cache, task.request_id)
            event_type = "swap_out"
        else:
            cpu_pool.complete_swap_in(task.request_id)
            event_type = "swap_in"
        mapping = task.mapping
        device_ms = 0.0
        gpu_start_ms = None
        gpu_end_ms = None
        if self.is_cuda:
            device_ms = task.start_event.elapsed_time(task.end_event)
            gpu_start_ms = self.epoch_event.elapsed_time(task.start_event)
            gpu_end_ms = self.epoch_event.elapsed_time(task.end_event)
        exposed_wait_ms = (
            device_ms if self.mode == "sync" else task.exposed_wait_ms
        )
        result = {
            key: value for key, value in mapping.items()
            if key not in ("gpu_ids", "cpu_ids")
        }
        result.update({
            "type": event_type,
            "direction": task.direction,
            "transfer_mode": self.mode,
            "submit_clock_ms": task.submit_clock_ms,
            "complete_clock_ms": complete_clock_ms,
            "submit_wall_ms": task.submit_wall_ms,
            "device_ms": device_ms,
            "exposed_wait_ms": exposed_wait_ms,
            "gpu_start_ms": gpu_start_ms,
            "gpu_end_ms": gpu_end_ms,
            "wall_ms": (
                task.submit_wall_ms if self.mode == "sync"
                else task.exposed_wait_ms
            ),
            **task.metadata,
        })
        result["gb_per_second"] = (
            result["bytes"] / (device_ms / 1000) / 1e9 if device_ms else 0.0
        )
        return result
