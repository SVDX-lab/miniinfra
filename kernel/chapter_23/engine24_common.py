# -*- coding: utf-8 -*-
"""chapter_23 共用工具：JSON 落盘、时间戳、峰值显存、prompt 构造。

复用口径声明：输入 token id 与 Greedy 真值全部来自 chapter_01/truth（只读），
本文件不生成任何 fixtures。计时统一 wall-clock（perf_counter），
显存统一 torch 私有加速器 allocator 口径（musa 上为 torch.musa.*）。
"""
import json
import os
import time
from datetime import datetime, timezone

# 官方 torch 的 fork 守卫：父进程触碰 CUDA 后 fork EngineCore 会报
# "Cannot re-initialize CUDA in forked subprocess"（torch 2.14 实测；
# torch_musa 无此守卫）。统一 spawn 规避——必须在 import vllm 之前生效。
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%FT%TZ")


def dump_json(obj, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    return path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class AccelMem:
    """私有加速器设备内存口径（musa: torch.musa / cuda: torch.cuda 自动适配）。"""

    def __init__(self):
        import torch
        if hasattr(torch, "musa") and torch.musa.is_available():
            self.mod = torch.musa
            self.device_type = "musa"
        elif torch.cuda.is_available():
            self.mod = torch.cuda
            self.device_type = "cuda"
        else:
            self.mod = None
            self.device_type = "cpu"

    def reset_peak(self):
        if self.mod is not None:
            self.mod.reset_peak_memory_stats()

    def snapshot(self):
        if self.mod is None:
            return {"device_type": "cpu", "peak_allocated_bytes": 0, "reserved_bytes": 0}
        return {
            "device_type": self.device_type,
            "peak_allocated_bytes": self.mod.max_memory_allocated(),
            "peak_allocated_MiB": round(self.mod.max_memory_allocated() / 2**20, 1),
            "reserved_bytes": self.mod.memory_reserved(),
            "reserved_MiB": round(self.mod.memory_reserved() / 2**20, 1),
            "device_total_MiB": round(
                self.mod.get_device_properties(0).total_memory / 2**20, 1
            ),
        }


def build_prompts(truth_ids, target_len, batch):
    """构造测量用 prompt_token_ids。

    p20 直接用 ch01 truth 的 20 token（含 chat template）。
    更长 prompt = truth 模板头 + 内容 token 循环填充到 target_len
    （填充源 = truth prompt 的内容 token，避开 EOS/BOS 特殊 token），
    保证确定性且可复现（同一函数在任何机器上产生同一 id 序列）。
    """
    content = [t for t in truth_ids if t < 151643]  # 去掉 chat 模板特殊 token
    if target_len <= len(truth_ids):
        base = list(truth_ids[:target_len])
    else:
        base = list(truth_ids)
        i = 0
        while len(base) < target_len:
            base.append(content[i % len(content)])
            i += 1
    return [{"prompt_token_ids": base} for _ in range(batch)]


class Timer:
    def __init__(self):
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
