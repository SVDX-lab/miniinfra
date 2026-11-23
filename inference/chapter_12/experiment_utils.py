"""第 12 期实验辅助：环境记录、固定负载生成与结果保存。"""

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch


def seed_everything(seed):
    import random

    random.seed(seed)
    torch.manual_seed(seed)


def collect_environment(device):
    cpu_model = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    memory_gib = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("MemTotal:"):
                    memory_gib = int(line.split()[1]) / 1024**2
                    break
    except OSError:
        pass
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "processor": cpu_model,
        "memory_gib": memory_gib,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
    }
    if device.type == "cuda":
        environment["gpu_name"] = torch.cuda.get_device_name(device)
        environment["gpu_capability"] = ".".join(
            map(str, torch.cuda.get_device_capability(device))
        )
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,pcie.link.gen.current,"
                    "pcie.link.width.current,pcie.link.gen.max,"
                    "pcie.link.width.max",
                    "--format=csv,noheader",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if query.returncode == 0:
                fields = [item.strip() for item in query.stdout.strip().split(",")]
                environment["driver_version"] = fields[0]
                if len(fields) >= 3:
                    environment["pcie_link_gen"] = fields[1]
                    environment["pcie_link_width"] = fields[2]
                if len(fields) >= 5:
                    environment["pcie_link_gen_max"] = fields[3]
                    environment["pcie_link_width_max"] = fields[4]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return environment


def synthesize_workload(vocab_size, lengths, max_new_tokens, seed, arrivals=None):
    """生成固定合成 Token 序列：与 tokenizer/模型内容无关，可完全复现。"""
    generator = torch.Generator().manual_seed(seed)
    sequences = []
    for length in lengths:
        tokens = torch.randint(
            low=10, high=vocab_size, size=(length,), generator=generator
        )
        sequences.append([int(token) for token in tokens])
    if arrivals is None:
        arrivals = [float(index) * 40.0 for index in range(len(lengths))]
    return sequences, arrivals


def save_results(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print("结果已写入 %s" % target)


def format_ms(value):
    return "%.2f ms" % value if value is not None else "n/a"
