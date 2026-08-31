"""第 10 期实验入口共用的轻量辅助函数。"""

import json
import os
import platform
import random
from pathlib import Path

import torch

from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name_or_path, device, dtype, attention_backend):
    model_directory = resolve_model_directory(
        model_name_or_path, revision=DEFAULT_MODEL_REVISION
    )
    model = load_handwritten_model(
        model_directory,
        torch.device(device),
        dtype=DTYPES[dtype],
        attention_backend=attention_backend,
    )
    return model_directory, model


def synthetic_prompts(count, length, vocab_size, seed=2026):
    if count < 1 or length < 1:
        raise ValueError("count 和 length 必须大于 0")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    low = min(1000, max(0, vocab_size // 4))
    return [
        torch.randint(
            low,
            vocab_size,
            (length,),
            generator=generator,
            dtype=torch.long,
        ).tolist()
        for _ in range(count)
    ]


def environment_snapshot(model_path, dtype, attention_backend):
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": "cpu",
        "model": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "model_path_kind": "directory" if Path(model_path).is_dir() else "model_id",
        "dtype": dtype,
        "attention_backend": attention_backend,
        "pid": os.getpid(),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update({
            "device": properties.name,
            "compute_capability": "%d.%d" % (
                properties.major, properties.minor
            ),
            "total_memory_bytes": properties.total_memory,
        })
    return result


def write_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def bytes_to_mib(value):
    return value / (1024 * 1024)
