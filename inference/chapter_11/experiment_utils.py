"""第 11 期实验入口共用的模型加载、环境记录和计时辅助函数。"""

import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

import torch

from qwen3_model import (
    DEFAULT_DRAFT_MODEL_ID,
    DEFAULT_TARGET_MODEL_ID,
    MODEL_REVISIONS,
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


def load_model(model_name_or_path, device, dtype, revision=None):
    model_directory = resolve_model_directory(model_name_or_path, revision=revision)
    model = load_handwritten_model(
        model_directory,
        device=torch.device(device),
        dtype=DTYPES[dtype],
    )
    return model_directory, model


def load_target_and_draft(
    target_model,
    draft_model,
    device,
    dtype,
    target_revision=None,
    draft_revision=None,
):
    target_directory, target = load_model(
        target_model, device, dtype, revision=target_revision
    )
    draft_directory, draft = load_model(
        draft_model, device, dtype, revision=draft_revision
    )
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError("Target 与 Draft 的词表大小不同")
    return target_directory, target, draft_directory, draft


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def timed_call(function, device):
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - start


def summarize(values):
    if not values:
        raise ValueError("values 不能为空")
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def model_parameter_bytes(model):
    storages = {}
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return sum(storages.values())


def environment_snapshot(
    target_model_path,
    draft_model_path,
    dtype,
    target_model=DEFAULT_TARGET_MODEL_ID,
    draft_model=DEFAULT_DRAFT_MODEL_ID,
):
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": "cpu",
        "target_model": target_model,
        "target_revision": MODEL_REVISIONS.get(target_model),
        "target_path_kind": (
            "directory" if Path(target_model_path).is_dir() else "model_id"
        ),
        "draft_model": draft_model,
        "draft_revision": MODEL_REVISIONS.get(draft_model),
        "draft_path_kind": (
            "directory" if Path(draft_model_path).is_dir() else "model_id"
        ),
        "dtype": dtype,
        "pid": os.getpid(),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "device": properties.name,
                "compute_capability": "%d.%d"
                % (properties.major, properties.minor),
                "total_memory_bytes": properties.total_memory,
            }
        )
    return result


def memory_snapshot():
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def write_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def bytes_to_mib(value):
    return value / (1024 * 1024)
