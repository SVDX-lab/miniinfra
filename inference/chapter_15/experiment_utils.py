"""第 15 期实验公共入口：模型、Handoff Service、环境与 JSON。"""

import json
import os
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from handoff_protocol import HandoffClient
from qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3Config,
    load_handwritten_model,
    resolve_model_directory,
)


def parse_dtype(value):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError("dtype 必须是 float32、float16 或 bfloat16")
    return mapping[value]


def free_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def handoff_service(port=None):
    port = free_local_port() if port is None else int(port)
    script = Path(__file__).with_name("handoff_server.py")
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    client = HandoffClient(port=port)
    deadline = time.time() + 10
    while time.time() < deadline:
        if process.poll() is not None:
            error = process.stderr.read()
            raise RuntimeError("Handoff Service 启动失败: " + error)
        try:
            client.ping()
            break
        except (ConnectionError, OSError):
            time.sleep(0.05)
    else:
        process.terminate()
        raise RuntimeError("Handoff Service 启动超时")
    try:
        yield client, port
    finally:
        if process.poll() is None:
            try:
                client.shutdown()
                process.wait(timeout=5)
            except Exception:
                process.terminate()
                process.wait(timeout=5)


def load_model(model_name, revision, device, dtype):
    directory = resolve_model_directory(model_name, revision)
    config = Qwen3Config.from_model_directory(directory)
    model = load_handwritten_model(directory, device, dtype=dtype)
    return directory, config, model


def environment_record(model_name, revision, device, dtype):
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "model": model_name,
        "revision": revision,
        "pid": os.getpid(),
    }
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(torch.device(device))
        record["gpu_memory_bytes"] = torch.cuda.get_device_properties(
            torch.device(device)
        ).total_memory
    return record


def write_json(path, value):
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def add_model_arguments(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
