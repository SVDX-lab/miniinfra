#!/usr/bin/env python3
"""nsys 取证目标进程：四种 decode 步实现各回放 N 次（NVTX 分段）。

被 run_nsys_decode.sh 以 nsys profile 包裹执行；本文件不输出实验 JSON，
只产生被 trace 的确定性负载（同 seq=46、同输入、固定迭代数），
供 nsys graph trace 读取"图内 kernel 实例"证据（K-13 大纲 §10.4 归因）。

分段：eager_native ×20 / eager_fused ×20 / graph_native_replay ×20 /
      graph_fused_replay ×20 / micro composite_chain ×100 / micro fused ×100
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1]
for _path in (str(CHAPTER), str(CHAPTER / "implementations"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import backend_sync  # noqa: E402

MODEL = CHAPTER.parent / "chapter_01" / "models" / "qwen3-0.6b"
TRUTH = CHAPTER.parent / "chapter_01" / "truth"
REPEATS = 20


def main() -> int:
    device = torch.device("cuda")
    import qwen3_model_fused as model_lib
    from implementations.backend_setup import setup_backend

    model = model_lib.load_handwritten_model(str(MODEL), device, dtype=torch.float16).eval()
    setup = setup_backend("cuda")
    truth = json.loads((TRUTH / "truth-manifest.json").read_text(encoding="utf-8"))
    generated_index = 26
    seq = len(truth["prompt_token_ids"]) + generated_index
    ids = (truth["prompt_token_ids"] + truth["generated_token_ids"])[:seq]
    static_ids = torch.tensor([ids], dtype=torch.long, device=device)

    def step():
        logits = model(static_ids)
        return logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.nvtx.range_push("warmup")
    for _ in range(5):
        step()
    backend_sync(device)
    torch.cuda.nvtx.range_pop()

    graphs = {}
    for name, fused in (("graph_native", None), ("graph_fused", setup["router"])):
        model.model.fused_ops = fused
        side_stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        side_stream.wait_stream(current)
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                step()
        current.wait_stream(side_stream)
        backend_sync(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step()
        backend_sync(device)
        graphs[name] = graph
    model.model.fused_ops = None

    with torch.cuda.nvtx.range("eager_native_x20"):
        for _ in range(REPEATS):
            step()
    backend_sync(device)

    model.model.fused_ops = setup["router"]
    with torch.cuda.nvtx.range("eager_fused_x20"):
        for _ in range(REPEATS):
            step()
    backend_sync(device)
    model.model.fused_ops = None

    for name, graph in graphs.items():
        with torch.cuda.nvtx.range(f"{name}_replay_x20"):
            for _ in range(REPEATS):
                graph.replay()
        backend_sync(device)

    rows = seq
    generator = torch.Generator().manual_seed(20261121 + rows * 100000 + 1024)
    r = torch.randn(rows, 1024, generator=generator).to(device=device, dtype=torch.float16)
    h = torch.randn(rows, 1024, generator=generator).to(device=device, dtype=torch.float16)
    w = torch.randn(1024, generator=generator).to(device=device, dtype=torch.float16)

    def composite():
        t = r + h
        float_states = t.float()
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        return w * (float_states * torch.rsqrt(variance + 1e-6)).to(t.dtype)

    with torch.cuda.nvtx.range("micro_composite_x100"):
        for _ in range(100):
            composite()
    backend_sync(device)
    with torch.cuda.nvtx.range("micro_fused_x100"):
        for _ in range(100):
            setup["router"].fused_add_rmsnorm(h, r, w, 1e-6)
    backend_sync(device)

    print("[nsys-target] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
