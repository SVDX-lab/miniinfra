#!/usr/bin/env python3
"""第 09 期冒烟：边界行为验证（正式基准前的最小闭环）。

覆盖四类边界：
  1. Graph 捕获/回放最小链（K=8 add）与 eager 位级对账；
  2. 正常 shape 抽查的最小版（rows=256）图捕获（含捕获期分配进 pool）；
  3. fixtures 内联生成路径与落盘文件位级一致（协议 §2）；
  4. decode 步最小版（seq=22）图捕获 + argmax 对账（--with-model 时）。

输出：results/smoke-test-{backend}.json；任何一项失败即退出码 1。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHAPTER = HERE
for _path in (str(CHAPTER), str(CHAPTER / "implementations")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import (  # noqa: E402
    GraphCaptureCompat,
    backend_module,
    backend_sync,
    resolve_device,
    write_results_json,
)

SCHEMA_VERSION = "kernel-chapter10.smoke.v1"


def check1_chain_graph(device) -> dict:
    module = backend_module(device)
    x = torch.arange(1024, device=device, dtype=torch.float16).clamp(-100, 100)
    t_eager = x.clone()
    for _ in range(8):
        t_eager.add_(1.0)
    t_ref = t_eager.clone()
    side_stream = module.Stream()
    current = module.current_stream()
    side_stream.wait_stream(current)
    t_warm = x.clone()
    with module.stream(side_stream):
        for _ in range(3):
            t_warm.add_(1.0)
    current.wait_stream(side_stream)
    backend_sync(device)
    graph = module.CUDAGraph()
    t_static = x.clone()
    with module.graph(graph):
        for _ in range(8):
            t_static.add_(1.0)
    backend_sync(device)
    graph.replay()
    backend_sync(device)
    equal = bool(torch.equal(t_static, t_ref))
    return {"pass": equal, "max_abs": float((t_static.float() - t_ref.float()).abs().max().item())}


def check2_spot_graph_capture(device) -> dict:
    module = backend_module(device)
    rows = 256
    generator = torch.Generator().manual_seed(20261121 + rows * 100000 + 1024)
    r = torch.randn(rows, 1024, generator=generator).to(device=device, dtype=torch.float16)
    h = torch.randn(rows, 1024, generator=generator).to(device=device, dtype=torch.float16)
    w = torch.randn(1024, generator=generator).to(device=device, dtype=torch.float16)

    def chain():
        t = r + h
        float_states = t.float()
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_states * torch.rsqrt(variance + 1e-6)
        return w * normalized.to(t.dtype)

    side_stream = module.Stream()
    current = module.current_stream()
    side_stream.wait_stream(current)
    with module.stream(side_stream):
        for _ in range(3):
            chain()
    current.wait_stream(side_stream)
    backend_sync(device)
    ref = chain()
    backend_sync(device)
    graph = module.CUDAGraph()
    with module.graph(graph):
        captured = chain()
    backend_sync(device)
    graph.replay()
    backend_sync(device)
    equal = bool(torch.equal(captured, ref))
    max_abs = float((captured.float() - ref.float()).abs().max().item())
    del graph
    return {"pass": equal, "max_abs": max_abs}


def check3_fixtures() -> dict:
    fixtures = HERE / "fixtures"
    spot_path = fixtures / "spot-rmsnorm-2048x1024.pt"
    decode_path = fixtures / "decode-step-inputs.json"
    if not spot_path.is_file() or not decode_path.is_file():
        return {"pass": False, "error": "fixtures 缺失，先跑 generate_fixtures.py"}
    payload = torch.load(spot_path, weights_only=False)
    seed = payload["seed"]
    generator = torch.Generator().manual_seed(seed)
    r = torch.randn(2048, 1024, generator=generator).to(torch.float16)
    h = torch.randn(2048, 1024, generator=generator).to(torch.float16)
    w = torch.randn(1024, generator=generator).to(torch.float16)
    bitwise = bool(torch.equal(payload["r"], r) and torch.equal(payload["h"], h)
                   and torch.equal(payload["w"], w))
    decode_inputs = json.loads(decode_path.read_text(encoding="utf-8"))
    expected = decode_inputs.pop("inputs_sha256")
    recomputed = hashlib.sha256(
        json.dumps(decode_inputs, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"pass": bitwise and recomputed == expected,
            "spot_bitwise": bitwise, "decode_sha256_match": recomputed == expected}


def check4_decode_graph(device, model_dir: str, truth_dir: str) -> dict:
    sys.path.insert(0, str(CHAPTER / "implementations"))
    import qwen3_model_fused as model_lib  # noqa: E402
    from implementations.backend_setup import setup_backend  # noqa: E402

    module = backend_module(device)
    model = model_lib.load_handwritten_model(model_dir, device, dtype=torch.float16).eval()
    setup = setup_backend(device.type if device.type in ("cuda", "musa") else "cuda")
    model.model.fused_ops = setup["router"]
    compat = GraphCaptureCompat(device, setup.get("pybind_module"))

    truth = json.loads((Path(truth_dir) / "truth-manifest.json").read_text(encoding="utf-8"))
    seq = 22
    ids = (truth["prompt_token_ids"] + truth["generated_token_ids"])[:seq]
    static_ids = torch.tensor([ids], dtype=torch.long, device=device)

    def step():
        logits = model(static_ids)
        return logits[:, -1, :].argmax(dim=-1, keepdim=True)

    side_stream = module.Stream()
    current = module.current_stream()
    side_stream.wait_stream(current)
    with module.stream(side_stream):
        for _ in range(3):
            step()
    current.wait_stream(side_stream)
    backend_sync(device)
    eager_next = int(step()[0].item())
    backend_sync(device)

    graph = module.CUDAGraph()
    with compat.capture(graph):
        captured_next = step()
    backend_sync(device)
    graph.replay()
    backend_sync(device)
    graph_next = int(captured_next[0].item())
    return {"pass": eager_next == graph_next,
            "eager_next": eager_next, "graph_next": graph_next,
            "capture_mode": "explicit_stream(musa_v2)" if compat.explicit_stream_mode
                            else "torch_graph_context"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="cuda", choices=["cuda", "musa"])
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--model-dir", default=str(HERE.parent / "chapter_01" / "models" / "qwen3-0.6b"))
    parser.add_argument("--truth-dir", default=str(HERE.parent / "chapter_01" / "truth"))
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    report: dict[str, dict] = {"schema_version": SCHEMA_VERSION, "backend": arguments.backend}
    ok = True
    for name, check in (("1_chain_graph", lambda: check1_chain_graph(device)),
                        ("2_spot_graph_capture", lambda: check2_spot_graph_capture(device)),
                        ("3_fixtures_bitwise", check3_fixtures)):
        try:
            outcome = check()
        except Exception as error:  # noqa: BLE001
            outcome = {"pass": False, "error": f"{type(error).__name__}: {error}"}
        report[name] = outcome
        ok = ok and outcome.get("pass", False)
        print(f"[smoke/{arguments.backend}] {name}: {outcome}")

    if arguments.with_model:
        try:
            outcome = check4_decode_graph(device, arguments.model_dir, arguments.truth_dir)
        except Exception as error:  # noqa: BLE001
            outcome = {"pass": False, "error": f"{type(error).__name__}: {error}"}
        report["4_decode_graph"] = outcome
        ok = ok and outcome.get("pass", False)
        print(f"[smoke/{arguments.backend}] 4_decode_graph: {outcome}")

    report["all_pass"] = ok
    output = Path("results") / f"smoke-test-{arguments.backend}.json"
    write_results_json(output, report)
    return 0 if ok else 1


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print(f"[smoke] 耗时 {time.time() - t0:.1f}s，退出码 {code}")
    raise SystemExit(code)
