#!/usr/bin/env python3
"""第 09 期实验 3：动态 Batch/长度变化时的重新捕获成本与显存驻留。

场景（贴近真实 decode 的 shape 演化）：
  A. 序列长度梯度 recapture（私有 pool，图保持存活）：seq ∈ {22,26,...,46}
     ——真实 decode 每步长度 +1（无 KV Cache 手写模型每步新 shape），
     Graph 只对定长 shape 复用，量化"每新 shape 的一次性成本 + 常驻显存"。
  B. 共享 graph pool（graph_pool_handle，vLLM 生产用法）同梯子：所有图共用
     一个 mempool，交替回放可复用内存块——与 A 的总驻留对照。
  C. batch 梯度 recapture：batch ∈ {1,2,4} @ seq=46（私有 pool）。

口径要点（首版教训，见 EXPERIMENT_LOG）：
  - 图必须保持存活（返回后由调用方持有引用）——首版图在函数返回时被
    回收，驻留曲线被 allocator 回收行为污染（出现负 delta）；
  - 每次捕获前 empty_cache 只归还"未占用缓存块"，存活图的 pool 块不受
    影响——warmup 的一次性分配不污染快照；
  - 记录绝对 reserved 值（曲线可审计），差分只是派生量。

每张图测：warmup / capture+instantiate / 首次回放 / 稳态回放 / argmax 对账。
输出：results/recapture-memory-{backend}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1]
for _path in (str(CHAPTER), str(CHAPTER / "implementations"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiment_utils import (  # noqa: E402
    GraphCaptureCompat,
    backend_module,
    backend_sync,
    device_environment_record,
    memory_snapshot,
    resolve_device,
    write_results_json,
)

SCHEMA_VERSION = "kernel-chapter10.recapture-memory.v2"
DEFAULT_BACKEND = "cuda"
MODEL_DEFAULT = CHAPTER.parent / "chapter_01" / "models" / "qwen3-0.6b"
TRUTH_DEFAULT = CHAPTER.parent / "chapter_01" / "truth"
WARMUP_STEPS = 3


def empty_cache(device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "musa":
        torch.musa.empty_cache()


def capture_forward(world, ids_list: list[int], batch: int, pool=None,
                    clear_cache: bool = True, compat=None):
    """对给定 (seq, batch) 捕获一张 decode 前向图（fused 路径）。

    返回 (record, graph)：graph 由调用方持有引用（keep-alive），显存快照在
    图存活期间完成。
    """
    model = world["model"]
    device = world["device"]
    module = backend_module(device)
    seq = len(ids_list)
    static_ids = torch.tensor([ids_list] * batch, dtype=torch.long, device=device)
    fuse = model.model.fused_ops

    def forward_step():
        logits = model(static_ids)
        return logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # 捕获前 warmup（旁路流，vLLM 同款三步；cuBLAS workspace 等惰性初始化）
    side_stream = module.Stream()
    current = module.current_stream()
    side_stream.wait_stream(current)
    t_w0 = time.perf_counter()
    with module.stream(side_stream):
        for _ in range(WARMUP_STEPS):
            forward_step()
    current.wait_stream(side_stream)
    backend_sync(device)
    warmup_ms = (time.perf_counter() - t_w0) * 1000.0

    eager_next = forward_step()
    backend_sync(device)
    eager_next_value = int(eager_next[0].item())

    if clear_cache:
        empty_cache(device)
    before = memory_snapshot(device)

    graph = module.CUDAGraph()
    t0 = time.perf_counter()
    try:
        if compat is not None and compat.supported:
            with compat.capture(graph, pool=pool):
                captured_next = forward_step()
        else:
            kwarg = {"pool": pool} if pool is not None else {}
            with module.graph(graph, **kwarg):
                captured_next = forward_step()
    except TypeError:
        # 旧版 torch_musa 的 graph 上下文可能不收 pool/stream 参数
        with module.graph(graph):
            captured_next = forward_step()
    capture_ms = (time.perf_counter() - t0) * 1000.0
    after = memory_snapshot(device)

    t1 = time.perf_counter()
    graph.replay()
    backend_sync(device)
    first_replay_ms = (time.perf_counter() - t1) * 1000.0

    next_value = int(captured_next[0].item())
    t2 = time.perf_counter()
    for _ in range(20):
        graph.replay()
    backend_sync(device)
    steady_ms = (time.perf_counter() - t2) * 1000.0 / 20

    record = {
        "seq": seq,
        "batch": batch,
        "warmup_ms": round(warmup_ms, 3),
        "capture_plus_instantiate_ms": round(capture_ms, 3),
        "first_replay_ms": round(first_replay_ms, 3),
        "steady_replay_ms": round(steady_ms, 4),
        "reserved_before_bytes": before["reserved_bytes_all_current"],
        "reserved_after_bytes": after["reserved_bytes_all_current"],
        "reserved_delta_bytes": (after["reserved_bytes_all_current"] or 0)
                                - (before["reserved_bytes_all_current"] or 0),
        "allocated_before_bytes": before["allocated_bytes_all_current"],
        "allocated_after_bytes": after["allocated_bytes_all_current"],
        "next_token_equal_eager": next_value == eager_next_value,
        "next_token_eager": eager_next_value,
        "next_token_graph": next_value,
        "fused_at_capture": fuse is not None,
    }
    return record, graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=["cuda", "musa"])
    parser.add_argument("--model-dir", default=str(MODEL_DEFAULT))
    parser.add_argument("--truth-dir", default=str(TRUTH_DEFAULT))
    parser.add_argument("--seq-step", type=int, default=4)
    parser.add_argument("--max-seq", type=int, default=46)
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    module = backend_module(device)
    module.empty_cache()

    sys.path.insert(0, str(CHAPTER / "implementations"))
    import qwen3_model_fused as model_lib  # noqa: E402
    from implementations.backend_setup import setup_backend  # noqa: E402

    model = model_lib.load_handwritten_model(
        arguments.model_dir, device, dtype=torch.float16).eval()
    setup = setup_backend(arguments.backend)
    model.model.fused_ops = setup["router"]

    truth = json.loads((Path(arguments.truth_dir) / "truth-manifest.json").read_text(encoding="utf-8"))
    prompt_ids = truth["prompt_token_ids"]
    generated_ids = truth["generated_token_ids"]

    world = {"model": model, "device": device}
    compat = GraphCaptureCompat(device, setup.get("pybind_module"))
    results = {
        "schema_version": SCHEMA_VERSION,
        "backend": arguments.backend,
        "model_dtype": "float16",
        "warmup_steps_per_capture": WARMUP_STEPS,
        "pool_note": (
            "捕获期分配进 graph pool，图存活期间驻留；每图快照在其存活期间完成；"
            "keep-alive 模式下所有图保持存活（贴近 vLLM 每shape一张图的部署形态），"
            "总驻留 = 末图 reserved_after − 基线（首图捕获前 empty_cache 后）"),
    }

    def ids_for(seq: int) -> list[int]:
        ids = (prompt_ids + generated_ids)[:seq]
        if len(ids) < seq:  # 真值生成不足时用 prompt token 平铺补齐（shape 实验与值无关）
            ids = ids + prompt_ids[: seq - len(ids)]
        return ids

    # A. 序列长度梯度（私有 pool，全部 keep-alive）
    seq_ladder = list(range(22, arguments.max_seq + 1, arguments.seq_step))
    empty_cache(device)
    baseline_a = memory_snapshot(device)["reserved_bytes_all_current"]
    seq_results = []
    keep_alive_private = []
    for seq in seq_ladder:
        record, graph = capture_forward(world, ids_for(seq), batch=1, pool=None, compat=compat)
        keep_alive_private.append(graph)
        seq_results.append(record)
        print(f"[recapture/{arguments.backend}] seq={seq}: capture={record['capture_plus_instantiate_ms']:.1f}ms "
              f"warmup={record['warmup_ms']:.1f}ms 首回放={record['first_replay_ms']:.3f}ms "
              f"稳态回放={record['steady_replay_ms']:.3f}ms "
              f"pool驻留累计={(record['reserved_after_bytes'] - baseline_a) / 1e6:.1f}MB "
              f"argmax一致={record['next_token_equal_eager']}")
    results["seq_recapture_private_pool"] = {
        "seq_ladder": seq_ladder,
        "baseline_reserved_bytes": baseline_a,
        "captures": seq_results,
        "total_capture_plus_instantiate_ms": round(
            sum(r["capture_plus_instantiate_ms"] for r in seq_results), 3),
        "per_capture_ms_mean": round(
            sum(r["capture_plus_instantiate_ms"] for r in seq_results) / len(seq_results), 3),
        "total_pool_reserved_bytes": seq_results[-1]["reserved_after_bytes"] - baseline_a,
        "per_capture_pool_mb_mean": round(
            (seq_results[-1]["reserved_after_bytes"] - baseline_a) / len(seq_results) / 1e6, 2),
    }

    # B. 共享 pool 策略（vLLM 用法）：先释放 A 的图，再同梯子重捕
    del keep_alive_private
    empty_cache(device)
    shared_pool = None
    try:
        shared_pool = module.graph_pool_handle()
    except Exception as error:  # noqa: BLE001
        results["shared_pool_probe"] = f"graph_pool_handle 不可用: {type(error).__name__}: {error}"
    if shared_pool is not None:
        baseline_b = memory_snapshot(device)["reserved_bytes_all_current"]
        shared_results = []
        keep_alive_shared = []
        for seq in seq_ladder:
            record, graph = capture_forward(world, ids_for(seq), batch=1, pool=shared_pool, compat=compat)
            keep_alive_shared.append(graph)
            shared_results.append(record)
            print(f"[recapture-shared/{arguments.backend}] seq={seq}: "
                  f"capture={record['capture_plus_instantiate_ms']:.1f}ms "
                  f"pool驻留累计={(record['reserved_after_bytes'] - baseline_b) / 1e6:.1f}MB "
                  f"argmax一致={record['next_token_equal_eager']}")
        total_shared = shared_results[-1]["reserved_after_bytes"] - baseline_b
        results["seq_recapture_shared_pool"] = {
            "seq_ladder": seq_ladder,
            "baseline_reserved_bytes": baseline_b,
            "captures": shared_results,
            "total_pool_reserved_bytes": total_shared,
            "private_pool_total_for_compare": results["seq_recapture_private_pool"]["total_pool_reserved_bytes"],
            "sharing_saving_bytes": results["seq_recapture_private_pool"]["total_pool_reserved_bytes"] - total_shared,
            "note": "共享 pool 总驻留低于私有求和 = 复用生效；相等/更高 = 复用失效（重要发现照实记录）",
        }
        del keep_alive_shared
        empty_cache(device)

    # C. batch 梯度 @ max_seq（私有 pool，keep-alive）
    baseline_c = memory_snapshot(device)["reserved_bytes_all_current"]
    batch_results = []
    keep_alive_batch = []
    for batch in (1, 2, 4):
        record, graph = capture_forward(world, ids_for(arguments.max_seq), batch=batch, pool=None, compat=compat)
        keep_alive_batch.append(graph)
        batch_results.append(record)
        print(f"[recapture/{arguments.backend}] batch={batch}: capture={record['capture_plus_instantiate_ms']:.1f}ms "
              f"累计驻留={(record['reserved_after_bytes'] - baseline_c) / 1e6:.1f}MB")
    results["batch_recapture"] = {
        "batch_ladder": [1, 2, 4],
        "seq": arguments.max_seq,
        "captures": batch_results,
        "total_pool_reserved_bytes": batch_results[-1]["reserved_after_bytes"] - baseline_c,
    }

    results["graph_capture_compat"] = {
        "supported": compat.supported,
        "explicit_stream_mode": compat.explicit_stream_mode,
    }
    results["environment"] = device_environment_record(device)
    output = Path("results") / f"recapture-memory-{arguments.backend}.json"
    write_results_json(output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
