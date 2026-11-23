#!/usr/bin/env python3
"""第 07 期实验 4：PyTorch deterministic 模式开关行为对照（三平台通用）。

大纲实验项："PyTorch deterministic 模式开关的行为对照"。

问题分解：
1. RMSNorm 组合链（x.pow(2).mean → rsqrt → 广播乘 → 乘 w）在开关
   False/True 下，同一输入 100 次运行是否位级一致？（预期：两者都一致
   —— 组合链的每个 kernel 都是固定顺序归约/逐元素，不涉及非确定算子；
   这个"开关无操作"本身就是结论：**原生组合链天然确定但顺序与 CPU 不同**）
2. F.rms_norm（torch≥2.4 融合实现）同样对照 —— 融合 kernel 的顺序与
   组合链不同 → 与组合链互相位级不同但各自确定；
3. 对照组：index_add（scatter 类，CUDA 上默认非确定）—— 证明开关本身
   在本环境生效（False 下 100 次不一致 / True 下报错或变一致）；
4. CPU 侧附加：torch.set_num_threads 变化是否改变组合链位级结果
   （CPU reduce 的分块边界是否随线程数变化）；
5. MUSA：torch_musa 对 use_deterministic_algorithms 的支持情况
   （设置是否成功/报错文本/行为）—— 不支持则记平台边界。

用法：
  python exp_torch_deterministic.py --backend cpu  --out results/...
  python exp_torch_deterministic.py --backend cuda --out results/...
  python exp_torch_deterministic.py --backend musa --out results/...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from experiment_utils import backend_sync, resolve_device  # noqa: E402

SEED = 20260804
ROWS, H = 2048, 1024
REPEATS = 100


def set_deterministic(enabled: bool) -> str:
    """设置 torch.use_deterministic_algorithms 并返回平台行为记录。"""
    import torch
    try:
        torch.use_deterministic_algorithms(enabled)
        return "set_ok"
    except Exception as error:  # noqa: BLE001
        return f"set_failed: {type(error).__name__}: {error}"


def bit_identities_of_repeats(run_once, repeats: int = REPEATS) -> dict:
    """运行 repeats 次，统计与第一次位级相同的次数 + 漂移带（fp32 视图）。

    返回 identical_count / unique_outputs（用张量 bytes 哈希计唯一值）。
    """
    import torch
    first = run_once()
    backend_sync(first.device if hasattr(first, "device") else torch.device("cpu"))
    identical = 0
    hashes = {hashlib.sha256(first.cpu().numpy().tobytes()).hexdigest()}
    max_drift_vs_first = 0.0
    for _ in range(repeats - 1):
        out = run_once()
        backend_sync(out.device)
        if torch.equal(out, first):
            identical += 1
        hashes.add(hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest())
        diff = (out.float() - first.float()).abs().max().item()
        max_drift_vs_first = max(max_drift_vs_first, diff)
    return {
        "repeats": repeats,
        "identical_vs_first": identical,
        "unique_outputs": len(hashes),
        "max_abs_drift_vs_first": max_drift_vs_first,
    }


def composite_rmsnorm(x, w, eps=1e-6):
    """第 01 期同语义组合链（平台原生路径）。"""
    input_dtype = x.dtype
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    normalized = xf * torch.rsqrt(variance + eps)
    return w * normalized.to(input_dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "musa"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    arguments = parser.parse_args()

    device = resolve_device(arguments.backend)
    rng = np.random.default_rng(SEED)
    x_np = rng.standard_normal((ROWS, H), dtype=np.float32)
    w_np = rng.standard_normal(H, dtype=np.float32)
    x = torch.from_numpy(x_np).to(device)
    w = torch.from_numpy(w_np).to(device)
    x16 = x.to(torch.float16)
    w16 = w.to(torch.float16)

    results: dict = {
        "schema_version": "kernel-chapter08.torch-deterministic.v1",
        "backend": arguments.backend,
        "torch_version": torch.__version__,
        "shape": f"{ROWS}x{H}",
        "repeats": arguments.repeats,
        "switch_behavior": {},
        "composite_chain": {},
        "f_rms_norm": {},
        "index_add_control": {},
        "index_put_accumulate": {},
        "matmul_cublas_workspace": {},
        "notes": [],
    }

    # ---- 1. RMSNorm 组合链：开关 False / True × fp32 / fp16 ----
    for switch in (False, True):
        status = set_deterministic(switch)
        results["switch_behavior"][f"set_deterministic({switch})"] = status
        if status != "set_ok" and switch:
            results["notes"].append(
                f"{arguments.backend}: use_deterministic_algorithms(True) 设置失败 —— 平台边界证据")
            break
        for tag, xt, wt in (("fp32", x, w), ("fp16", x16, w16)):
            key = f"{'on' if switch else 'off'}_{tag}"
            results["composite_chain"][key] = bit_identities_of_repeats(
                lambda xt=xt, wt=wt: composite_rmsnorm(xt, wt), arguments.repeats)
            print(f"[chain] switch={'ON ' if switch else 'OFF'} {tag}: "
                  f"identical={results['composite_chain'][key]['identical_vs_first']}"
                  f"/{arguments.repeats} unique={results['composite_chain'][key]['unique_outputs']} "
                  f"drift={results['composite_chain'][key]['max_abs_drift_vs_first']:.2e}")

        # F.rms_norm 融合实现对照（torch>=2.4；不支持记边界）
        try:
            fn_rms = getattr(torch.nn.functional, "rms_norm", None)
            if fn_rms is None:
                results["f_rms_norm"][f"{'on' if switch else 'off'}"] = {"available": False}
            else:
                for tag, xt, wt in (("fp32", x, w), ("fp16", x16, w16)):
                    key = f"{'on' if switch else 'off'}_{tag}"
                    results["f_rms_norm"][key] = bit_identities_of_repeats(
                        lambda fn_rms=fn_rms, xt=xt, wt=wt: fn_rms(xt, (H,), wt, 1e-6),
                        arguments.repeats)
                    results["f_rms_norm"][key]["available"] = True
                    print(f"[f_rms] switch={'ON ' if switch else 'OFF'} {tag}: "
                          f"identical={results['f_rms_norm'][key]['identical_vs_first']}"
                          f"/{arguments.repeats} unique={results['f_rms_norm'][key]['unique_outputs']} "
                          f"drift={results['f_rms_norm'][key]['max_abs_drift_vs_first']:.2e}")
        except Exception as error:  # noqa: BLE001
            results["f_rms_norm"][f"{'on' if switch else 'off'}_error"] = (
                f"{type(error).__name__}: {error}")

        # 组合链 vs F.rms_norm 互相位级差异（一次运行）
        try:
            fn_rms = getattr(torch.nn.functional, "rms_norm", None)
            if fn_rms is not None:
                a = composite_rmsnorm(x, w)
                b = fn_rms(x, (H,), w, 1e-6)
                results["composite_vs_f_rms"] = {
                    "bit_equal": bool(torch.equal(a, b)),
                    "max_abs_diff": float((a.float() - b.float()).abs().max().item()),
                }
        except Exception as error:  # noqa: BLE001
            results["composite_vs_f_rms"] = f"{type(error).__name__}: {error}"

        # ---- 3. index_add 对照组（scatter 类，开关"有牙"探针）----
        def scatter_once():
            src = torch.ones(4096, device=device)
            idx = torch.arange(4096, device=device) % 512
            out = torch.zeros(512, device=device)
            out.index_add_(0, idx, src)
            return out

        try:
            switch_key = "on" if switch else "off"
            scatter_stats = bit_identities_of_repeats(scatter_once, min(arguments.repeats, 30))
            results["index_add_control"][switch_key] = scatter_stats
            print(f"[scatter] switch={'ON ' if switch else 'OFF'}: "
                  f"identical={scatter_stats['identical_vs_first']}/{scatter_stats['repeats']}")
        except Exception as error:  # noqa: BLE001
            results["index_add_control"]["on" if switch else "off"] = (
                f"{type(error).__name__}: {error}")

        # ---- 3b. index_put accumulate=True（文档声明的非确定 op）----
        try:
            t = torch.zeros(8, device=device)
            t.index_put_((torch.tensor([0, 1, 0], device=device),),
                         torch.ones(3, device=device), accumulate=True)
            backend_sync(device)
            results["index_put_accumulate"][f"{'on' if switch else 'off'}"] = "no_error_raised"
        except Exception as error:  # noqa: BLE001
            results["index_put_accumulate"][f"{'on' if switch else 'off'}"] = (
                f"{type(error).__name__}: {str(error)[:200]}")

        # ---- 3c. matmul × CUBLAS_WORKSPACE_CONFIG（开关在 cuBLAS 上的"牙"）----
        if device.type == "cuda":
            import os
            if switch and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
                try:
                    a = torch.randn(256, 256, device=device)
                    b = torch.randn(256, 256, device=device)
                    _ = a @ b
                    backend_sync(device)
                    results["matmul_cublas_workspace"][f"{'on' if switch else 'off'}_unset_env"] = (
                        "no_error_raised")
                except Exception as error:  # noqa: BLE001
                    results["matmul_cublas_workspace"][f"{'on' if switch else 'off'}_unset_env"] = (
                        f"{type(error).__name__}: {str(error)[:200]}")

    set_deterministic(False)  # 复位，避免污染后续实验

    # ---- 4. CPU 附加：线程数变化 ----
    if arguments.backend == "cpu":
        threads_behavior = {}
        for threads in (1, 4, 12):
            try:
                torch.set_num_threads(threads)
                out = composite_rmsnorm(x, w)
                threads_behavior[str(threads)] = hashlib.sha256(
                    out.numpy().tobytes()).hexdigest()
            except Exception as error:  # noqa: BLE001
                threads_behavior[str(threads)] = f"{type(error).__name__}: {error}"
        torch.set_num_threads(max(1, 12))
        results["num_threads_bit_hash"] = threads_behavior
        results["num_threads_all_same"] = (
            len(set(v for v in threads_behavior.values() if isinstance(v, int))) == 1)
        print(f"[threads] 不同线程数位级一致: {results['num_threads_all_same']}")

    # ---- 与 CPU 真值的数值漂移（一次运行）----
    sys.path.insert(0, str(HERE / "implementations" / "cpu"))
    from serial_rmsnorm import naive_rmsnorm_serial  # noqa: E402
    truth = naive_rmsnorm_serial(x_np, w_np, 1e-6)
    native = composite_rmsnorm(x, w).float().cpu().numpy()
    results["drift_vs_cpu_serial_truth"] = {
        "max_abs": float(np.abs(native - truth).max()),
        "mean_abs": float(np.abs(native - truth).mean()),
    }
    try:
        fn_rms = getattr(torch.nn.functional, "rms_norm", None)
        if fn_rms is not None:
            fused = fn_rms(x, (H,), w, 1e-6).float().cpu().numpy()
            results["drift_vs_cpu_serial_truth"]["f_rms_norm_max_abs"] = (
                float(np.abs(fused - truth).max()))
    except Exception:  # noqa: BLE001
        pass
    print(f"[truth] drift vs CPU serial: max={results['drift_vs_cpu_serial_truth']['max_abs']:.2e}")

    out_path = Path(arguments.out) if arguments.out else (
        HERE / f"results/torch-deterministic-{arguments.backend}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    print(f"[done] 已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
