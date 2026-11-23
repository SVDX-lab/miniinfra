#!/usr/bin/env python3
"""第 13 期编译产物与首次运行成本实验（大纲实验清单第 5 项 + 纸面第 6 项取证）。

内容：
  1. 冷/热缓存编译时间：TRITON_CACHE_DIR 指向全新目录 → 首次 launch（含编译）
     vs 第二次 launch；再测换 dtype/换 BLOCK 的重编译代价；
  2. 编译产物 introspection：从 JITFunction 缓存取 CompiledKernel，
     枚举 .asm dict 全部 key（CUDA 预期 ttir/ttgir/llir/ptx/cubin；
     MUSA 侧对应物实测——ttir/ttgir/llir 存在与否、目标 ISA 产物名），
     每个产物存 results/ir/{backend}/<name>.<ext>；
  3. kernel.asm['ptx']（CUDA）/对应物（MUSA）查看方法验证；
  4. cubin/mubin 反汇编工具探测：CUDA nvdisasm → SASS；MUSA 侧候选工具实测；
  5. metadata（num_warps/shared/寄存器等）记录。

用法：python compile_artifacts.py --backend cuda|musa
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

BACKEND = os.environ.get("CH13_BACKEND") or os.environ.get("CH14_BACKEND", "cuda")
RESULT_DIR = HERE.parent.parent / "results"


def sync(backend: str) -> None:
    if backend == "cuda":
        torch_mod = __import__("torch").cuda
        torch_mod.synchronize()
    else:
        torch_mod = __import__("torch").musa
        torch_mod.synchronize()


def find_compiled_kernels(jit_fn) -> dict:
    """从 JITFunction 的多版本缓存字段里挖出全部 CompiledKernel（版本容忍）。"""
    found = {}
    for attr in ("cache", "device_caches", "autotune_cache"):
        caches = getattr(jit_fn, attr, None)
        if not caches:
            continue
        try:
            for device_key, device_cache in caches.items():
                if isinstance(device_cache, tuple):
                    # 3.3+ device_caches[device] = (cache_dict, target, backend, binder)
                    for item in device_cache:
                        if hasattr(item, "values"):
                            device_cache = item
                            break
                if not hasattr(device_cache, "values"):
                    continue
                for kernel_key, compiled in device_cache.items():
                    if hasattr(compiled, "asm") or hasattr(compiled, "metadata"):
                        found[f"{attr}[{device_key}][{str(kernel_key)[:60]}]"] = compiled
        except Exception as error:  # noqa: BLE001
            found[f"{attr}@error"] = f"{type(error).__name__}: {error}"
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=BACKEND, choices=["cuda", "musa"])
    arguments = parser.parse_args()
    backend = arguments.backend
    import torch  # noqa: PLC0415
    import triton  # noqa: PLC0415
    from rmsnorm_triton import rmsnorm, rmsnorm_fwd_kernel, rmsnorm_fwd_looped  # noqa: PLC0415

    device = torch.device(backend)
    eps = 1e-6
    result: dict = {"backend": backend, "triton_version": triton.__version__,
                    "torch_version": torch.__version__}
    ir_dir = RESULT_DIR / "ir" / backend
    ir_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 冷/热缓存编译时间（独立 TRITON_CACHE_DIR） ----
    print("===== [artifacts] 1. 冷/热缓存编译时间 =====")
    cache_root = Path(tempfile.mkdtemp(prefix="ch14_triton_cache_"))
    old_cache_env = os.environ.get("TRITON_CACHE_DIR")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)
    try:
        torch.manual_seed(0)
        x = torch.randn(512, 1024, device=device, dtype=torch.float32)
        w = torch.randn(1024, device=device, dtype=torch.float32)
        timings = []
        # 第一次：全新缓存目录 + 全新 kernel -> 编译
        t0 = time.perf_counter()
        y = rmsnorm(x, w, eps)
        sync(backend)
        timings.append({"call": "cold_fp32_SP", "wall_s": round(time.perf_counter() - t0, 4)})
        # 第二次：热缓存
        t0 = time.perf_counter()
        y = rmsnorm(x, w, eps)
        sync(backend)
        timings.append({"call": "warm_fp32_SP", "wall_s": round(time.perf_counter() - t0, 4)})
        # 换 dtype -> 重编译（新签名特化）
        x16 = x.to(torch.float16)
        w16 = w.to(torch.float16)
        t0 = time.perf_counter()
        y16 = rmsnorm(x16, w16, eps)
        sync(backend)
        timings.append({"call": "cold_fp16_SP_new_sig", "wall_s": round(time.perf_counter() - t0, 4)})
        # 换 BLOCK（looped kernel BLOCK=256）-> 新特化
        t0 = time.perf_counter()
        y2 = rmsnorm(x, w, eps, block_size=256, looped=True)
        sync(backend)
        timings.append({"call": "cold_fp32_looped_B256", "wall_s": round(time.perf_counter() - t0, 4)})
        # 热
        t0 = time.perf_counter()
        y2 = rmsnorm(x, w, eps, block_size=256, looped=True)
        sync(backend)
        timings.append({"call": "warm_fp32_looped_B256", "wall_s": round(time.perf_counter() - t0, 4)})
        result["compile_timings"] = timings
        for entry in timings:
            print(f"  {entry['call']:<26} {entry['wall_s']:.4f}s")

        # cache 目录里的产物文件清单（磁盘证据）
        files = []
        for path in sorted(cache_root.rglob("*")):
            if path.is_file():
                files.append({"rel": str(path.relative_to(cache_root)),
                              "bytes": path.stat().st_size})
        result["cache_dir_files"] = files
        print(f"  cache 目录产物 {len(files)} 个文件")
    finally:
        if old_cache_env is not None:
            os.environ["TRITON_CACHE_DIR"] = old_cache_env
        shutil.rmtree(cache_root, ignore_errors=True)

    # ---- 2/3. CompiledKernel introspection + asm 产物落盘 ----
    print("===== [artifacts] 2/3. CompiledKernel .asm 枚举与落盘 =====")
    kernels = find_compiled_kernels(rmsnorm_fwd_kernel)
    kernels.update({f"looped::{k}": v for k, v in find_compiled_kernels(rmsnorm_fwd_looped).items()})
    result["introspection"] = {}
    saved = {}
    for name, compiled in kernels.items():
        if not hasattr(compiled, "asm"):
            result["introspection"][name] = {"no_asm_attr": True,
                                             "type": type(compiled).__name__}
            continue
        asm = compiled.asm
        entry = {"asm_keys": list(asm.keys())}
        meta = getattr(compiled, "metadata", None)
        if meta is not None:
            entry["metadata_fields"] = {k: repr(v)[:200] for k, v in
                                        vars(meta).items()} if hasattr(meta, "__dict__") else repr(meta)[:600]
        for key, blob in asm.items():
            if isinstance(blob, bytes):
                out_name = f"{name.replace('/', '_').replace('[', '_').replace(']', '_')}__{key}.bin"
                (ir_dir / out_name).write_bytes(blob)
                entry[key] = {"kind": "bytes", "bytes": len(blob), "saved": out_name}
            else:
                text = str(blob)
                out_name = f"{name.replace('/', '_').replace('[', '_').replace(']', '_')}__{key}.txt"
                (ir_dir / out_name).write_text(text, encoding="utf-8")
                entry[key] = {"kind": "text", "chars": len(text), "saved": out_name,
                              "head": text[:400]}
        result["introspection"][name] = entry
        saved[name] = entry
        print(f"  {name}: asm keys = {entry.get('asm_keys')}")

    # kernel.asm['ptx'] 显式验证（CUDA 侧任务要求；MUSA 侧记录对应物）
    ptx_entries = {k: v for k, v in saved.items() if "ptx" in v.get("asm_keys", [])}
    result["ptx_access_verified"] = bool(ptx_entries)
    result["ptx_note"] = ("kernel.asm['ptx'] 可读" if ptx_entries
                          else "asm dict 中无 ptx key —— MUSA 侧对应物见 introspection")

    # ---- 4. 反汇编工具探测（SASS / 目标 ISA） ----
    print("===== [artifacts] 4. 反汇编工具探测 =====")
    tools = {}
    candidates = ["nvdisasm", "cuobjdump", "nvcc", "mcc", "musair", "muobjdump",
                  "mudisasm", "objdump", "llvm-objdump"] if backend == "musa" else \
                 ["nvdisasm", "cuobjdump", "nvcc", "objdump", "llvm-objdump"]
    for tool in candidates:
        which = shutil.which(tool)
        tools[tool] = which
    result["disasm_tools"] = tools
    # CUDA：把 cubin 落盘后 nvdisasm -> SASS 片段
    cubin_entry = next((v for k, v in saved.items()
                        if "cubin" in v.get("asm_keys", [])), None)
    if cubin_entry and tools.get("nvdisasm"):
        cubin_path = ir_dir / cubin_entry["cubin"]["saved"]
        try:
            proc = subprocess.run([tools["nvdisasm"], "-c", str(cubin_path)],
                                  capture_output=True, text=True, timeout=120)
            sass = proc.stdout
            (ir_dir / "sass_snippet.txt").write_text(sass[:20000], encoding="utf-8")
            result["sass"] = {"rc": proc.returncode, "chars": len(sass),
                              "head": sass[:600], "saved": "sass_snippet.txt"}
        except Exception as error:  # noqa: BLE001
            result["sass"] = {"error": str(error)}
    elif backend == "musa":
        result["sass"] = {"note": "MUSA 侧目标 ISA 反汇编：按实机探测（见 disasm_tools 与 introspection asm_keys）"}

    result["ir_dir"] = str(ir_dir)
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path = RESULT_DIR / f"compile-artifacts-{backend}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[artifacts] 已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
