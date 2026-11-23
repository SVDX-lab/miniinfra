#!/usr/bin/env python3
"""第 13 期冒烟测试：Triton 路径最小可跑集（服务器 GPU 上执行）。

链：import triton → probe driver → RMSNorm SP/looped/autotune 三路径 ×
fp32/fp16 → fused → 手写对照 runner 可构建性（--with-handwritten）。
任何一环失败即非零退出并打印原因（协议：失败如实记录）。

用法：python smoke_test.py --backend cuda|musa [--with-handwritten]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "musa"])
    parser.add_argument("--with-handwritten", action="store_true")
    arguments = parser.parse_args()
    backend = arguments.backend
    device = torch.device(backend)
    eps = 1e-6
    started = time.perf_counter()
    sys.path.insert(0, str(HERE / "implementations" / backend))

    # 1. import 与版本
    import triton  # noqa: PLC0415
    print(f"[smoke/{backend}] triton={triton.__version__} torch={torch.__version__}")
    from triton.runtime import driver  # noqa: PLC0415
    try:
        print(f"[smoke/{backend}] target={driver.active.get_current_target()}")
    except Exception as error:  # noqa: BLE001
        print(f"[smoke/{backend}] target 不可用: {error}")

    # 2. RMSNorm 三路径 × 两 dtype
    from rmsnorm_triton import AutotuneRMSNorm, rmsnorm  # noqa: PLC0415
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(64, 1024, device=device, dtype=dtype)
        w = torch.randn(1024, device=device, dtype=dtype)
        var = x.float().pow(2).mean(-1, keepdim=True)
        ref = (w * (x.float() * torch.rsqrt(var + eps)).to(dtype))
        y1 = rmsnorm(x, w, eps)
        y2 = rmsnorm(x, w, eps, block_size=256, looped=True)
        y3 = AutotuneRMSNorm("parity")(x, w, eps)
        for name, y in (("SP", y1), ("looped-B256", y2), ("autotune", y3)):
            err = (y.float() - ref.float()).abs().max().item()
            print(f"[smoke/{backend}] rmsnorm {name} {dtype}: max_abs_vs_torch={err:.3e}")
            assert err < (2e-2 if dtype == torch.float16 else 1e-5), f"{name} {dtype} 误差超差"

    # 3. 非整 H（mask 路径）
    x = torch.randn(8, 127, device=device, dtype=torch.float16)
    w = torch.randn(127, device=device, dtype=torch.float16)
    var = x.float().pow(2).mean(-1, keepdim=True)
    ref = (w * (x.float() * torch.rsqrt(var + eps)).to(torch.float16))
    for block in (128, 256):
        y = rmsnorm(x, w, eps, block_size=block, looped=True)
        err = (y.float() - ref.float()).abs().max().item()
        print(f"[smoke/{backend}] rmsnorm H=127 BLOCK={block}: max_abs={err:.3e}")
        assert err < 2e-2, "H=127 mask 路径误差超差"

    # 4. fused
    from fused_triton import fused_add_rmsnorm  # noqa: PLC0415
    r = torch.randn(64, 1024, device=device, dtype=torch.float16)
    h_in = torch.randn(64, 1024, device=device, dtype=torch.float16)
    w = torch.randn(1024, device=device, dtype=torch.float16)
    r0 = r.clone()
    y, r_after = fused_add_rmsnorm(r, h_in, w, eps)
    t = r0 + h_in
    err = (y.float() - (w.float() * (t.float() * torch.rsqrt(
        t.float().pow(2).mean(-1, keepdim=True) + eps)).to(torch.float16).float())).abs().max().item()
    resid_err = (r_after.float() - t.float()).abs().max().item()
    print(f"[smoke/{backend}] fused fp16: y max_abs={err:.3e} residual max_abs={resid_err:.3e}")
    assert err < 2e-2 and resid_err < 2e-2, "fused 语义超差"

    # 5. 手写对照 runner（可选：构建耗时较长，正式对照在 benchmark --with-handwritten）
    if arguments.with_handwritten:
        from reference_runners import HandwrittenRMSNorm  # noqa: PLC0415
        hw = HandwrittenRMSNorm(backend)
        y = hw.run(torch.randn(8, 1024, device=device, dtype=torch.float32),
                   torch.randn(1024, device=device, dtype=torch.float32), eps, 256)
        print(f"[smoke/{backend}] handwritten-ch03 构建成功 build={hw.build_seconds}s")

    print(f"[smoke/{backend}] 全部通过，用时 {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
