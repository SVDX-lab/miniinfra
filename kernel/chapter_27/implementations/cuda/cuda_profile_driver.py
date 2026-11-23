#!/usr/bin/env python3
"""nsys 分相位驱动：只跑 chunked forward 若干次（K1/K2/K3 是三个独立 kernel
launch，nsys kernsum 按核名归因分相位时长）。

用法：nsys profile --output=/tmp/ch28_prof python3 cuda_profile_driver.py
然后：nsys stats --report cuda_gpu_kern_sum /tmp/ch28_prof.nsys-rep
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parent.parent
sys.path.insert(0, str(CHAPTER))
sys.path.insert(0, str(HERE.parent / "cpu"))

import deltanet_config as cfg  # noqa: E402
import deltanet_reference as ref  # noqa: E402
from cuda_test import load_ext  # noqa: E402


def main() -> int:
    ext = load_ext()
    weights = ref.make_weights(cfg.SEED_BASE)
    torch.cuda.init()
    for n in (2048, 32768):
        gen = torch.Generator().manual_seed(cfg.SEED_BASE + 7000 + n)
        x = torch.randn(n, cfg.D_MODEL, generator=gen, dtype=torch.float32)
        proj = ref.project_inputs(x, weights, "scalar")
        qc = proj["q"].cuda(); kc = proj["k"].cuda(); vc = proj["v"].cuda()
        bc = proj["beta"].cuda(); gc = proj["g_log"].cuda()
        for _ in range(3):  # warmup（JIT/上下文）
            ext.gdn_chunked_forward(qc, kc, vc, bc, gc, 64, cfg.SCALE, None, 0)
        torch.cuda.synchronize()
        for _ in range(10):
            ext.gdn_chunked_forward(qc, kc, vc, bc, gc, 64, cfg.SCALE, None, 0)
        torch.cuda.synchronize()
        print(f"[profile] N={n} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
