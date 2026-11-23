"""第 14 期 CUDA 列（待补）：Inductor 生成 Triton 源码导出。

4090 修复后在服务器执行：
    TORCH_LOGS=output_code python export_inductor_triton.py \
        > ../../results/inductor-cuda/output_code-add_rmsnorm.txt 2>&1
再配合 bash run_cuda_session.sh 的缓存目录扫描（把 Inductor 写到
/tmp/torchinductor_root/*/…/triton/*.py 的生成 kernel 收进 results/）。

对照目标（ch14 手写 Triton，results/ir/cuda/ 有其全套下降产物）：
- Inductor 是否生成与手写等价的 BLOCK=next_pow2(H) 单遍 SP kernel；
- 归约树/num_warps 选择 vs ch14 autotune 选中（B=1024, nw=2 fp16）；
- fused add+rmsnorm：Inductor 自动合成 vs ch14 fused_triton.py 手写。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def add_rmsnorm(r, h, w, eps: float = 1e-6):
    t = r + h
    ft = t.float()
    var = ft.pow(2).mean(-1, keepdim=True)
    y = w * (ft * torch.rsqrt(var + eps)).to(t.dtype)
    return t, y


def rmsnorm_expr(hidden, weight, eps: float = 1e-6):
    input_dtype = hidden.dtype
    float_states = hidden.float()
    variance = float_states.pow(2).mean(dim=-1, keepdim=True)
    normalized = float_states * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用——待补项，4090 修复后执行")

    device = torch.device("cuda")
    for dtype in (torch.float16, torch.float32):
        gen = torch.Generator(device="cpu").manual_seed(20260415)
        r = torch.randn(args.rows, args.hidden, generator=gen).to(device=device, dtype=dtype)
        h = torch.randn(args.rows, args.hidden, generator=gen).to(device=device, dtype=dtype)
        w = torch.randn(args.hidden, generator=gen).to(device=device, dtype=dtype)
        torch._dynamo.reset()
        compiled = torch.compile(add_rmsnorm)
        compiled(r, h, w)
        torch.cuda.synchronize()
        torch._dynamo.reset()
        compiled_rms = torch.compile(rmsnorm_expr)
        compiled_rms(r, w)
        torch.cuda.synchronize()
        print(f"exported codegen for {dtype}（见 TORCH_LOGS 输出与 inductor 缓存目录）")

    cache_root = Path("/tmp/torchinductor_root")
    out_dir = Path(__file__).resolve().parent.parent.parent / "results" / "inductor-cuda"
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_root.exists():
        for source in cache_root.rglob("*.py"):
            if "triton" in str(source.parent) or "triton_" in source.name:
                target = out_dir / f"cache-{source.parent.parent.name}-{source.name}"
                target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"triton 缓存源码已收集到 {out_dir}")
    else:
        print(f"未找到 {cache_root}（TORCHINDUCTOR 环境变量自定义时需调整）")


if __name__ == "__main__":
    main()
