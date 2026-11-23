#!/usr/bin/env python3
"""generate_fixtures.py —— 第 10 期 fixtures。

本期是工具链体检期，无外部输入 fixtures（所有 kernel 输入由固定种子在
二进制内部生成，splitmix64 + Box-Muller，见 defect_suite.cu / health_targets.cu）。
因此本脚本产出两类"fixture 等价物"：

1. 源码同源清单 results/source-manifest.sha256：
   体检对象（ch03/ch04/ch08 复制源）的 SHA256，用于跨服务器复跑时验证
   "逐字节同源"声明；
2. 输入可复现性自证：用 Python 复刻 splitmix64+Box-Muller，打印前 8 个数，
   供人工与 C++ 侧核对（同一 seed 20261105 应逐位一致）。

用法：python3 generate_fixtures.py [--emit-source-manifest]
"""
import argparse
import hashlib
import pathlib
import sys

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def splitmix64(state: int) -> tuple[int, float]:
    state = (state + GOLDEN) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    u = ((z ^ (z >> 31)) & MASK64) / 18446744073709551616.0
    return state, u


def fill_normal(n: int, seed: int) -> list[float]:
    state = seed
    out: list[float] = []
    i = 0
    while i + 1 < n:
        state, u1 = splitmix64(state)
        state, u2 = splitmix64(state)
        u1 = max(u1, 1e-12)
        r = (-2.0 * __import__("math").log(u1)) ** 0.5
        theta = 2.0 * 3.14159265358979323846 * u2
        out.append(r * __import__("math").cos(theta))
        out.append(r * __import__("math").sin(theta))
        i += 2
    if n % 2 == 1:
        state, u1 = splitmix64(state)
        state, u2 = splitmix64(state)
        u1 = max(u1, 1e-12)
        out.append(((-2.0 * __import__("math").log(u1)) ** 0.5)
                   * __import__("math").cos(2.0 * 3.14159265358979323846 * u2))
    return [float(__import__("struct").unpack("f", __import__("struct").pack("f", v))[0]) for v in out]


SOURCES = [
    "implementations/cuda/rmsnorm_kernel.cu",
    "implementations/cuda/rmsnorm_kernel_ep03.cu",
    "implementations/cuda/rmsnorm_bench.cu",
    "implementations/cuda/fused_rmsnorm_kernels.cu",
    "implementations/cuda/atomic_rmsnorm.cu",
    "implementations/cuda/kernels_lib.cu",
]

ORIGIN = {
    "implementations/cuda/rmsnorm_kernel.cu": "chapter_02/implementations/cuda/rmsnorm_kernel.cu",
    "implementations/cuda/rmsnorm_kernel_ep03.cu": "chapter_07/implementations/cuda/rmsnorm_kernel_ep03.cu",
    "implementations/cuda/rmsnorm_bench.cu": "chapter_02/implementations/cuda/rmsnorm_bench.cu",
    "implementations/cuda/fused_rmsnorm_kernels.cu": "chapter_03/implementations/cuda/fused_rmsnorm_kernels.cu",
    "implementations/cuda/atomic_rmsnorm.cu": "chapter_07/implementations/cuda/atomic_rmsnorm.cu",
    "implementations/cuda/kernels_lib.cu": "chapter_07/implementations/cuda/kernels_lib.cu",
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-source-manifest", action="store_true")
    args = ap.parse_args()
    here = pathlib.Path(__file__).resolve().parent

    print("== 输入可复现性自证（seed=20261105 前 8 个 float32）==")
    vals = fill_normal(8, 20261105)
    for i, v in enumerate(vals):
        print(f"x[{i}] = {v!r}")
    print("（与服务器上 health_targets.cu / defect_suite.cu 内 splitmix64+Box-Muller 同法；")
    print("  如需逐位核对，可在实机跑 good 自检输出 max_abs_error=0 即证明一致）")

    if args.emit_source_manifest:
        out = here / "results" / "source-manifest.sha256"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for rel in SOURCES:
            p = here / rel
            if not p.exists():
                print(f"缺失: {rel}", file=sys.stderr)
                return 1
            lines.append(f"{sha256(p)}  {rel}  # <- {ORIGIN[rel]}")
        out.write_text("\n".join(lines) + "\n")
        print(f"== 源码同源清单写入 {out} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
