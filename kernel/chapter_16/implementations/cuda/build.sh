#!/usr/bin/env bash
# 第 16 期 CUDA Kernel 构建：nvcc 编译为 C ABI 共享库（ctypes 直接加载，无 torch 扩展层）。
# 纯 Kernel 启动（无 cudaMalloc 等运行时调用），显存管理在 Python/torch 侧完成。
set -euo pipefail
cd "$(dirname "$0")"

NVCC="${NVCC:-nvcc}"
# RTX 4090 = SM 8.9（Ada）；换卡时可用 CUDA_ARCH=sm_xx 覆盖
ARCH="${CUDA_ARCH:-sm_89}"

"$NVCC" -O3 -arch="$ARCH" -std=c++17 -shared -Xcompiler -fPIC \
  ops_kernels.cu -o libops_cuda.so

echo "[build] libops_cuda.so 构建完成（arch=$ARCH）"
"$NVCC" --version | tail -2
