#!/usr/bin/env bash
# ============================================================================
# 第 03 期 CUDA 构建：nvcc 编译为 C ABI 共享库（M-17 ctypes 直传模式，
# 与 nvcc 独立 bench 两个产物）。
#
# 构建选择记录（合同"编译期 specialization 与运行时限制"条目）：
#   - libmini_fused_cuda.so：单编译单元 fused_rmsnorm_kernels.cu
#     （#include rmsnorm_kernel_ep03.cu），Kernel 模板实例化
#     TILE ∈ {64,128,256,512,1024} × VEC ∈ {0,1} × dtype ∈ {fp32,fp16}
#     一次性编入；零 Runtime API 调用（显存/同步留 torch 侧）；
#   - fused_bench：独立可执行程序（nsys 被测对象 + Event 计时第二口径 +
#     文件 IO 通道），含自身 cudaMalloc/Event（bench 程序允许用 Runtime API）；
#   - CUDA_ARCH 默认 89（RTX 4090, SM 8.9），可用环境变量覆盖。
# 用法：./build.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

CUDA_ARCH="${CUDA_ARCH:-89}"
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

echo "[build] nvcc arch=sm_${CUDA_ARCH}：libmini_fused_cuda.so（ctypes 直传）"
nvcc -O3 -std=c++17 -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
     -shared -Xcompiler -fPIC \
     fused_rmsnorm_kernels.cu -o libmini_fused_cuda.so
echo "[build] libmini_fused_cuda.so OK"

echo "[build] nvcc 独立 bench：fused_bench（nsys 被测对象）"
nvcc -O3 -std=c++17 -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
     -o "$BUILD_DIR/fused_bench" fused_bench.cu fused_rmsnorm_kernels.cu
echo "[build] $BUILD_DIR/fused_bench OK"

nvcc --version | tail -2
echo "[build] 完成"
