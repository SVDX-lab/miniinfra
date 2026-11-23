#!/usr/bin/env bash
# ============================================================================
# 第 07 期 CUDA 构建：nvcc 编译 C ABI 共享库 + 独立 bench。
#
# 产物：
#   libmini_detatomic_cuda.so —— ctypes 直传库（单 TU kernels_lib.cu：
#       ep03 确定性 kernel + atomic 两档 + probe/zero，编译期实例化
#       det TILE{64..1024} × atomic TILE{128,256,512} × ACC{fp32,half} ×
#       VEC × dtype）；
#   build/det_atomic_bench —— 独立可执行程序（nsys 被测对象 + 大形状
#       determinism 通道：host FP32 串行真值 C 速度现算）。
#
# ACC=half（fp16 atomicAdd）需要 sm_70+；4090 = sm_89 默认编入。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

CUDA_ARCH="${CUDA_ARCH:-89}"
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

echo "[build] nvcc arch=sm_${CUDA_ARCH}：libmini_detatomic_cuda.so（ctypes 直传）"
nvcc -O3 -std=c++17 -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
     -shared -Xcompiler -fPIC \
     kernels_lib.cu -o libmini_detatomic_cuda.so
echo "[build] libmini_detatomic_cuda.so OK"

echo "[build] nvcc 独立 bench：det_atomic_bench（nsys 被测对象）"
nvcc -O3 -std=c++17 -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
     -o "$BUILD_DIR/det_atomic_bench" det_atomic_bench.cu
echo "[build] $BUILD_DIR/det_atomic_bench OK"

nvcc --version | tail -2
echo "[build] 完成"
