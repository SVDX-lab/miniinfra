#!/usr/bin/env bash
# ============================================================================
# 第 02 期 CUDA 构建脚本。
#
# 构建选择记录（合同"编译期 specialization 与运行时限制"条目）：
#   - torch 扩展路径：torch.utils.cpp_extension.load + PYBIND11_MODULE
#     （TORCH_LIBRARY 注册留给第 04 期"进入框架 Dispatch"）；
#   - 独立可执行程序 rmsnorm_bench：ncu/nsys 的被测对象 + 文件 IO 通道；
#   - Kernel 模板实例化 TILE ∈ {64,128,256,512,1024} × VEC ∈ {0,1} ×
#     dtype ∈ {fp32,fp16} 一次性编入 —— Tile 扫描无需重编译；
#   - CUDA_ARCH 默认 89（RTX 4090, SM 8.9），可用环境变量覆盖。
# 用法：./build.sh            （构建 bench + torch 扩展）
#       BUILD_EXT=0 ./build.sh（只构建 bench）
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

CUDA_ARCH="${CUDA_ARCH:-89}"
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

echo "[build] nvcc arch=sm_${CUDA_ARCH}"
nvcc -O3 -std=c++17 -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
     -o "$BUILD_DIR/rmsnorm_bench" rmsnorm_bench.cu rmsnorm_kernel.cu
echo "[build] $BUILD_DIR/rmsnorm_bench OK"

if [ "${BUILD_EXT:-1}" = "1" ]; then
  echo "[build] 构建 torch 扩展（cpp_extension.load 路径）"
  python3 build_ext.py --verbose
fi
echo "[build] 完成"
