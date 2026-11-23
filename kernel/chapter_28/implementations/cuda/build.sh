#!/usr/bin/env bash
# 第 28 期 CUDA 构建脚本（4090 上执行；本地无 nvcc）
# 用法：bash build.sh && ./moe_route [--bench]
set -euo pipefail
cd "$(dirname "$0")"

ARCH=${CUDA_ARCH:-sm_89}   # 4090 = sm_89；换卡时覆盖
nvcc -O3 -std=c++17 -arch=${ARCH} moe_route.cu -o moe_route
echo "[build] moe_route built (arch=${ARCH})"
