#!/usr/bin/env bash
# 第 26 期 CUDA 列构建脚本（4090 / nvcc 13.0 / sm_89）。
# 用法：bash build.sh
set -euo pipefail

cd "$(dirname "$0")"

if nvcc -O3 -std=c++17 -arch=native -o verify_batch_bench verify_batch_bench.cu -lcublas 2>build-native.log; then
    echo "[build] -arch=native 构建成功"
else
    echo "[build] -arch=native 失败（build-native.log），回退 -arch=sm_89"
    nvcc -O3 -std=c++17 -arch=sm_89 -o verify_batch_bench verify_batch_bench.cu -lcublas 2>build-sm89.log
    echo "[build] -arch=sm_89 构建成功"
fi
echo "[build] verify_batch_bench 构建完成"
