#!/usr/bin/env bash
# ============================================================================
# 第 24 期 CUDA 构建脚本（torch 扩展路径，沿用第 02 期模式）。
#   - torch.utils.cpp_extension.load + PYBIND11_MODULE（TORCH_LIBRARY 注册留给第 04 期）
#   - CUDA_ARCH 默认 89（RTX 4090, SM 8.9），可用环境变量覆盖
# 用法：bash build.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

CUDA_ARCH="${CUDA_ARCH:-89}"
export CUDA_ARCH
echo "[build] 构建 torch 扩展（cpp_extension.load, arch=sm_${CUDA_ARCH}）"
python3 build_ext.py --verbose
echo "[build] 完成"
