#!/usr/bin/env bash
# ============================================================================
# 第 06 期 CUDA 构建脚本：torch 扩展（三策略 kernel 单编译单元）。
# 本期无独立 bench 二进制（三策略统一从 torch 层计时 + nsys 证据）。
# 用法：./build.sh   （构建 + 自检）
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p build   # cpp_extension 不自建 build 目录（M-29 实机经验）

echo "[build] nvcc 版本："
nvcc --version || true

echo "[build] 构建 torch 扩展（cpp_extension.load，单编译单元）"
python3 build_ext.py --verbose
echo "[build] 完成"
