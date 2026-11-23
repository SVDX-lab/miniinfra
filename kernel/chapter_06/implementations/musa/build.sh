#!/usr/bin/env bash
# ============================================================================
# 第 06 期 MUSA 构建脚本（CUDA 版 build.sh 的迁移对应物）。
# torch_musa 扩展（musa_extension.load，源 .cu→.mu 复制 + include 改写，
# 见 build_ext.py / M-08/M-09）。本期无独立 bench 二进制。
# 用法：./build.sh   （构建 + 自检）
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "[build] mcc 版本："
mcc --version || true

echo "[build] 构建 torch_musa 扩展（musa_extension 路径，三源 .mu 链）"
python3 build_ext.py --verbose
echo "[build] 完成"
