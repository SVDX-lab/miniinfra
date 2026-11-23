#!/usr/bin/env bash
# ============================================================================
# 第 21 期 CUDA 构建冒烟（服务器锁内；torch cpp_extension.load）
# 用法：bash implementations/cuda/build.sh   # 只构建 + 导出面检查
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
python3 build_ext.py
