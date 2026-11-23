#!/usr/bin/env bash
# 第 27 期 CUDA chunked kernel 构建（4090 回填用；本文件为待回填骨架的构建入口）
# 骨架 deltanet_chunked.cu 的 host 入口目前 TORCH_CHECK(false) 占位——
# 回填时先展开核体再跑本脚本；预期编译告警点见 deltanet_chunked_design.md §4。
set -euo pipefail
cd "$(dirname "$0")"

nvcc --version | tail -1
python3 build_ext.py 2>&1 | tee build.log
echo "build 完成；冒烟：python3 ../../smoke_test.py --backend cuda（回填后）"
