#!/usr/bin/env bash
# 第 06 期 CUDA 列一键执行（服务器侧，锁内使用；目标 ≤45min）。
# 顺序：build+selftest → correctness → benchmark → layout-probe → nsys → 清理。
set -euo pipefail
cd "$(dirname "$0")"

echo "=== [1/6] smoke（CPU-only，验 fixtures 语义） ==="
python3 smoke_test.py

echo "=== [2/6] build + selftest ==="
(cd implementations/cuda && ./build.sh)

echo "=== [3/6] correctness ==="
python3 compare_correctness.py --backend cuda

echo "=== [4/6] benchmark ==="
python3 benchmark.py --backend cuda

echo "=== [5/6] layout probe + nsys ==="
python3 probe_layout.py --device cuda
(cd implementations/cuda && ./run_nsys.sh)

echo "=== [6/6] 清理构建产物 ==="
rm -rf implementations/cuda/build __pycache__ implementations/*/__pycache__ || true
df -h / | tail -1
echo "=== CUDA 列完成 ==="
