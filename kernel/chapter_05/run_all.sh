#!/usr/bin/env bash
# 第 05 期一键验收（在已配置好 Python 环境的机器上执行）。
# 用法：
#   BACKENDS="cpu" ./run_all.sh            # 本地（真值 + 位布局实验 + 累加实验 + 三 dtype 模拟路径）
#   BACKENDS="cuda" ./run_all.sh           # 4090 服务器（build + CUDA 列）
#   BACKENDS="musa" ./run_all.sh           # S4000 服务器（build + MUSA 列）
#   BACKENDS="cuda" SKIP_FIXTURES=1 ./run_all.sh   # fixtures 已就位时跳过生成
set -euo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cpu}"
KERNEL_BACKEND="${KERNEL_BACKEND:-auto}"

echo "=== [1/5] Smoke Test（CPU-only，dtype 布局工具与模拟器自洽）==="
python3 smoke_test.py

echo "=== [2/5] Fixtures（CPU，固定 seed；含实验 4 构造输入；跨服务器共用同一份）==="
if [ "${SKIP_FIXTURES:-0}" = "1" ]; then
  echo "SKIP_FIXTURES=1：使用已上传的 fixtures/"
else
  python3 generate_fixtures.py
fi

echo "=== [3/5] 实验 1（位布局对照，本地 CPU-only）==="
if [ "${SKIP_LAYOUT:-0}" != "1" ]; then
  python3 exp_dtype_layout.py
fi

echo "=== [4/5] 构建各后端实现 ==="
for backend in $BACKENDS; do
  if [ "$backend" = "cuda" ]; then
    (cd implementations/cuda && ./build.sh)
  elif [ "$backend" = "musa" ]; then
    (cd implementations/musa && ./build.sh)
  fi
done

echo "=== [5/5] 各后端正确性对照（三 dtype + 累加实验）+ 性能（理论字节比例实验）==="
for backend in $BACKENDS; do
  python3 compare_correctness.py --backend "$backend" --kernel-backend "$KERNEL_BACKEND"
  python3 benchmark.py --backend "$backend" --kernel-backend "$KERNEL_BACKEND"
  if [ "$backend" = "cpu" ]; then
    python3 exp_accumulation.py   # 实验 4 CPU 侧深挖（位精确模拟器）
  fi
done

echo "=== 完成。结果见 results/ ==="
