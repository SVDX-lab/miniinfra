#!/usr/bin/env bash
# 第 06 期一键验收（在已配置好 Python 环境的机器上执行）。
# 用法：
#   BACKENDS="cpu" ./run_all.sh            # 本地（stride 教学 + CPU 全实验）
#   BACKENDS="cuda" ./run_all.sh           # 4090 服务器（build + CUDA 列）
#   BACKENDS="musa" ./run_all.sh           # S4000 服务器（build + MUSA 列）
#   BACKENDS="cuda" SKIP_FIXTURES=1 ./run_all.sh   # fixtures 已就位时跳过生成
set -euo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cpu}"

echo "=== [1/3] Smoke Test（CPU-only：stride 语义 + 三策略边界 + 报错快照）==="
python3 smoke_test.py

echo "=== [2/3] Fixtures（CPU，固定 seed；跨服务器共用同一份）==="
if [ "${SKIP_FIXTURES:-0}" = "1" ]; then
  echo "SKIP_FIXTURES=1：使用已上传的 fixtures/"
else
  python3 generate_fixtures.py
fi

echo "=== [3/3] 各后端正确性对照 + 性能 ==="
for backend in $BACKENDS; do
  if [ "$backend" != "cpu" ]; then
    (cd implementations/"$backend" && ./build.sh)
  fi
  python3 compare_correctness.py --backend "$backend"
  python3 benchmark.py --backend "$backend"
done

echo "=== 完成。结果见 results/ ==="
