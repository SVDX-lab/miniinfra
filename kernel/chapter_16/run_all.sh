#!/usr/bin/env bash
# 第 16 期一键验收。
# 用法：
#   BACKENDS="cpu" ./run_all.sh               # 本地（无 GPU）：CPU 真值 + CPU 基准
#   BACKENDS="cpu cuda" ./run_all.sh          # 4090 服务器：构建 CUDA Kernel + CUDA 列
#   BACKENDS="cpu musa" ./run_all.sh          # S4000 服务器：构建 MUSA Kernel + MUSA 列
set -euo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cpu}"
PY="${PY:-python3}"

echo "=== [1/4] Smoke Test（CPU-only）==="
"$PY" smoke_test.py

echo "=== [2/4] Fixtures（固定 seed；已存在则 --verify-only 重放校验 SHA256）==="
if [ -f fixtures/correctness.npz ] && [ -f fixtures/manifest.json ]; then
  "$PY" generate_fixtures.py --verify-only
else
  "$PY" generate_fixtures.py
fi

echo "=== [3/4] 构建 Kernel 库（按需）==="
for backend in $BACKENDS; do
  case "$backend" in
    cuda)
      (cd implementations/cuda && bash build.sh)
      ;;
    musa)
      (cd implementations/musa && bash build.sh)
      ;;
  esac
done

echo "=== [4/4] 各后端正确性对照 + 基准 ==="
for backend in $BACKENDS; do
  "$PY" compare_correctness.py --backend "$backend"
  "$PY" benchmark.py --backend "$backend"
done

echo "=== 完成。结果见 results/ ==="
