#!/usr/bin/env bash
# 第 18 期一键验收。
# 用法：
#   BACKENDS="cuda" NSYS=1 ./run_all.sh       # 4090（WMMA 全量 + 同会话朴素对照 + nsys）
#   BACKENDS="musa" ./run_all.sh              # S4000（MMA 探测 + 回退编译 + 可用项实验）
set -uo pipefail
cd "$(dirname "$0")"
BACKENDS="${BACKENDS:-cuda}"
NSYS="${NSYS:-0}"

echo "=== [1/3] Smoke Test（CPU-only）==="
python3 smoke_test.py || exit 1

if [ ! -f fixtures/correctness.npz ]; then
  echo "=== [2/3] fixtures 缺失，从第 17 期单源重生成 + SHA 对拍 ==="
  python3 generate_fixtures.py || exit 1
else
  echo "=== [2/3] fixtures 已存在（第 17 期单源，见 manifest provenance）==="
fi

echo "=== [3/3] 各后端：构建 + 探测 + 正确性 + 基准 ==="
for backend in $BACKENDS; do
  if [ "$backend" = "musa" ]; then
    bash implementations/musa/mma_probe/probe.sh || echo "[warn] MMA 探测脚本失败（记录，不阻断）"
  fi
  bash "implementations/$backend/build.sh" || exit 1
  echo "--- $backend 二进制 selftest（含 WMMA 段）---"
  "implementations/$backend/naive_gemm_bench" selftest || exit 1
  python3 compare_correctness.py --backend "$backend" || exit 1
  python3 benchmark.py --backend "$backend" || exit 1
  if [ "$NSYS" = "1" ] && [ "$backend" = "cuda" ]; then
    bash implementations/cuda/run_nsys.sh || echo "[warn] nsys 采集失败（记录，不阻断）"
  fi
done
echo "=== 完成。结果见 results/ ==="
