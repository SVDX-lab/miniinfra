#!/usr/bin/env bash
# 第 17 期一键验收（在已配好环境的机器上执行）。
# 用法：
#   BACKENDS="cpu cuda" ./run_all.sh          # 4090 服务器（CPU 自洽 + CUDA 列）
#   BACKENDS="cpu musa" ./run_all.sh          # S4000 服务器（MUSA 列 + CPU 自洽校验）
#   NSYS=1 BACKENDS="cuda" ./run_all.sh       # 附带 nsys 证据采集（4090）
set -uo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cpu}"
NSYS="${NSYS:-0}"

echo "=== [1/3] Smoke Test（CPU-only）==="
python3 smoke_test.py || exit 1

if [ ! -f fixtures/correctness.npz ]; then
  echo "=== [2/3] fixtures 不存在（真值机器上生成后应随代码上传；此处应急重新生成）==="
  python3 generate_fixtures.py || exit 1
else
  echo "=== [2/3] fixtures 已存在（跨平台共用同一份，见 manifest.json）==="
fi

echo "=== [3/3] 各后端：构建 + 正确性 + 基准 ==="
for backend in $BACKENDS; do
  if [ "$backend" = "cpu" ]; then
    python3 compare_correctness.py --backend cpu || exit 1
    python3 benchmark.py --backend cpu || exit 1
  else
    bash "implementations/$backend/build.sh" || exit 1
    echo "--- $backend 二进制 selftest ---"
    "implementations/$backend/naive_gemm_bench" selftest || exit 1
    python3 compare_correctness.py --backend "$backend" || exit 1
    python3 benchmark.py --backend "$backend" || exit 1
    if [ "$NSYS" = "1" ] && [ "$backend" = "cuda" ]; then
      echo "--- nsys 证据采集 ---"
      bash implementations/cuda/run_nsys.sh || echo "[warn] nsys 采集失败（记录，不阻断）"
    fi
  fi
done

echo "=== 完成。结果见 results/ ==="
