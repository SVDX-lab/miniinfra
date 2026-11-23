#!/usr/bin/env bash
# 第 19 期一键验收（在已配好环境的机器上执行）。
# 用法：
#   BACKENDS="cuda" ./run_all.sh           # 4090 服务器（CUDA 列全套）
#   BACKENDS="musa" ./run_all.sh           # S4000 服务器（MUSA 列）
#   RACECHECK=1 BACKENDS="cuda" ./run_all.sh  # 附带 racecheck 体检（4090）
# 无 CPU 列（期首声明：设备侧数据供给组织概念；smoke/真值链在本地/服务器
# CPU 上跑但不计性能）。
set -uo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cuda}"
RACECHECK="${RACECHECK:-0}"

echo "=== [1/3] Smoke Test（CPU-only，真值链自洽）==="
python3 smoke_test.py || exit 1

echo "=== [2/3] fixtures 就位（优先复用 chapter_17 同源；缺失则按同 seed 规则生成）==="
python3 generate_fixtures.py || exit 1

echo "=== [3/3] 各后端：构建 + selftest + 正确性 + 基准 ==="
for backend in $BACKENDS; do
  bash "implementations/$backend/build.sh" || exit 1
  echo "--- $backend 二进制 selftest ---"
  "implementations/$backend/async_gemm_bench" selftest || exit 1
  python3 compare_correctness.py --backend "$backend" || exit 1
  python3 benchmark.py --backend "$backend" || exit 1
  if [ "$RACECHECK" = "1" ] && [ "$backend" = "cuda" ]; then
    echo "--- racecheck 体检（接第 10 期工具链）---"
    (cd implementations/cuda && bash run_racecheck.sh) || echo "[warn] racecheck 采集失败（记录，不阻断）"
    cp implementations/cuda/racecheck-*.log results/ 2>/dev/null || true
  fi
done

echo "=== 完成。结果见 results/ ==="
