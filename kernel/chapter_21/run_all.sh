#!/usr/bin/env bash
# ============================================================================
# 第 21 期一键验收（BACKENDS="cpu cuda" / "cpu musa" 风格，参照 ch21）。
# 用法：
#   BACKENDS="cpu" PY=/home/aiuser/venvs/ml/bin/python bash run_all.sh   # 本地
#   BACKENDS="cpu cuda" PY=python3 bash run_all.sh                       # 4090
#   BACKENDS="cpu musa" PY=python3 bash run_all.sh                       # S4000
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
BACKENDS="${BACKENDS:-cpu}"
PY="${PY:-python3}"

echo "=== [0] fixtures 校验（--verify-only）==="
"$PY" generate_fixtures.py --verify-only || "$PY" generate_fixtures.py

echo "=== [1] kernel_sim（核算法仿真 + grid/块表覆盖断言）==="
"$PY" implementations/cpu/kernel_sim.py || exit 1

echo "=== [2] smoke test ==="
if [[ "$BACKENDS" == *"cuda"* ]]; then "$PY" smoke_test.py --backend cuda || exit 1;
else "$PY" smoke_test.py || exit 1; fi

echo "=== [3] 实验2：显存碎片率记账模拟 ==="
"$PY" fragment_benchmark.py || exit 1

echo "=== [4] 实验4：KV 字节核算与容量对照 ==="
"$PY" capacity_accounting.py || exit 1

echo "=== [5] correctness ==="
if [[ "$BACKENDS" == *"musa"* ]]; then
  echo "（MUSA 列 correctness 由 implementations/musa/paged_bench selftest+correctness 承担）"
  (cd implementations/musa && ./paged_bench attrs && ./paged_bench selftest) || exit 1
fi
if [[ "$BACKENDS" == *"cuda"* ]]; then "$PY" compare_correctness.py --backend cuda || exit 1; fi
"$PY" compare_correctness.py --backend cpu || exit 1

echo "=== [6] benchmark ==="
for BACKEND in $BACKENDS; do
  "$PY" benchmark.py --backend "$BACKEND" || exit 1
done

echo "=== 完成：results/ 下 correctness-results-*.json / benchmark-results-*.json /"
echo "    fragmentation-results.json / capacity-accounting.json ==="
