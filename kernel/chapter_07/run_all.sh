#!/usr/bin/env bash
# 第 07 期一键验收（在已配置好 Python 环境的机器上执行）。
# 用法：
#   BACKENDS="cpu" ./run_all.sh            # 本地（CPU 真值 + fixtures 生成 + CPU 列实验）
#   BACKENDS="cuda" ./run_all.sh           # 4090 服务器（build + CUDA 列）
#   BACKENDS="musa" ./run_all.sh           # S4000 服务器（build + MUSA 列）
#   BACKENDS="cuda" SKIP_FIXTURES=1 ./run_all.sh   # fixtures 已就位时跳过生成
set -euo pipefail
cd "$(dirname "$0")"

BACKENDS="${BACKENDS:-cpu}"

echo "=== [1/5] Smoke Test（CPU-only）==="
python3 smoke_test.py

echo "=== [2/5] Fixtures（CPU，固定 seed；跨服务器共用同一份）==="
if [ "${SKIP_FIXTURES:-0}" = "1" ]; then
  echo "SKIP_FIXTURES=1：使用已上传的 fixtures/"
else
  python3 generate_fixtures.py
fi

echo "=== [3/5] CPU 列专属实验（sum-order / drift-curve）==="
python3 exp_cpu_sum_order.py
python3 exp_cpu_drift_curve.py

echo "=== [4/5] 构建各后端实现 ==="
for backend in $BACKENDS; do
  if [ "$backend" = "cuda" ]; then
    (cd implementations/cuda && ./build.sh)
  elif [ "$backend" = "musa" ]; then
    (cd implementations/musa && ./build.sh)
  fi
done

echo "=== [5/5] 各后端正确性对照 + 性能 + 确定性 ==="
for backend in $BACKENDS; do
  python3 compare_correctness.py --backend "$backend"
  if [ "$backend" = "cpu" ]; then
    python3 benchmark.py --backend "$backend" --perf --determinism
  else
    python3 benchmark.py --backend "$backend" --perf --determinism --curve --splits-scan
  fi
  python3 exp_torch_deterministic.py --backend "$backend"
done

echo "=== bench 通道（GPU 后端：大形状真值漂移 + nsys 对象）==="
for backend in $BACKENDS; do
  if [ "$backend" = "cuda" ]; then
    BENCH=implementations/cuda/build/det_atomic_bench
    PEAK=1008.0
  elif [ "$backend" = "musa" ]; then
    BENCH=implementations/musa/build/det_atomic_bench
    PEAK=768.0
  else
    continue
  fi
  for impl in det shared global; do
    "$BENCH" --mode determinism --impl "$impl" --rows 128 --h 65536 --dtype fp16 \
      --repeats 100 --peak-gbs "$PEAK" --device-tag "$backend" \
      --json-file "results/bench-determinism-$backend.jsonl"
    "$BENCH" --mode determinism --impl "$impl" --rows 2048 --h 65536 --dtype fp16 \
      --repeats 100 --peak-gbs "$PEAK" --device-tag "$backend" \
      --json-file "results/bench-determinism-$backend.jsonl"
    "$BENCH" --mode perf --impl "$impl" --rows 2048 --h 1024 --dtype fp16 \
      --repeats 100 --peak-gbs "$PEAK" --device-tag "$backend" \
      --json-file "results/bench-perf-$backend.jsonl"
    "$BENCH" --mode perf --impl "$impl" --rows 2048 --h 65536 --dtype fp16 \
      --repeats 50 --peak-gbs "$PEAK" --device-tag "$backend" \
      --json-file "results/bench-perf-$backend.jsonl"
  done
done

echo "=== 完成。结果见 results/ ==="
