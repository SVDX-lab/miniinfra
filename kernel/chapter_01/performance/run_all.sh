#!/usr/bin/env bash
# 第 01 期性能诊断，位于第一期 performance/ 目录。
# 用法：
#   BACKENDS="cpu cuda" ./run_all.sh          # 4090 服务器（CUDA 列 + ncu 证据）
#   BACKENDS="cpu musa" ./run_all.sh          # S4000 服务器（MUSA 列 + profiler 探索）
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

BACKENDS="${BACKENDS:-cpu}"
for backend in $BACKENDS; do
  case "$backend" in
    cpu|cuda|musa) ;;
    *)
      echo "不支持的后端：$backend" >&2
      exit 2
      ;;
  esac
done
MODEL_DIR_CANDIDATES=("models/qwen3-0.6b" "../models/qwen3-0.6b")
MODEL_DIR=""
for candidate in "${MODEL_DIR_CANDIDATES[@]}"; do
  if [ -f "$candidate/model.safetensors" ]; then MODEL_DIR="$candidate"; break; fi
done

echo "=== [1/6] Smoke Test（CPU-only 方法论自检）==="
python3 smoke_test.py

echo "=== [2/6] 生成 fixtures（CPU，固定 seed）==="
python3 generate_fixtures.py

echo "=== [3/6] 设备属性/概念探针（concept-map 证据）==="
if [[ " $BACKENDS " == *" cuda "* ]]; then
  command -v nvcc >/dev/null 2>&1 || export PATH="/usr/local/cuda/bin:$PATH"
  if nvcc -O2 -o results/device_query_cuda implementations/cuda/device_query.cu 2> results/device-query-cuda-build.log; then
    ./results/device_query_cuda | tee results/device-query-cuda.txt
  else
    echo "nvcc 构建失败（见 results/device-query-cuda-build.log）"
  fi
fi
if [[ " $BACKENDS " == *" musa "* ]]; then
  # mcc 构建按候选命令逐个尝试，成败与报错全文留档（迁移台账素材）
  BUILT=0
  for cmd in \
    "mcc -O2 -o results/device_query_musa implementations/musa/device_query.mu" \
    "mcc -O2 -o results/device_query_musa implementations/musa/device_query.mu -lmusart" \
    "mcc -O2 --musa -o results/device_query_musa implementations/musa/device_query.mu"; do
    echo "尝试: $cmd"
    if $cmd > results/device-query-musa-build.log 2>&1; then BUILT=1; echo "  构建成功"; break; fi
    echo "  失败（报错存 results/device-query-musa-build.log）"
    cp results/device-query-musa-build.log "results/device-query-musa-build-fail-$BUILT.log"
  done
  if [ "$BUILT" = "1" ]; then
    ./results/device_query_musa | tee results/device-query-musa.txt
  else
    echo "mcc 全部候选构建命令失败，报错已留档（平台边界记录）"
  fi
fi

echo "=== [4/6] 各后端：峰值自测 + Baseline + launch-bound 场景 ==="
for backend in $BACKENDS; do
  python3 compare_correctness.py --backend "$backend"
  if [ "$backend" != "cpu" ]; then
    python3 measure_peak.py --backend "$backend"
  fi
  python3 benchmark.py --backend "$backend"
  python3 "implementations/${backend}/launch_bound_scenario.py" --backend "$backend"
done

echo "=== [5/6] Profiler 证据 ==="
if [[ " $BACKENDS " == *" cuda "* ]]; then
  if ! bash implementations/cuda/run_ncu.sh; then
    echo "Nsight Compute 未完成，详见 results/ 中的日志。" >&2
  fi
fi
if [[ " $BACKENDS " == *" musa "* ]]; then
  if ! python3 implementations/musa/musa_profiler_probe.py; then
    echo "MUSA 性能探针未完成，详见 results/ 中的日志。" >&2
  fi
fi

echo "=== [6/6] 估算题演练（自动核版）==="
if [ -n "$MODEL_DIR" ]; then
  python3 estimate_qwen3.py --model-dir "$MODEL_DIR"
else
  echo "未找到模型权重，估算题仅做公式计算（无文件系统校核）"
  python3 estimate_qwen3.py
fi

echo "=== 完成。结果见 results/ ==="
