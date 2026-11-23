#!/usr/bin/env bash
# 第 02 期 Nsight Compute 证据采集（CUDA 4090）。
# 在 chapter_01/performance/ 目录执行：bash implementations/cuda/run_ncu.sh [ROWS] [HIDDEN]
# 产出（results/ 下）：
#   ncu-rmsnorm-cuda-all.csv      全 kernel 原始指标（CSV，供逐 kernel 流量分析）
#   ncu-rmsnorm-cuda.txt          SpeedOfLight + MemoryWorkloadAnalysis 文本证据
#   ncu-stderr.log                ncu 标准错误（权限/版本问题在此可见）
set -uo pipefail

ROWS="${1:-2048}"
HIDDEN="${2:-1024}"
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd)"  # 性能诊断目录，供 import rmsnorm_reference
TARGET="python3 implementations/cuda/ncu_profile_rmsnorm.py --rows ${ROWS} --hidden ${HIDDEN} --warmup 3 --iters 3"
mkdir -p results

METRICS="gpu__time_duration.sum,dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_bytes.sum,l1tex__t_bytes.sum,launch__grid_size,launch__block_size"

echo "=== [1/2] 全 kernel CSV 指标（含 setup+warmup，上限 80 kernel）==="
ncu --target-processes all --csv --launch-count 80 --metrics "${METRICS}" \
    ${TARGET} > results/ncu-rmsnorm-cuda-all.csv 2> results/ncu-stderr.log
echo "exit=$? rows=$(wc -l < results/ncu-rmsnorm-cuda-all.csv)"

echo "=== [2/2] SpeedOfLight + MemoryWorkloadAnalysis 文本（launch-count 80）==="
ncu --target-processes all --launch-count 80 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section LaunchStats \
    ${TARGET} > results/ncu-rmsnorm-cuda.txt 2>&1
echo "exit=$? lines=$(wc -l < results/ncu-rmsnorm-cuda.txt)"
echo "=== 完成，输出在 results/ ==="
