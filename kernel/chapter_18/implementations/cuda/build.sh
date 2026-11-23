#!/usr/bin/env bash
# 第 18 期 CUDA 构建（RTX 4090 sm_89；WMMA + 第 17 期朴素版同源同二进制）
set -uo pipefail
cd "$(dirname "$0")"
ARCH="${CUDA_ARCH:-sm_89}"
LOG="build-log-cuda.txt"

echo "== nvcc 版本 ==" | tee "$LOG"
nvcc --version 2>&1 | tee -a "$LOG"
echo "== 编译开始 $(date -Iseconds) ==" | tee -a "$LOG"
START=$(date +%s.%N)
# shellcheck disable=SC2086
nvcc -O3 -std=c++17 -arch="$ARCH" --resource-usage \
    -o naive_gemm_bench naive_gemm_wmma.cu 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
END=$(date +%s.%N)
COMPILE_S=$(echo "$END $START" | awk '{printf "%.1f", $1-$2}')
echo "== 编译结束 $(date -Iseconds)，耗时 ${COMPILE_S}s，rc=$RC ==" | tee -a "$LOG"
echo "{\"compile_wall_seconds\": ${COMPILE_S}, \"arch\": \"${ARCH}\"}" > compile-time.json

echo "== 资源占用摘要（wmma 实例）==" | tee -a "$LOG"
grep -E "gemm_wmma" "$LOG" | head -30 | tee -a "$LOG"
exit "$RC"
