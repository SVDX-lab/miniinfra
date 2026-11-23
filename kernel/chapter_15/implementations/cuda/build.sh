#!/usr/bin/env bash
# 第 15 期 CUDA 构建（RTX 4090，sm_89）——【待 4090 修复后执行】
# 产物：int8_gemm_bench；编译日志（含 --resource-usage 占用）写 build-log-cuda.txt。
set -uo pipefail
cd "$(dirname "$0")"

ARCH="${CUDA_ARCH:-sm_89}"
LOG="build-log-cuda.txt"

echo "== nvcc 版本 ==" | tee "$LOG"
nvcc --version 2>&1 | tee -a "$LOG"

echo "== 编译开始 $(date -Iseconds) ==" | tee -a "$LOG"
START=$(date +%s.%N)
nvcc -O3 -std=c++17 -arch="$ARCH" --resource-usage \
    -o int8_gemm_bench int8_gemm.cu 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
END=$(date +%s.%N)

COMPILE_S=$(echo "$END $START" | awk '{printf "%.1f", $1-$2}')
echo "== 编译结束 $(date -Iseconds)，耗时 ${COMPILE_S}s，rc=$RC ==" | tee -a "$LOG"
echo "{\"compile_wall_seconds\": ${COMPILE_S}, \"arch\": \"${ARCH}\"}" > compile-time.json

echo "== 资源占用摘要 ==" | tee -a "$LOG"
grep -E "Function properties|registers|smem|spill" "$LOG" | head -80 | tee -a "$LOG"
exit "$RC"
