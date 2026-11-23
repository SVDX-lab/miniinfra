#!/usr/bin/env bash
# 第 17 期 CUDA 构建（RTX 4090，sm_89；单层 tiling GEMM）
# 产物：naive_gemm_bench；编译日志（含 --resource-usage 的寄存器/共享内存占用）
# 写入 build-log-cuda.txt——寄存器占用估算的原始证据（面试追问素材）。
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
    -o naive_gemm_bench naive_gemm.cu 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
END=$(date +%s.%N)

COMPILE_S=$(echo "$END $START" | awk '{printf "%.1f", $1-$2}')
echo "== 编译结束 $(date -Iseconds)，耗时 ${COMPILE_S}s，rc=$RC ==" | tee -a "$LOG"
echo "compile_wall_seconds=${COMPILE_S}" > compile-time.txt
echo "{\"compile_wall_seconds\": ${COMPILE_S}, \"arch\": \"${ARCH}\"}" > compile-time.json

# 摘要：每个实例化的寄存器/共享内存占用
echo "== 资源占用摘要 ==" | tee -a "$LOG"
grep -E "Function properties|registers|smem|spill" "$LOG" | head -80 | tee -a "$LOG"
exit "$RC"
