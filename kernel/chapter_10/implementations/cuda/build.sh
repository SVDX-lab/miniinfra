#!/usr/bin/env bash
# ============================================================================
# build.sh —— 第 10 期 CUDA 列构建（RTX 4090 / SM 8.9 / CUDA 13.0）。
#
# 产物：
#   build/defect_suite           四类缺陷 Kernel 集合（memcheck/racecheck/
#                                initcheck/synccheck 体检体）
#   build/printf_trap            printf 陷阱演示（order/trunc/frag/cost）
#   build/health_ch04            ch04 融合 kernel 体检靶（逐字节同源复制）
#   build/health_ch08            ch08 det/atomic 体检靶（逐字节同源复制）
#   build/rmsnorm_bench          ch03 期原样 bench（开销测量被测对象 +
#                                ch03 体检靶；--check 自检口径原样）
#   build/defect_suite_lineinfo  defect_suite 的 -lineinfo 版（源级关联对照）
#
# -lineinfo 不改变优化（-O3 保持），只注入源行号到 SASS 映射——
# sanitizer 报告从裸地址升级为 file:line 的开关即此 flag。
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CUDA_ARCH="${CUDA_ARCH:-89}"
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
COMMON=(-O3 -std=c++17 -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")

echo "[build] defect_suite（含 bad/good 全 case）"
nvcc "${COMMON[@]}" defect_suite.cu -o "$BUILD_DIR/defect_suite"

echo "[build] defect_suite -lineinfo 版（源级关联对照）"
nvcc "${COMMON[@]}" -lineinfo defect_suite.cu -o "$BUILD_DIR/defect_suite_lineinfo"

echo "[build] printf_trap"
nvcc "${COMMON[@]}" printf_trap.cu -o "$BUILD_DIR/printf_trap"

echo "[build] health_ch04（fused_rmsnorm_kernels.cu 逐字节同源）"
nvcc "${COMMON[@]}" -DCH11_TARGET_CH04 health_targets.cu -o "$BUILD_DIR/health_ch04"

echo "[build] health_ch08（kernels_lib.cu 逐字节同源）"
nvcc "${COMMON[@]}" -DCH11_TARGET_CH08 health_targets.cu -o "$BUILD_DIR/health_ch08"

echo "[build] rmsnorm_bench（ch03 期原样：开销测量 + ch03 体检）"
nvcc "${COMMON[@]}" rmsnorm_bench.cu rmsnorm_kernel.cu -o "$BUILD_DIR/rmsnorm_bench"

nvcc --version | tail -2
echo "[build] 全部完成"
