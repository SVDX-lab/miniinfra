#!/usr/bin/env bash
# ============================================================================
# 第 07 期 MUSA 构建：mcc 编译 C ABI 共享库 + 独立 bench（CUDA 版 build.sh
# 的迁移对应物）。
#
# 迁移要点（本章实测记录，同步迁移台账）：
#   - .mu 后缀（M-07：mcc 拒绝 .cu，报 cannot find libdevice for sm_35）；
#   - --offload-arch=mp_22（M-18/M-22：唯一有效目标旗标）；
#   - 单 TU kernels_lib.mu（M-16：直接 mcc CLI 编译 .mu 内部 #include 可行）；
#   - API 命名 musa* 前缀（M-06）；
#   - **half atomicAdd（fp16 原子）可用性待实机核对** —— 第一次编译带
#     ACC=half 实例；失败则以 -DEP08_NO_HALF_ATOMIC 重试（ACC=half 档
#     记平台边界，fp32 累加档不受影响）。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

echo "[build] mcc 版本："
mcc --version || true

MUSA_FLAGS="-O3 -std=c++17 --offload-arch=mp_22"

echo "[build] [1/3] libmini_detatomic_musa.so（ctypes 直传，含 ACC=half 实例）"
if mcc $MUSA_FLAGS -shared -Xcompiler -fPIC \
        kernels_lib.mu -o libmini_detatomic_musa.so 2> "$BUILD_DIR/mcc_half_atomic_err.log"; then
  echo "[build] half atomicAdd 可用（ACC=half 档编入，平台边界未触发）"
  echo "HALF_ATOMIC=available" > "$BUILD_DIR/half_atomic_status.txt"
else
  echo "[build] half atomicAdd 编译失败（错误全文 $BUILD_DIR/mcc_half_atomic_err.log），"
  echo "[build] 以 -DEP08_NO_HALF_ATOMIC 重试（ACC=half 档记平台边界）"
  mcc $MUSA_FLAGS -DEP08_NO_HALF_ATOMIC -shared -Xcompiler -fPIC \
      kernels_lib.mu -o libmini_detatomic_musa.so
  echo "HALF_ATOMIC=disabled" > "$BUILD_DIR/half_atomic_status.txt"
fi
echo "[build] libmini_detatomic_musa.so OK"

echo "[build] [2/3] det_atomic_bench（独立 bench，同 half-atomic 开关策略）"
HALF_FLAG=""
if [ -f "$BUILD_DIR/half_atomic_status.txt" ] && grep -q disabled "$BUILD_DIR/half_atomic_status.txt"; then
  HALF_FLAG="-DEP08_NO_HALF_ATOMIC"
fi
if ! mcc $MUSA_FLAGS $HALF_FLAG \
        -o "$BUILD_DIR/det_atomic_bench" det_atomic_bench.mu 2> "$BUILD_DIR/mcc_bench_err.log"; then
  echo "[build] bench 首选旗标失败，检查 mcc_bench_err.log 中非 half-atomic 项"
  cat "$BUILD_DIR/mcc_bench_err.log"
  exit 1
fi
echo "[build] $BUILD_DIR/det_atomic_bench OK"

echo "[build] [3/3] 半原子符号核对（实机 API 记录）"
if grep -q "atomicAdd" /usr/local/musa/include/musa_fp16.h 2>/dev/null; then
  echo "[build] musa_fp16.h 中发现 atomicAdd 符号（头文件级证据）"
else
  echo "[build] musa_fp16.h 中未发现 atomicAdd（头文件级证据，与编译结果对照）"
fi
mcc --version | tail -2
echo "[build] 完成"
