#!/usr/bin/env bash
# ============================================================================
# 第 03 期 MUSA 构建（CUDA 版 build.sh 的迁移对应物，M-17 ctypes 直传模式）。
#
# 构建选择记录（理由：第 16 期已验证"Kernel 库零 Runtime API 调用 + ctypes
# 直传"是迁移差异面最小的模式——M-17：443 行仅 5 行注释差异、mcc 一次编译
# 通过；本期 kernel 库同样零 Runtime API 调用，故选同一模式。对照：
# torch musa_extension 路径有 M-08/M-09/M-10 三条已知坑，第 04 期才需要）：
#   - 编译器：mcc（clang 14 底座），SDK 在 /usr/local/musa；
#   - libmini_fused_musa.so：单编译单元 fused_rmsnorm_kernels.mu
#     （#include rmsnorm_kernel_ep03.mu），零 Runtime API；
#   - fused_bench：独立可执行程序（自带 musaMalloc/Event，bench 程序允许
#     Runtime API；MUSA 侧无 nsys，它只做 Event 计时第二口径 + 自检）；
#   - 架构开关：--offload-arch=mp_22（M-18：mcc CLI 下 -march=mp_22 不可用、
#     --offload-arch 有效；torch_musa musa_extension.py 同款；S4000 CC 2.2）；
#   - 源文件直接以 .mu 后缀入库（M-07：mcc 把 .cu 当 CUDA 源找 CUDA toolkit）。
# 用法：./build.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

MCC="${MCC:-mcc}"
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
ARCH_ARGS=("--offload-arch=${MUSA_ARCH:-mp_22}")

echo "[build] mcc 版本："
mcc --version || true

echo "[build] mcc 构建 libmini_fused_musa.so（arch args: ${ARCH_ARGS[*]}）"
"$MCC" -O3 -std=c++17 -shared -fPIC "${ARCH_ARGS[@]}" \
  fused_rmsnorm_kernels.mu -o libmini_fused_musa.so
echo "[build] libmini_fused_musa.so OK"

echo "[build] mcc 独立 bench：fused_bench"
"$MCC" -O3 -std=c++17 "${ARCH_ARGS[@]}" \
  -o "$BUILD_DIR/fused_bench" fused_bench.mu fused_rmsnorm_kernels.mu
echo "[build] $BUILD_DIR/fused_bench OK"

echo "[build] 完成"
