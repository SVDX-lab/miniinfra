#!/usr/bin/env bash
# 第 16 期 MUSA Kernel 构建：mcc 编译为 C ABI 共享库（与 cuda/build.sh 逐行对应）。
# 注意：MUSA_ARCH 需实机探测（S4000 为 MUSA CC 2.2）；探测过程记录进 EXPERIMENT_LOG。
set -euo pipefail
cd "$(dirname "$0")"

MCC="${MCC:-mcc}"
# 与 nvcc -arch=sm_89 对应的 mcc 架构开关（实机探测 2026-11-20）：
# torch_musa/utils/musa_extension.py:863 用 --offload-arch=mp_{arch}，
# S4000 设备能力 (2,2) → mp_22（受支持列表 ["10","21","22","31","32"]）。
ARCH_ARGS=("--offload-arch=${MUSA_ARCH:-mp_22}")

"$MCC" -O3 -std=c++17 -shared -fPIC "${ARCH_ARGS[@]}" \
  ops_kernels.mu -o libops_musa.so

echo "[build] libops_musa.so 构建完成（arch args: ${ARCH_ARGS[*]}）"
"$MCC" --version | head -3
