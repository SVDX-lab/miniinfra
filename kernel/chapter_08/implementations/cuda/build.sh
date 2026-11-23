#!/usr/bin/env bash
# ============================================================================
# 第 08 期 CUDA 构建脚本。
#
# 产物：
#   build/api_probe                —— 传输 API 面最小程序（编译验证 + 逐 API 行为探测）
#   build/libtransfer_ops_cuda.so  —— extern "C" 传输/流/算子库（ctypes 直传，M-17 模式）
#
# T9_HAVE_*（函数/枚举符号）CUDA 全量注入；
# T9_DEV_*（cudaDeviceProp 字段）编译探测注入 —— CUDA 13 移除了 deviceOverlap
# 等旧字段（4090 实机编译报 "has no member deviceOverlap"，API 面差异证据），
# 与 MUSA 版 build.sh 的字段探测同方法（对照表素材）。
# 用法：CUDA_ARCH=89 ./build.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="build"
PROBE_DIR="$BUILD_DIR/probe"
mkdir -p "$BUILD_DIR" "$PROBE_DIR"
CUDA_ARCH="${CUDA_ARCH:-89}"

DEFS="-DT9_HAVE_SET_DEVICE_FLAGS=1 -DT9_HAVE_DEVICE_MAP_HOST=1 -DT9_HAVE_ERROR_NOT_READY=1 -DT9_HAVE_PCI_BUS_ID=1 -DT9_HAVE_HOSTALLOC_PORTABLE=1 -DT9_HAVE_HOSTALLOC_MAPPED=1 -DT9_HAVE_HOST_REGISTER=1 \
-DT9_HAVE_STREAM_NONBLOCKING=1"
: > "$PROBE_DIR/fields.txt"
probe_field() {
    local name="$1"
    cat > "$PROBE_DIR/p.cu" <<EOF
#include <cuda_runtime.h>
int main() { cudaDeviceProp p; return (int)p.$name; }
EOF
    if nvcc -O0 -o "$PROBE_DIR/p.out" "$PROBE_DIR/p.cu" 2>>"$PROBE_DIR/last_err.log"; then
        echo "FIELD prop.$name = EXISTS" | tee -a "$PROBE_DIR/fields.txt"
        DEFS="$DEFS -DT9_DEV_$(echo "$name" | awk '{print toupper($0)}')=1"
    else
        echo "FIELD prop.$name = MISSING（CUDA 13 移除，API 面差异实机证据）" | tee -a "$PROBE_DIR/fields.txt"
    fi
}
probe_field asyncEngineCount
probe_field deviceOverlap
probe_field canMapHostMemory
probe_field unifiedAddressing
probe_field pciDomainID
probe_field l2CacheSize
probe_field clockRate
probe_field maxThreadsPerBlock

echo "[build] nvcc arch=sm_${CUDA_ARCH}"
echo "[build] DEFS=$DEFS"
nvcc -O3 -std=c++17 -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
     $DEFS -o "$BUILD_DIR/api_probe" api_probe.cu
echo "[build] $BUILD_DIR/api_probe OK"

nvcc -O3 -std=c++17 -gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" \
     $DEFS -shared -Xcompiler -fPIC -o "$BUILD_DIR/libtransfer_ops_cuda.so" transfer_ops.cu
echo "[build] $BUILD_DIR/libtransfer_ops_cuda.so OK"
echo "[build] 完成（api_probe 运行与正式基准由 run_all.sh 编排）"
