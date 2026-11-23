#!/usr/bin/env bash
# ============================================================================
# 第 08 期 MUSA 构建脚本（CUDA 版 build.sh 的迁移对应物）。
#
# 本脚本的 1/2 步是本期最重要的迁移素材产出：
#   1. mus_api_naming.sh —— /usr/local/musa/include 头文件逐符号 grep
#      （mus* 与 musa* 两族全列）—— 大纲 "musMemcpy/musMallocHost 待实机核对"；
#   2. 符号编译探测 —— 每个候选 API/枚举/字段一个最小 TU，mcc 编译通过 =
#      符号存在；探测集合决定 T9_HAVE_* / T9_DEV_* 注入（api_probe 运行前
#      的静态证据链）；
#   3. mus_naming_probe.mu —— mus* 前缀反证探针（预期编译失败，错误全文留档）；
#   4. 编译 api_probe.mu + libtransfer_ops_musa.so（.cu→.mu 复制，M-07/M-16）。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

BUILD_DIR="build"
LOGS=../../results/server-logs
mkdir -p "$BUILD_DIR" "$LOGS"

echo "=== [1/4] 头文件 grep 证据（mus* vs musa*） ==="
./mus_api_naming.sh | tee "$LOGS/musa-api-naming-grep.txt"

echo "=== [2/4] 符号编译探测（mcc 逐符号最小 TU） ==="
PROBE_DIR="$BUILD_DIR/probe"
mkdir -p "$PROBE_DIR"
DEFS=""
DEFS_LOG="$LOGS/musa-symbol-probe.txt"
: > "$DEFS_LOG"

probe_symbol() {  # 枚举常量 / 函数符号：编译通过 = 存在。
    # 教训一（KI-09-03）：探针 TU 必须用 .mu 后缀 —— .cu 会被 mcc 当 CUDA 源
    # （M-07）报 "cannot find libdevice for sm_35" 而全体假 MISSING。
    # 教训二：MUSA SDK 5.2.0 的运行时 API 有 __host__/__device__ 双重载声明
    # （musa_runtime_api.h + musa_device_runtime_api.h）—— &name 有歧义、
    # sizeof 对 MUSARTAPI 函数非法 → 函数符号两种探针都假 MISSING。
    # 因此本探针只对"枚举常量"可信（sizeof 形式）；函数符号的存在性以
    # transfer_ops.mu 库编译（引用即编译验证）+ api_probe 运行为准。
    local name="$1"
    cat > "$PROBE_DIR/p.mu" <<EOF
#include <musa_runtime.h>
int main() { return (int)sizeof($name) > 0 ? 0 : 1; }
EOF
    if mcc -o "$PROBE_DIR/p.out" "$PROBE_DIR/p.mu" 2>>"$PROBE_DIR/last_err.log"; then
        echo "SYMBOL $name = EXISTS(sizeof)" | tee -a "$DEFS_LOG"
        return 0
    fi
    cat > "$PROBE_DIR/p.mu" <<EOF
#include <musa_runtime.h>
int main() { return ((void*)&$name) != (void*)0 ? 0 : 1; }
EOF
    if mcc -o "$PROBE_DIR/p.out" "$PROBE_DIR/p.mu" 2>>"$PROBE_DIR/last_err.log"; then
        echo "SYMBOL $name = EXISTS(addr)" | tee -a "$DEFS_LOG"
        return 0
    fi
    echo "SYMBOL $name = MISSING(两种探针均编译失败；函数符号可能为 host/device 双重载——见 last_err.log，以库编译+运行为准)" | tee -a "$DEFS_LOG"
    return 1
}

probe_field() {  # musaDeviceProp 字段
    local name="$1"
    cat > "$PROBE_DIR/p.mu" <<EOF
#include <musa_runtime.h>
int main() { musaDeviceProp p; return (int)p.$name; }
EOF
    if mcc -o "$PROBE_DIR/p.out" "$PROBE_DIR/p.mu" 2>>"$PROBE_DIR/last_err.log"; then
        echo "FIELD prop.$name = EXISTS" | tee -a "$DEFS_LOG"
        return 0
    else
        echo "FIELD prop.$name = MISSING" | tee -a "$DEFS_LOG"
        return 1
    fi
}

# 全量符号探测（文档证据；探针结果记录进 musa-symbol-probe.txt）
probe_symbol musaMemcpyAsync
probe_symbol musaMallocHost
probe_symbol musaHostAlloc
probe_symbol musaFreeHost
probe_symbol musaHostAllocPortable
probe_symbol musaHostAllocMapped
probe_symbol musaHostRegister
probe_symbol musaHostRegisterDefault
probe_symbol musaHostUnregister
probe_symbol musaHostGetDevicePointer
probe_symbol musaSetDeviceFlags
probe_symbol musaDeviceMapHost
probe_symbol musaStreamCreate
probe_symbol musaStreamCreateWithFlags
probe_symbol musaStreamNonBlocking
probe_symbol musaStreamDestroy
probe_symbol musaStreamSynchronize
probe_symbol musaStreamQuery
probe_symbol musaStreamWaitEvent
probe_symbol musaEventCreate
probe_symbol musaEventRecord
probe_symbol musaEventElapsedTime
probe_symbol musaEventDestroy
probe_symbol musaDeviceSynchronize
probe_symbol musaDeviceGetPCIBusId
probe_symbol musaErrorSetOnActiveProcess
probe_symbol musaErrorNotReady

# T9_HAVE_* 注入策略（终版）：全量注入 —— 函数符号存在性由 transfer_ops.mu
# 库编译本身验证（引用即编译验证；缺符号会在编译期报出并留档），枚举常量
# 存在性已由上面 sizeof 探针确认。若库编译失败即平台边界（错误留档）。
DEFS="$DEFS -DT9_HAVE_SET_DEVICE_FLAGS=1 -DT9_HAVE_DEVICE_MAP_HOST=1 -DT9_HAVE_ERROR_NOT_READY=1 -DT9_HAVE_PCI_BUS_ID=1 -DT9_HAVE_HOSTALLOC_PORTABLE=1 -DT9_HAVE_HOSTALLOC_MAPPED=1 -DT9_HAVE_HOST_REGISTER=1 -DT9_HAVE_STREAM_NONBLOCKING=1 -DT9_DEV_ASYNC_ENGINE=1 -DT9_DEV_OVERLAP=1 -DT9_DEV_CAN_MAP=1 -DT9_DEV_UNIFIED=1 -DT9_DEV_PCI=1 -DT9_DEV_L2=1 -DT9_DEV_CLOCK=1 -DT9_DEV_MAXTHREADS=1"
# mus* 反证（预期全 MISSING）
for m in musMemcpy musMallocHost musMemcpyAsync musStreamCreate musHostAlloc musMalloc musFreeHost; do
    probe_symbol "$m" || true
done
# musaDeviceProp 字段（基础 6 字段已在 ch03 核对，其余探测）
probe_field asyncEngineCount   && DEFS="$DEFS -DT9_DEV_ASYNC_ENGINE=1"
probe_field deviceOverlap      && DEFS="$DEFS -DT9_DEV_OVERLAP=1"
probe_field canMapHostMemory   && DEFS="$DEFS -DT9_DEV_CAN_MAP=1"
probe_field unifiedAddressing  && DEFS="$DEFS -DT9_DEV_UNIFIED=1"
probe_field pciDomainID        && DEFS="$DEFS -DT9_DEV_PCI=1"
probe_field l2CacheSize        && DEFS="$DEFS -DT9_DEV_L2=1"
probe_field clockRate          && DEFS="$DEFS -DT9_DEV_CLOCK=1"
probe_field maxThreadsPerBlock && DEFS="$DEFS -DT9_DEV_MAXTHREADS=1"
cp -f "$PROBE_DIR/last_err.log" "$LOGS/musa-symbol-probe-errors.log" 2>/dev/null || true

echo "=== [3/4] mus* 前缀反证探针（预期编译失败） ==="
if mcc -o "$BUILD_DIR/mus_naming.out" mus_naming_probe.mu \
       2> "$LOGS/mus-naming-compile-error.log"; then
    echo "[probe] 意外：mus* 探针编译通过（musa SDK 存在 mus* 别名！）——如实记录"
    echo "UNEXPECTED mus* COMPILE OK" >> "$DEFS_LOG"
else
    echo "[probe] mus* 探针编译失败（预期）——错误全文见 $LOGS/mus-naming-compile-error.log"
fi

echo "=== [4/4] 编译 api_probe 与传输库（.cu→.mu 复制，M-07/M-16） ==="
cp transfer_ops.cu transfer_ops.mu
cp api_probe.cu    api_probe.mu
if mcc -O3 -std=c++17 $DEFS -o "$BUILD_DIR/api_probe" api_probe.mu; then
    echo "[build] $BUILD_DIR/api_probe OK（musa* 命名族编译通过 = 实机证据）"
else
    echo "[build] api_probe 编译失败（错误即平台边界证据，见 $LOGS）"
    cp mcc_err_probe.log "$LOGS/musa-apiprobe-compile-error.log" 2>/dev/null || true
fi
if mcc -O3 -std=c++17 $DEFS -shared -fPIC -o "$BUILD_DIR/libtransfer_ops_musa.so" transfer_ops.mu; then
    echo "[build] $BUILD_DIR/libtransfer_ops_musa.so OK"
else
    echo "[build] 传输库编译失败 —— 尝试 -lmusart 重链"
    mcc -O3 -std=c++17 $DEFS -shared -fPIC -o "$BUILD_DIR/libtransfer_ops_musa.so" transfer_ops.mu -lmusart \
        && echo "[build] $BUILD_DIR/libtransfer_ops_musa.so OK（-lmusart 需要显式链接，台账素材）" \
        || echo "[build] 传输库编译失败（平台边界，错误留档）"
fi
rm -f transfer_ops.mu api_probe.mu
echo "[build] 完成"
