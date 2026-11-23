// ============================================================================
// musa_compat.h —— 第 10 期 MUSA 列：单点 shim（内核 0 改动，M-22 口径）。
//
// cuda/*.cu 缺陷集合与 printf 演示原样 include 本 shim 编译到 MUSA：
//   Runtime API 名：cudaMalloc→musaMalloc 等（M-06 机械替换族，shim 承担）；
//   kernel 语法：__global__/__shared__/__syncthreads/<<<>>> 同构（mcc clang 底座）；
//   printf FIFO limit 枚举是否存在 = 本期编译期探测点（见 musa_fifo_limit_probe.cc），
//   用 -DMUSA_HAS_PRINTF_FIFO_LIMIT=1 开启映射；缺省按"不存在"处理——
//   cudaDeviceSetLimit/GetLimit 映射为记录型 stub（返回不支持错误），
//   对应行为差异照实进台账。
// ============================================================================

#pragma once

#ifndef MUSA_COMPAT_H_
#define MUSA_COMPAT_H_

#include <musa_runtime.h>

// ---- S4000/mp_22 实测怪癖（本 shim 承担，kernel 零改动）----
// 运行时 init 期内部设置 printf FIFO 失败，遗留 sticky error
// （"limit is not supported on this architecture"），用户进程首个
// musaGetLastError 必读到——不冲洗会把正常 launch 误判为失败。
// 静态构造器在 main 前冲洗一次。
static struct Ep11MusaStickyErrorFlush {
    Ep11MusaStickyErrorFlush() { musaGetLastError(); }
} ep11_musa_sticky_flush_instance;

// ---- 机械替换（M-06 族，单点承担）----
#define cudaError_t                musaError_t
#define cudaSuccess                musaSuccess
#define cudaGetErrorString         musaGetErrorString
#define cudaGetLastError           musaGetLastError
#define cudaMalloc                 musaMalloc
#define cudaFree                   musaFree
#define cudaMemset                 musaMemset
#define cudaMemcpy                 musaMemcpy
#define cudaMemcpyHostToDevice     musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost     musaMemcpyDeviceToHost
#define cudaDeviceSynchronize      musaDeviceSynchronize
#define cudaEvent_t                musaEvent_t
#define cudaEventCreate            musaEventCreate
#define cudaEventRecord            musaEventRecord
#define cudaEventElapsedTime       musaEventElapsedTime
#define cudaEventDestroy           musaEventDestroy

#if MUSA_HAS_PRINTF_FIFO_LIMIT
// 枚举名以 musa_fifo_limit_probe.cc 的编译探测结果为准（候选：
// musaLimitPrintfFifoSize / MUSA_LIMIT_PRINTF_FIFO_SIZE），实机确认后填此处。
#define cudaLimitPrintfFifoSize    MUSA_PRINTF_FIFO_LIMIT_ENUM
#define cudaDeviceSetLimit         musaDeviceSetLimit
#define cudaDeviceGetLimit         musaDeviceGetLimit
#else
// "无对应 limit" 路径：行为差异证据本身（记录进 EXPERIMENT_LOG）。
#define cudaLimitPrintfFifoSize    ((musaLimit) -1)
static inline musaError_t musaDeviceSetLimit_stub(musaLimit, size_t) {
    return musaErrorNotSupported;
}
static inline musaError_t musaDeviceGetLimit_stub(musaLimit, size_t* value) {
    if (value) *value = 0;
    return musaErrorNotSupported;
}
#define cudaDeviceSetLimit         musaDeviceSetLimit_stub
#define cudaDeviceGetLimit         musaDeviceGetLimit_stub
#endif

#endif  // MUSA_COMPAT_H_
