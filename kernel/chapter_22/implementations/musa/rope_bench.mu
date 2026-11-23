// ============================================================================
// rope_bench.mu —— 第 22 期 MUSA 列 standalone 入口（S4000，ch18/ch21 模式）。
//
// 与 CUDA 列共用全部核与入口体：
//   rope_kernels.cuh      —— 三份核 + 转换核 + 探针（本文件 include）；
//   rope_bench_main.cuh   —— hash-RNG/host 真值/六模式/main（本文件 include）。
// 平台差异集中在本 shim 块（机械替换单点化；与 rope_bench.cu 逐行对照即
// cross-backend-comparison 的 diff 基线）。
//
// 入口宏：.mu 必须自带 #define USE_MUSA 1（ch18/ch21 实录：缺它走 #else 分支
// 报 fatal error: 'cuda_runtime.h' file not found）。
//
// 用法：./rope_bench {attrs|selftest|correctness|cachedtype|bench|apply desc} [musa]
// ============================================================================

#define USE_MUSA 1
#define RPE_NO_FP16_INCLUDE 1

#include <musa_runtime.h>
#if defined(__has_include)
  #if __has_include(<musa_fp16.h>)
    #include <musa_fp16.h>
    #define HAVE_MUSA_FP16_H 1
  #endif
#endif

// ---- gpu* shim（ch09 实测 musa* 传输族机械替换兼容率 100%；M-34）----
#define gpuError_t             musaError_t
#define gpuSuccess             musaSuccess
#define gpuGetErrorString      musaGetErrorString
#define gpuDeviceProp_t        musaDeviceProp
#define gpuGetDeviceProperties musaGetDeviceProperties
#define gpuMalloc              musaMalloc
#define gpuFree                musaFree
#define gpuMemcpy              musaMemcpy
#define gpuMemcpyHostToDevice  musaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  musaMemcpyDeviceToHost
#define gpuEvent_t             musaEvent_t
#define gpuEventCreate         musaEventCreate
#define gpuEventDestroy        musaEventDestroy
#define gpuEventRecord         musaEventRecord
#define gpuEventSynchronize    musaEventSynchronize
#define gpuEventElapsedTime    musaEventElapsedTime
#define gpuDeviceSynchronize   musaDeviceSynchronize
#define gpuGetLastError        musaGetLastError

#include "../cuda/rope_bench_main.cuh"
