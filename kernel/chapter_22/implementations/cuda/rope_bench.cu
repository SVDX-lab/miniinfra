// ============================================================================
// rope_bench.cu —— 第 22 期 CUDA 列 standalone 入口（RTX 4090）。
//
// 与 MUSA 列共用全部核与入口体（rope_kernels.cuh + rope_bench_main.cuh）；
// 与 rope_bench.mu 的唯一差异 = 本 shim 块（机械替换对照基线）。
//
// ⚠️ 状态：待 4090 修复后执行（CUDA_PENDING.md）——本文件与 build.sh 已就绪，
// 恢复后 `bash build.sh && bash run_session.sh` 一键完成。
//
// 用法：./rope_bench {attrs|selftest|correctness|cachedtype|bench|apply desc} [cuda]
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---- gpu* shim（CUDA 原生名直通，与 .mu 的 musa* 替换一一对应）----
#define gpuError_t             cudaError_t
#define gpuSuccess             cudaSuccess
#define gpuGetErrorString      cudaGetErrorString
#define gpuDeviceProp_t        cudaDeviceProp
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuMalloc              cudaMalloc
#define gpuFree                cudaFree
#define gpuMemcpy              cudaMemcpy
#define gpuMemcpyHostToDevice  cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost  cudaMemcpyDeviceToHost
#define gpuEvent_t             cudaEvent_t
#define gpuEventCreate         cudaEventCreate
#define gpuEventDestroy        cudaEventDestroy
#define gpuEventRecord         cudaEventRecord
#define gpuEventSynchronize    cudaEventSynchronize
#define gpuEventElapsedTime    cudaEventElapsedTime
#define gpuDeviceSynchronize   cudaDeviceSynchronize
#define gpuGetLastError        cudaGetLastError

#include "../cuda/rope_bench_main.cuh"
