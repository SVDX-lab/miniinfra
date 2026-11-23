// 第 15 期 C++ stub 语法检查用假 runtime 头（本地无 nvcc/mcc；
// 与 ep21/ep22 的 stub 检查同思路：把 <<<launch>>> 语法 sed 成普通函数调用，
// 全文件做 g++ -fsyntax-only 类型检查——抓 C++ 层错误，不抓 device codegen）。
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

// 启动语法替身：stub_launch(kernel, grid, block)(args) → 直接调用 kernel(args)
template <typename F>
F stub_launch(F f, dim3, dim3) {
    return f;
}

typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;
inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }

struct cudaDeviceProp {
    char name[256];
    int multiProcessorCount;
    int warpSize;
};
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int) { return cudaSuccess; }

template <typename T>
cudaError_t cudaMalloc(T** /*devPtr*/, size_t /*size*/) { return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }

enum cudaMemcpyKind {
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
};

inline cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) { return cudaSuccess; }

struct cudaEvent_t_stub {};
typedef cudaEvent_t_stub* cudaEvent_t;
inline cudaError_t cudaEventCreate(cudaEvent_t*) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t, int) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
    *ms = 0.0f;
    return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

struct cudaFuncAttributes {
    int numRegs;
    size_t sharedSizeBytes;
    size_t localSizeBytes;
    int maxThreadsPerBlock;
};
inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, const void*) { return cudaSuccess; }

inline void __syncthreads() {}

struct uint3_stub {
    unsigned x, y, z;
};
inline uint3_stub threadIdx, blockIdx, blockDim;

#define __global__
#define __device__
#define __host__
#define __shared__
