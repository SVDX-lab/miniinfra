// P4 探针：pipeline intrinsics 头文件（__pipeline_memcpy_async 家族）是否存在
// 按 __has_include 依次探测 musa_pipeline.h / musa_pipeline_primitives.h /
// cuda_pipeline.h / cuda_pipeline_primitives.h；命中则用 intrinsic 跑一遍。
#include <cstdio>
#include <musa_runtime.h>

#if defined(__has_include)
#if __has_include(<musa_pipeline.h>)
#include <musa_pipeline.h>
#define FOUND_PIPELINE_HEADER 1
#define PIPELINE_HEADER "<musa_pipeline.h>"
#elif __has_include(<musa_pipeline_primitives.h>)
#include <musa_pipeline_primitives.h>
#define FOUND_PIPELINE_HEADER 1
#define PIPELINE_HEADER "<musa_pipeline_primitives.h>"
#elif __has_include(<cuda_pipeline_primitives.h>)
#include <cuda_pipeline_primitives.h>
#define FOUND_PIPELINE_HEADER 1
#define PIPELINE_HEADER "<cuda_pipeline_primitives.h>"
#elif __has_include(<cuda_pipeline.h>)
#include <cuda_pipeline.h>
#define FOUND_PIPELINE_HEADER 1
#define PIPELINE_HEADER "<cuda_pipeline.h>"
#endif
#endif

#if defined(FOUND_PIPELINE_HEADER)
__global__ void intrinsic_probe_kernel(const float* g, float* out) {
    __shared__ float tile[128];
    if (threadIdx.x < 32) {
        __pipeline_memcpy_async(&tile[threadIdx.x * 4], &g[threadIdx.x * 4], 16);
        __pipeline_commit();
        __pipeline_wait_prior(0);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < 128; ++i) s += tile[i];
        out[0] = s;
    }
}
#endif

int main() {
#if !defined(FOUND_PIPELINE_HEADER)
    printf("P4-ABSENT 未找到 pipeline intrinsics 头文件（musa_pipeline/cuda_pipeline 均无）\n");
    return 1;
#else
    const int N = 128;
    float h[N];
    float expected = 0.0f;
    for (int i = 0; i < N; ++i) {
        h[i] = i * 0.125f;
        expected += h[i];
    }
    float *dIn = nullptr, *dOut = nullptr;
    if (musaMalloc((void**)&dIn, N * sizeof(float)) != musaSuccess ||
        musaMalloc((void**)&dOut, sizeof(float)) != musaSuccess) {
        printf("P4-FAIL musaMalloc\n");
        return 2;
    }
    musaMemcpy(dIn, h, N * sizeof(float), musaMemcpyHostToDevice);
    intrinsic_probe_kernel<<<1, 32>>>(dIn, dOut);
    musaError_t err = musaDeviceSynchronize();
    if (err != musaSuccess) {
        printf("P4-FAIL launch/sync: %s\n", musaGetErrorString(err));
        return 2;
    }
    float got = -1.0f;
    musaMemcpy(&got, dOut, sizeof(float), musaMemcpyDeviceToHost);
    bool ok = (got - expected) < 1e-3f && (expected - got) < 1e-3f;
    printf("P4 header=%s got=%.3f expected=%.3f\n", PIPELINE_HEADER, got, expected);
    printf("P4-%s\n", ok ? "PASS pipeline intrinsics 全链可用"
                          : "FAIL 数据不一致");
    musaFree(dIn);
    musaFree(dOut);
    return ok ? 0 : 3;
#endif
}
