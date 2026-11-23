// P3 探针：cp.async.cg.shared.global（16B 向量拷贝）在 mcc 是否接受 + 跑对
#include <cstdio>
#include <musa_runtime.h>

__global__ void cpasync16_probe_kernel(const float* g, float* out) {
    __shared__ float tile[128];  // 512B = 32 线程 × 16B
    unsigned int dst = (unsigned int)__cvta_generic_to_shared(&tile[threadIdx.x * 4]);
    const float* src = g + threadIdx.x * 4;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < 128; ++i) s += tile[i];
        out[0] = s;
    }
}

int main() {
    const int N = 128;
    float h[N];
    float expected = 0.0f;
    for (int i = 0; i < N; ++i) {
        h[i] = i * 0.25f;
        expected += h[i];
    }
    float *dIn = nullptr, *dOut = nullptr;
    if (musaMalloc((void**)&dIn, N * sizeof(float)) != musaSuccess ||
        musaMalloc((void**)&dOut, sizeof(float)) != musaSuccess) {
        printf("P3-FAIL musaMalloc\n");
        return 2;
    }
    musaMemcpy(dIn, h, N * sizeof(float), musaMemcpyHostToDevice);
    cpasync16_probe_kernel<<<1, 32>>>(dIn, dOut);
    musaError_t err = musaDeviceSynchronize();
    if (err != musaSuccess) {
        printf("P3-FAIL launch/sync: %s\n", musaGetErrorString(err));
        return 2;
    }
    float got = -1.0f;
    musaMemcpy(&got, dOut, sizeof(float), musaMemcpyDeviceToHost);
    bool ok = (got - expected) < 1e-3f && (expected - got) < 1e-3f;
    printf("P3 got=%.2f expected=%.2f\n", got, expected);
    printf("P3-%s\n", ok ? "PASS cp.async.cg 16B 全链可用" : "FAIL 数据不一致");
    musaFree(dIn);
    musaFree(dOut);
    return ok ? 0 : 3;
}
