// P2 探针：cp.async.ca.shared.global（4B）内联 PTX 在 mcc 是否接受 + 跑对
// 判据：编过 + 128 个 float 异步拷入 shared + 求和与 host 一致
#include <cstdio>
#include <musa_runtime.h>

__global__ void cpasync4_probe_kernel(const float* g, float* out) {
    __shared__ float tile[128];
    unsigned int dst = (unsigned int)__cvta_generic_to_shared(&tile[threadIdx.x]);
    const float* src = g + threadIdx.x;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src));
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
        h[i] = i * 0.5f;
        expected += h[i];
    }
    float *dIn = nullptr, *dOut = nullptr;
    if (musaMalloc((void**)&dIn, N * sizeof(float)) != musaSuccess ||
        musaMalloc((void**)&dOut, sizeof(float)) != musaSuccess) {
        printf("P2-FAIL musaMalloc\n");
        return 2;
    }
    musaMemcpy(dIn, h, N * sizeof(float), musaMemcpyHostToDevice);
    cpasync4_probe_kernel<<<1, N>>>(dIn, dOut);
    musaError_t err = musaDeviceSynchronize();
    if (err != musaSuccess) {
        printf("P2-FAIL launch/sync: %s\n", musaGetErrorString(err));
        return 2;
    }
    float got = -1.0f;
    musaMemcpy(&got, dOut, sizeof(float), musaMemcpyDeviceToHost);
    bool ok = (got - expected) < 1e-3f && (expected - got) < 1e-3f;
    printf("P2 got=%.2f expected=%.2f\n", got, expected);
    printf("P2-%s\n", ok ? "PASS cp.async.ca 4B 全链可用（编译+执行+数据正确）"
                          : "FAIL 数据不一致（指令可编但语义/可见性异常）");
    musaFree(dIn);
    musaFree(dOut);
    return ok ? 0 : 3;
}
