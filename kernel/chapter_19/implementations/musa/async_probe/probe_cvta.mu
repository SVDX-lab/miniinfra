// P1 探针：__cvta_generic_to_shared 在 mcc/mp_22 是否可用（cp.async 寻址的前置）
// 判据：编过 + 跑出合理 shared 窗口地址（非零、远小于 generic 地址空间）
#include <cstdio>
#include <musa_runtime.h>

__global__ void cvta_probe_kernel(unsigned int* out) {
    __shared__ float x[32];
    unsigned int addr = (unsigned int)__cvta_generic_to_shared(&x[16]);
    if (threadIdx.x == 0) out[0] = addr;
    if (threadIdx.x == 16) out[1] = (unsigned int)__cvta_generic_to_shared(&x[0]);
}

int main() {
    unsigned int* dOut = nullptr;
    if (musaMalloc((void**)&dOut, 2 * sizeof(unsigned int)) != musaSuccess) {
        printf("P1-FAIL musaMalloc\n");
        return 2;
    }
    cvta_probe_kernel<<<1, 32>>>(dOut);
    musaError_t err = musaDeviceSynchronize();
    if (err != musaSuccess) {
        printf("P1-FAIL launch/sync: %s\n", musaGetErrorString(err));
        return 2;
    }
    unsigned int h[2] = {0, 0};
    musaMemcpy(h, dOut, 2 * sizeof(unsigned int), musaMemcpyDeviceToHost);
    // x[0] 与 x[16] 的 shared 偏移应差 64B（16 floats）
    printf("P1 addr(&x[0])=%u addr(&x[16])=%u delta=%u\n", h[1], h[0], h[0] - h[1]);
    bool ok = (h[0] != 0) && (h[0] != h[1]) && ((h[0] - h[1]) == 64u);
    printf("P1-%s\n", ok ? "PASS __cvta_generic_to_shared 可用" : "FAIL 地址换算异常");
    musaFree(dOut);
    return ok ? 0 : 3;
}
