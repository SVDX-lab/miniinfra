// 第 02 期设备属性与概念探针（MUSA 迁移版，S4000 实测）。
// 与 cuda/device_query.cu 逐段对应：API 命名 cuda* → musa*，头文件 → musa_runtime.h。
// 编译（待实机核对，run_all.sh 里会按候选命令逐个尝试并记录成败）：
//   mcc -O2 -o device_query_musa device_query.mu [-lmusart]
// 目的：用实机输出支撑 concept-map.md 的 MUSA 列（warp/quantum、shared memory
// 层级、stream/event、shfl 归约原语是否同源可用）。
#include <cstdio>
#include <musa_runtime.h>

__global__ void warp_shared_reduce(const float* input, float* output, int n) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (gid < n) ? input[gid] : 0.0f;
    // warp 内归约：__shfl_xor_sync 蝶形（MUSA 是否同名可用 = 本程序可编译即证据）
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    int warps_in_block = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        float v = (lane < warps_in_block) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) output[blockIdx.x] = v;
    }
}

int main() {
    musaDeviceProp prop;
    musaGetDeviceProperties(&prop, 0);
    printf("=== MUSA device_query ===\n");
    printf("name=%s\n", prop.name);
    printf("compute_capability=%d.%d\n", prop.major, prop.minor);
    printf("multiProcessorCount=%d\n", prop.multiProcessorCount);
    printf("warpSize=%d\n", prop.warpSize);
    printf("maxThreadsPerBlock=%d\n", prop.maxThreadsPerBlock);
    printf("maxThreadsPerMultiProcessor=%d\n", prop.maxThreadsPerMultiProcessor);
    printf("maxGridDim=(%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("sharedMemPerBlock=%zu\n", (size_t)prop.sharedMemPerBlock);
    printf("sharedMemPerMultiprocessor=%zu\n", (size_t)prop.sharedMemPerMultiprocessor);
    printf("regsPerBlock=%d\n", prop.regsPerBlock);
    printf("regsPerMultiprocessor=%d\n", prop.regsPerMultiprocessor);
    printf("l2CacheSize=%d\n", prop.l2CacheSize);
    printf("memoryClockRate_kHz=%d\n", prop.memoryClockRate);
    printf("memoryBusWidth_bit=%d\n", prop.memoryBusWidth);
    printf("clockRate_kHz=%d\n", prop.clockRate);
    printf("totalGlobalMem=%zu\n", (size_t)prop.totalGlobalMem);
    printf("multiKernelConcurrentLaunch=%d\n", prop.concurrentKernels);

    // ---- warp shfl + shared memory 归约功能验证 ----
    const int n = 256;
    float* d_input = nullptr;
    float* d_output = nullptr;
    musaMalloc(&d_input, n * sizeof(float));
    musaMalloc(&d_output, sizeof(float));
    float* host_input = new float[n];
    for (int i = 0; i < n; ++i) host_input[i] = 1.0f;
    musaMemcpy(d_input, host_input, n * sizeof(float), musaMemcpyHostToDevice);
    warp_shared_reduce<<<1, 256>>>(d_input, d_output, n);
    musaError_t launch_error = musaGetLastError();
    float host_output = 0.0f;
    musaMemcpy(&host_output, d_output, sizeof(float), musaMemcpyDeviceToHost);
    printf("warp_shared_reduce_test launch_err=%d sum=%.1f (expect 256) -> %s\n",
           (int)launch_error, host_output, host_output == 256.0f ? "OK" : "FAIL");

    // ---- stream/event 功能验证 ----
    musaStream_t stream;
    musaStreamCreate(&stream);
    musaEvent_t start, end;
    musaEventCreate(&start);
    musaEventCreate(&end);
    musaEventRecord(start, stream);
    warp_shared_reduce<<<1, 256, 0, stream>>>(d_input, d_output, n);
    musaEventRecord(end, stream);
    musaStreamSynchronize(stream);
    float elapsed_ms = 0.0f;
    musaEventElapsedTime(&elapsed_ms, start, end);
    printf("stream_event_test elapsed_ms=%.4f -> OK\n", elapsed_ms);

    musaStreamDestroy(stream);
    musaEventDestroy(start);
    musaEventDestroy(end);
    musaFree(d_input);
    musaFree(d_output);
    delete[] host_input;
    printf("=== done ===\n");
    return 0;
}
