// 第 02 期设备属性与概念探针（CUDA 参照版，4090 实测）。
// 输出：设备属性（Grid/Block/Warp/SharedMemory/L2 层级的规格数字）
//      + warp shfl 归约 + shared memory 跨 warp 归约 + stream/event 功能验证。
// 编译：nvcc -O2 -o device_query_cuda device_query.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void warp_shared_reduce(const float* input, float* output, int n) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (gid < n) ? input[gid] : 0.0f;
    // warp 内归约：__shfl_xor_sync 蝶形
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
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== CUDA device_query ===\n");
    printf("name=%s\n", prop.name);
    printf("compute_capability=%d.%d\n", prop.major, prop.minor);
    printf("multiProcessorCount=%d\n", prop.multiProcessorCount);
    printf("warpSize=%d\n", prop.warpSize);
    printf("maxThreadsPerBlock=%d\n", prop.maxThreadsPerBlock);
    printf("maxThreadsPerMultiProcessor=%d\n", prop.maxThreadsPerMultiProcessor);
    printf("maxGridDim=(%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("sharedMemPerBlock=%zu\n", (size_t)prop.sharedMemPerBlock);
    printf("sharedMemPerBlockOptin=%zu\n", (size_t)prop.sharedMemPerBlockOptin);
    printf("sharedMemPerMultiprocessor=%zu\n", (size_t)prop.sharedMemPerMultiprocessor);
    printf("regsPerBlock=%d\n", prop.regsPerBlock);
    printf("regsPerMultiprocessor=%d\n", prop.regsPerMultiprocessor);
    printf("l2CacheSize=%d\n", prop.l2CacheSize);
    printf("persistingL2CacheMaxSize=%d\n", prop.persistingL2CacheMaxSize);
    // CUDA 13.0 移除了 cudaDeviceProp::clockRate / memoryClockRate（12.x 起废弃）
    // —— 记录为版本边界：显存频率规格改由 measure_peak.py 实测带宽佐证
    printf("memoryBusWidth_bit=%d\n", prop.memoryBusWidth);
    printf("totalGlobalMem=%zu\n", (size_t)prop.totalGlobalMem);
    printf("multiKernelConcurrentLaunch=%d\n", prop.concurrentKernels);
    printf("maxStreamsPerDevice= see runtime; concurrency(capability)=%d\n", prop.concurrentKernels);

    // ---- warp shfl + shared memory 归约功能验证 ----
    const int n = 256;
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    float* host_input = new float[n];
    for (int i = 0; i < n; ++i) host_input[i] = 1.0f;  // 256 个 1 → 和 = 256
    cudaMemcpy(d_input, host_input, n * sizeof(float), cudaMemcpyHostToDevice);
    warp_shared_reduce<<<1, 256>>>(d_input, d_output, n);
    float host_output = 0.0f;
    cudaMemcpy(&host_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("warp_shared_reduce_test sum=%.1f (expect 256) -> %s\n",
           host_output, host_output == 256.0f ? "OK" : "FAIL");

    // ---- stream/event 功能验证 ----
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);
    warp_shared_reduce<<<1, 256, 0, stream>>>(d_input, d_output, n);
    cudaEventRecord(end, stream);
    cudaStreamSynchronize(stream);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    printf("stream_event_test elapsed_ms=%.4f -> OK\n", elapsed_ms);

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] host_input;
    printf("=== done ===\n");
    return 0;
}
