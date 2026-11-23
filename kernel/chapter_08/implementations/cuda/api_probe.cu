// ============================================================================
// api_probe.cu —— 第 08 期传输 API 面：最小程序编译验证 + 逐 API 行为探测。
//
// 用途（CUDA 版；MUSA 版为逐行机械替换版，编译成功本身就是 musa* 命名的
// 实机证据 —— 大纲"musMemcpy/musMallocHost 待实机核对"项的答案载体）：
//   1. 编译验证：引用全部候选 API 符号，编译通过 = 符号存在于头文件/库；
//   2. 行为验证：每个 API 做一次最小真实调用，打印 PASS/FAIL 与返回码；
//   3. 设备能力位：asyncEngineCount（异步引擎对数）、deviceOverlap、
//      canMapHostMemory、unifiedAddressing —— 重叠实验的硬件前提。
//
// 输出：stdout 每行一个 JSON 对象 {"api": "...", "status": "PASS|FAIL|SKIP",
// "detail": "..."}，由 build.sh/run_all.sh 存入 results/server-logs/。
// ============================================================================

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define PROBE(name, call)                                                    \
    do {                                                                     \
        int _e = static_cast<int>(call);                                     \
        std::printf("{\"api\":\"%s\",\"status\":\"%s\",\"code\":%d}\n",      \
                    name, _e == cudaSuccess ? "PASS" : "FAIL", _e);           \
    } while (0)

int main() {
    // flags 必须在上下文创建前设置（mapped 内存前提）；失败如实打印继续
    // （"context 已激活"形态的枚举名跨版本/跨平台不做硬编码，看 code 即可）。
    cudaError_t fe = cudaSetDeviceFlags(cudaDeviceMapHost);
    std::printf("{\"api\":\"cudaSetDeviceFlags(cudaDeviceMapHost)\",\"status\":\"%s\",\"code\":%d}\n",
                fe == cudaSuccess ? "PASS" : "FAIL",
                static_cast<int>(fe));

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::printf("{\"api\":\"cudaGetDeviceProperties\",\"status\":\"FAIL\"}\n");
        return 1;
    }
    std::printf("{\"api\":\"device\",\"status\":\"INFO\",\"detail\":\"%s CC%d.%d SM%d %.1fGB warpSize(CAPI)=%d\"}\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / 1073741824.0, prop.warpSize);
#ifdef T9_DEV_ASYNC_ENGINE
    std::printf("{\"api\":\"prop.asyncEngineCount\",\"status\":\"INFO\",\"value\":%d}\n", prop.asyncEngineCount);
#endif
#ifdef T9_DEV_OVERLAP
    std::printf("{\"api\":\"prop.deviceOverlap\",\"status\":\"INFO\",\"value\":%d}\n", prop.deviceOverlap);
#endif
#ifdef T9_DEV_CAN_MAP
    std::printf("{\"api\":\"prop.canMapHostMemory\",\"status\":\"INFO\",\"value\":%d}\n", prop.canMapHostMemory);
#endif
#ifdef T9_DEV_UNIFIED
    std::printf("{\"api\":\"prop.unifiedAddressing\",\"status\":\"INFO\",\"value\":%d}\n", prop.unifiedAddressing);
#endif

    // ---- 逐 API 行为探测 -------------------------------------------------
    const size_t kBytes = 1 << 20;  // 1MB

    // 设备内存
    void* d = nullptr;
    PROBE("cudaMalloc", cudaMalloc(&d, kBytes));
    PROBE("cudaFree", d ? cudaFree(d) : cudaErrorInvalidValue);

    // pageable + memcpy 同步
    void* hpage = std::malloc(kBytes);
    void* d2 = nullptr;
    cudaMalloc(&d2, kBytes);
    std::memset(hpage, 0xAB, kBytes);
    PROBE("cudaMemcpy(H2D,pageable)", d2 ? cudaMemcpy(d2, hpage, kBytes, cudaMemcpyHostToDevice) : cudaErrorInvalidValue);
    void* hback = std::malloc(kBytes);
    PROBE("cudaMemcpy(D2H,pageable)", d2 ? cudaMemcpy(hback, d2, kBytes, cudaMemcpyDeviceToHost) : cudaErrorInvalidValue);
    std::printf("{\"api\":\"cudaMemcpy.data-integrity\",\"status\":\"%s\"}\n",
                std::memcmp(hpage, hback, kBytes) == 0 ? "PASS" : "FAIL");

    // pinned：cudaMallocHost
    void* hpinned = nullptr;
    PROBE("cudaMallocHost", cudaMallocHost(&hpinned, kBytes));
    if (hpinned) {
        std::memset(hpinned, 0xCD, kBytes);
        PROBE("cudaMemcpy(H2D,pinned)", d2 ? cudaMemcpy(d2, hpinned, kBytes, cudaMemcpyHostToDevice) : cudaErrorInvalidValue);
    }
    PROBE("cudaFreeHost", hpinned ? cudaFreeHost(hpinned) : cudaErrorInvalidValue);

    // pinned：cudaHostAlloc 三种 flags
    void* hdef = nullptr;
    PROBE("cudaHostAlloc(Default)", cudaHostAlloc(&hdef, kBytes, cudaHostAllocDefault));
    if (hdef) cudaFreeHost(hdef);
#ifdef T9_HAVE_HOSTALLOC_PORTABLE
    void* hport = nullptr;
    PROBE("cudaHostAlloc(Portable)", cudaHostAlloc(&hport, kBytes, cudaHostAllocPortable));
    if (hport) cudaFreeHost(hport);
#endif
#ifdef T9_HAVE_HOSTALLOC_MAPPED
    void* hmap = nullptr;
    PROBE("cudaHostAlloc(Mapped)", cudaHostAlloc(&hmap, kBytes, cudaHostAllocMapped));
    if (hmap) {
        void* dmap = nullptr;
        PROBE("cudaHostGetDevicePointer", cudaHostGetDevicePointer(&dmap, hmap, 0));
        cudaFreeHost(hmap);
    }
#endif

    // pageable 原地注册
#ifdef T9_HAVE_HOST_REGISTER
    PROBE("cudaHostRegister", cudaHostRegister(hpage, kBytes, cudaHostRegisterDefault));
    PROBE("cudaHostUnregister", cudaHostUnregister(hpage));
#endif

    // stream + event + async 传输链
    cudaStream_t s = nullptr;
    PROBE("cudaStreamCreate", cudaStreamCreate(&s));
    cudaEvent_t ev = nullptr;
    PROBE("cudaEventCreate", cudaEventCreate(&ev));
    void* hp2 = nullptr;
    cudaMallocHost(&hp2, kBytes);
    std::memset(hp2, 0x5A, kBytes);
    if (s && d2 && hp2) {
        PROBE("cudaMemcpyAsync(H2D,pinned)", cudaMemcpyAsync(d2, hp2, kBytes, cudaMemcpyHostToDevice, s));
        PROBE("cudaEventRecord", cudaEventRecord(ev, s));
        PROBE("cudaStreamSynchronize", cudaStreamSynchronize(s));
        float ms = -1.0f;
        cudaEvent_t ev2 = nullptr;
        cudaEventCreate(&ev2);
        cudaEventRecord(ev2, s);
        cudaEventSynchronize(ev2);
        PROBE("cudaEventElapsedTime", cudaEventElapsedTime(&ms, ev, ev2));
        std::printf("{\"api\":\"cudaEventElapsedTime.value_ms\",\"status\":\"INFO\",\"value\":%.6f}\n", ms);
        cudaEventDestroy(ev2);
    }
#ifdef T9_HAVE_STREAM_NONBLOCKING
    cudaStream_t snb = nullptr;
    PROBE("cudaStreamCreateWithFlags(NonBlocking)", cudaStreamCreateWithFlags(&snb, cudaStreamNonBlocking));
    if (snb) cudaStreamDestroy(snb);
#endif
    PROBE("cudaStreamQuery(idle)", cudaStreamQuery(s));
    PROBE("cudaStreamWaitEvent", cudaStreamWaitEvent(s, ev, 0));
    PROBE("cudaEventDestroy", cudaEventDestroy(ev));
    PROBE("cudaStreamDestroy", cudaStreamDestroy(s));
    if (hp2) cudaFreeHost(hp2);

#ifdef T9_DEV_PCI
    char busid[64] = {0};
    PROBE("cudaDeviceGetPCIBusId", cudaDeviceGetPCIBusId(busid, sizeof(busid), device));
    std::printf("{\"api\":\"pci_bus_id\",\"status\":\"INFO\",\"detail\":\"%s\"}\n", busid);
#endif

    std::free(hpage);
    std::free(hback);
    if (d2) cudaFree(d2);
    std::printf("{\"api\":\"probe\",\"status\":\"DONE\"}\n");
    return 0;
}
