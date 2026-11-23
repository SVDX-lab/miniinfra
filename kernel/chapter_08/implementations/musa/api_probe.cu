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

#include <musa_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define PROBE(name, call)                                                    \
    do {                                                                     \
        int _e = static_cast<int>(call);                                     \
        std::printf("{\"api\":\"%s\",\"status\":\"%s\",\"code\":%d}\n",      \
                    name, _e == musaSuccess ? "PASS" : "FAIL", _e);           \
    } while (0)

int main() {
    // flags 必须在上下文创建前设置（mapped 内存前提）；失败如实打印继续
    // （"context 已激活"形态的枚举名跨版本/跨平台不做硬编码，看 code 即可）。
    musaError_t fe = musaSetDeviceFlags(musaDeviceMapHost);
    std::printf("{\"api\":\"musaSetDeviceFlags(musaDeviceMapHost)\",\"status\":\"%s\",\"code\":%d}\n",
                fe == musaSuccess ? "PASS" : "FAIL",
                static_cast<int>(fe));

    int device = 0;
    musaGetDevice(&device);
    musaDeviceProp prop;
    if (musaGetDeviceProperties(&prop, device) != musaSuccess) {
        std::printf("{\"api\":\"musaGetDeviceProperties\",\"status\":\"FAIL\"}\n");
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
    PROBE("musaMalloc", musaMalloc(&d, kBytes));
    PROBE("musaFree", d ? musaFree(d) : musaErrorInvalidValue);

    // pageable + memcpy 同步
    void* hpage = std::malloc(kBytes);
    void* d2 = nullptr;
    musaMalloc(&d2, kBytes);
    std::memset(hpage, 0xAB, kBytes);
    PROBE("musaMemcpy(H2D,pageable)", d2 ? musaMemcpy(d2, hpage, kBytes, musaMemcpyHostToDevice) : musaErrorInvalidValue);
    void* hback = std::malloc(kBytes);
    PROBE("musaMemcpy(D2H,pageable)", d2 ? musaMemcpy(hback, d2, kBytes, musaMemcpyDeviceToHost) : musaErrorInvalidValue);
    std::printf("{\"api\":\"musaMemcpy.data-integrity\",\"status\":\"%s\"}\n",
                std::memcmp(hpage, hback, kBytes) == 0 ? "PASS" : "FAIL");

    // pinned：musaMallocHost
    void* hpinned = nullptr;
    PROBE("musaMallocHost", musaMallocHost(&hpinned, kBytes));
    if (hpinned) {
        std::memset(hpinned, 0xCD, kBytes);
        PROBE("musaMemcpy(H2D,pinned)", d2 ? musaMemcpy(d2, hpinned, kBytes, musaMemcpyHostToDevice) : musaErrorInvalidValue);
    }
    PROBE("musaFreeHost", hpinned ? musaFreeHost(hpinned) : musaErrorInvalidValue);

    // pinned：musaHostAlloc 三种 flags
    void* hdef = nullptr;
    PROBE("musaHostAlloc(Default)", musaHostAlloc(&hdef, kBytes, musaHostAllocDefault));
    if (hdef) musaFreeHost(hdef);
#ifdef T9_HAVE_HOSTALLOC_PORTABLE
    void* hport = nullptr;
    PROBE("musaHostAlloc(Portable)", musaHostAlloc(&hport, kBytes, musaHostAllocPortable));
    if (hport) musaFreeHost(hport);
#endif
#ifdef T9_HAVE_HOSTALLOC_MAPPED
    void* hmap = nullptr;
    PROBE("musaHostAlloc(Mapped)", musaHostAlloc(&hmap, kBytes, musaHostAllocMapped));
    if (hmap) {
        void* dmap = nullptr;
        PROBE("musaHostGetDevicePointer", musaHostGetDevicePointer(&dmap, hmap, 0));
        musaFreeHost(hmap);
    }
#endif

    // pageable 原地注册
#ifdef T9_HAVE_HOST_REGISTER
    PROBE("musaHostRegister", musaHostRegister(hpage, kBytes, musaHostRegisterDefault));
    PROBE("musaHostUnregister", musaHostUnregister(hpage));
#endif

    // stream + event + async 传输链
    musaStream_t s = nullptr;
    PROBE("musaStreamCreate", musaStreamCreate(&s));
    musaEvent_t ev = nullptr;
    PROBE("musaEventCreate", musaEventCreate(&ev));
    void* hp2 = nullptr;
    musaMallocHost(&hp2, kBytes);
    std::memset(hp2, 0x5A, kBytes);
    if (s && d2 && hp2) {
        PROBE("musaMemcpyAsync(H2D,pinned)", musaMemcpyAsync(d2, hp2, kBytes, musaMemcpyHostToDevice, s));
        PROBE("musaEventRecord", musaEventRecord(ev, s));
        PROBE("musaStreamSynchronize", musaStreamSynchronize(s));
        float ms = -1.0f;
        musaEvent_t ev2 = nullptr;
        musaEventCreate(&ev2);
        musaEventRecord(ev2, s);
        musaEventSynchronize(ev2);
        PROBE("musaEventElapsedTime", musaEventElapsedTime(&ms, ev, ev2));
        std::printf("{\"api\":\"musaEventElapsedTime.value_ms\",\"status\":\"INFO\",\"value\":%.6f}\n", ms);
        musaEventDestroy(ev2);
    }
#ifdef T9_HAVE_STREAM_NONBLOCKING
    musaStream_t snb = nullptr;
    PROBE("musaStreamCreateWithFlags(NonBlocking)", musaStreamCreateWithFlags(&snb, musaStreamNonBlocking));
    if (snb) musaStreamDestroy(snb);
#endif
    PROBE("musaStreamQuery(idle)", musaStreamQuery(s));
    PROBE("musaStreamWaitEvent", musaStreamWaitEvent(s, ev, 0));
    PROBE("musaEventDestroy", musaEventDestroy(ev));
    PROBE("musaStreamDestroy", musaStreamDestroy(s));
    if (hp2) musaFreeHost(hp2);

#ifdef T9_DEV_PCI
    char busid[64] = {0};
    PROBE("musaDeviceGetPCIBusId", musaDeviceGetPCIBusId(busid, sizeof(busid), device));
    std::printf("{\"api\":\"pci_bus_id\",\"status\":\"INFO\",\"detail\":\"%s\"}\n", busid);
#endif

    std::free(hpage);
    std::free(hback);
    if (d2) musaFree(d2);
    std::printf("{\"api\":\"probe\",\"status\":\"DONE\"}\n");
    return 0;
}
