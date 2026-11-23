// ============================================================================
// transfer_ops.cu —— 第 08 期传输与重叠：extern "C" 传输/流/算子库（CUDA 版）。
//
// 设计（协议 §1 + 任务书"ctypes 直传模式复用"）：
//   - 编译为共享库（libtransfer_ops_cuda.so），Python 经 ctypes 直接传设备/
//     Host 指针；显存分配、同步、计时边界全部留在 Python 侧掌控；
//   - 计算负载 = 第 02 期 RMSNorm kernel（rmsnorm_kernel.cu 逐字节复用，
//     sha256 见 experiment-contract.yaml；M-17 ctypes 直传模式）；
//   - 全部运行时 API 经本库单一出口调用，错误码返回 + 最近错误文本可查
//     （t_last_error），Python 侧 assert 返回 0；
//   - pinned/映射内存与流/事件 API 全部显式暴露 —— 这正是本期要实测的
//     "传输 API 面"（MUSA 迁移版逐 API 对照，差异入迁移台账）。
//
// T9_HAVE_* 编译期开关：MUSA 版由 build.sh 的符号探测结果注入
// （-DT9_HAVE_XXX=1），探测本身就是"该符号在 SDK 头文件中是否存在"的
// 实机证据；CUDA 版全部符号存在，build.sh 全量 -D 注入，两份源码保持
// 逐行同构（对照表按机械替换统计）。
//
// 三级递进的实验落点：
//   (a) 同步 cudaMemcpy 基线      → t_memcpy_sync
//   (b) cudaMemcpyAsync + pinned  → t_memcpy_async + t_malloc_host_pinned_*
//       （cudaMallocHost / cudaHostAlloc[Default/Portable/Mapped] 对比）
//   (c) 分块 + 双 stream 重叠     → t_stream_* / t_event_* + t_rmsnorm_*
//       （流水线编排见 Python bench_overlap.py：传输 stream + 计算 stream，
//        双缓冲 + 事件依赖；本库只提供原语，不做编排）
//
// 计时约定：库内不做计时（事件原语交给 Python），分配/同步的 Host 代价
// 由 Python perf_counter 包裹库调用测得 —— "Host 侧只做计时基准"。
// ============================================================================

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "rmsnorm_kernel.cu"  // 第 02 期 RMSNorm（device 代码 + host launch，零改动）

namespace {

thread_local char g_last_error[512] = {0};

void set_error(const char* what, int err, const char* file, int line) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s at %s:%d err=%d (%s)",
                  what, file, line, err, cudaGetErrorString(static_cast<cudaError_t>(err)));
}

#define T_CHECK(expr)                                                    \
    do {                                                                 \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                         \
            set_error(#expr, static_cast<int>(_e), __FILE__, __LINE__);  \
            return static_cast<int>(_e);                                 \
        }                                                                \
    } while (0)

// 传输方向枚举（与 cudaMemcpyKind 数值对齐，ctypes 层用同一张表）：
// 0=H2D 1=D2H 2=D2D
inline cudaMemcpyKind to_kind(int k) {
    switch (k) {
        case 0: return cudaMemcpyHostToDevice;
        case 1: return cudaMemcpyDeviceToHost;
        default: return cudaMemcpyDeviceToDevice;
    }
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------- 上下文初始化
// 允许 mapped host 内存（必须在上下文创建前设置 flags 才生效；若 torch 已先
// 创建上下文则返回错误码 —— 由调用方记录，属实测信息而非致命错误）。
int t_init(void) {
    // flags 必须在上下文创建前设置；上下文已激活时该调用返回专用错误码——
    // 不做枚举名依赖（CUDA 13/MUSA 侧枚举存在性均不硬编码），任意错误
    // 仅记录不失败：mapped 分配的成败本身就是 flags 是否生效的证据。
    // T9_HAVE_SET_DEVICE_FLAGS/T9_HAVE_DEVICE_MAP_HOST：MUSA 侧符号探测注入，
    // 缺失则跳过 flags 设置并记录（CUDA 版 build.sh 恒定义）。
#if defined(T9_HAVE_SET_DEVICE_FLAGS) && defined(T9_HAVE_DEVICE_MAP_HOST)
    cudaError_t e = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (e != cudaSuccess) {
        set_error("cudaSetDeviceFlags(cudaDeviceMapHost) 非致命", static_cast<int>(e), __FILE__, __LINE__);
    }
#else
    set_error("SetDeviceFlags/DeviceMapHost 符号探测未通过，flags 未设置", 0, __FILE__, __LINE__);
#endif
    // 显式创建/绑定 primary context（后续所有分配归本上下文）
    T_CHECK(cudaFree(nullptr));
    return 0;
}

// ---------------------------------------------------------------- 设备信息
// T9_DEV_* 开关由 build.sh 的 musaDeviceProp 字段探测注入（CUDA 版全量定义）；
// 基础 6 字段（name/major/minor/multiProcessorCount/totalGlobalMem/warpSize）
// 为第 02 期已实机核对的稳定面，不设开关。
int t_dev_info(char* buf, int cap) {
    int device = 0;
    T_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    T_CHECK(cudaGetDeviceProperties(&prop, device));
    int n = 0;
    n += std::snprintf(buf + n, cap - n,
                       "{\"name\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,"
                       "\"total_mem\":%llu,\"warp_size_capi\":%d",
                       prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                       static_cast<unsigned long long>(prop.totalGlobalMem), prop.warpSize);
#ifdef T9_DEV_ASYNC_ENGINE
    n += std::snprintf(buf + n, cap - n, ",\"async_engine_count\":%d", prop.asyncEngineCount);
#endif
#ifdef T9_DEV_OVERLAP
    n += std::snprintf(buf + n, cap - n, ",\"device_overlap\":%d", prop.deviceOverlap);
#endif
#ifdef T9_DEV_CAN_MAP
    n += std::snprintf(buf + n, cap - n, ",\"can_map_host_memory\":%d", prop.canMapHostMemory);
#endif
#ifdef T9_DEV_UNIFIED
    n += std::snprintf(buf + n, cap - n, ",\"unified_addressing\":%d", prop.unifiedAddressing);
#endif
#ifdef T9_DEV_PCI
    n += std::snprintf(buf + n, cap - n, ",\"pci\":\"%04x:%02x:%02x\"",
                       prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
#endif
#ifdef T9_DEV_L2
    n += std::snprintf(buf + n, cap - n, ",\"l2_bytes\":%zu",
                       static_cast<size_t>(prop.l2CacheSize));
#endif
#ifdef T9_DEV_CLOCK
    n += std::snprintf(buf + n, cap - n, ",\"clock_khz\":%d,\"memory_clock_khz\":%d,"
                                         "\"memory_bus_width\":%d",
                       prop.clockRate, prop.memoryClockRate, prop.memoryBusWidth);
#endif
#ifdef T9_DEV_MAXTHREADS
    n += std::snprintf(buf + n, cap - n, ",\"max_threads_per_block\":%d", prop.maxThreadsPerBlock);
#endif
    std::snprintf(buf + n, cap - n, "}");
    return 0;
}

int t_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return -1;
    return n;
}

// ---------------------------------------------------------------- 内存分配
// pageable host（对照基线：stdlib malloc，不可 pin；异步 API 走它时驱动经
// staging buffer 转运 —— 这正是 pageable/pinned 差距的来源）
void* t_malloc_host_pageable(size_t bytes) { return std::malloc(bytes); }

void t_free_host_pageable(void* p) { std::free(p); }

// pinned：legacy API cudaMallocHost（内部 ≈ cudaHostAlloc(Default)）
void* t_malloc_host_pinned_mallochost(size_t bytes) {
    void* p = nullptr;
    if (cudaMallocHost(&p, bytes) != cudaSuccess) return nullptr;
    return p;
}

// pinned：cudaHostAlloc 显式 flags 版（本实验的 API 对比项）
void* t_malloc_host_pinned_hostalloc(size_t bytes, unsigned flags) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, flags) != cudaSuccess) return nullptr;
    return p;
}

// pinned 三种 flags 的包装（flags 由 build.sh 探测注入；CUDA 全部存在）
void* t_malloc_host_pinned_default(size_t bytes) {
    return t_malloc_host_pinned_hostalloc(bytes, cudaHostAllocDefault);
}

#ifdef T9_HAVE_HOSTALLOC_PORTABLE
void* t_malloc_host_pinned_portable(size_t bytes) {
    return t_malloc_host_pinned_hostalloc(bytes, cudaHostAllocPortable);
}
#endif

#ifdef T9_HAVE_HOSTALLOC_MAPPED
// mapped pinned：需要 t_init 的 cudaDeviceMapHost flags 先行；
// 返回 NULL = 平台不支持/flags 未生效（Python 侧记录）
void* t_malloc_host_pinned_mapped(size_t bytes) {
    return t_malloc_host_pinned_hostalloc(bytes, cudaHostAllocMapped);
}

// mapped 内存对应的设备侧别名指针（zero-copy 实验用；失败返回 NULL）
void* t_host_get_device_pointer(void* p) {
    void* dp = nullptr;
    if (cudaHostGetDevicePointer(&dp, p, 0) != cudaSuccess) return nullptr;
    return dp;
}
#endif

int t_free_host_pinned(void* p) {
    cudaError_t e = cudaFreeHost(p);
    if (e != cudaSuccess) set_error("cudaFreeHost", static_cast<int>(e), __FILE__, __LINE__);
    return static_cast<int>(e);
}

#ifdef T9_HAVE_HOST_REGISTER
// 把已分配的 pageable 内存原地注册为 pinned（第三种获得 pinned 的方式）
int t_host_register(void* p, size_t bytes) {
    T_CHECK(cudaHostRegister(p, bytes, cudaHostRegisterDefault));
    return 0;
}

int t_host_unregister(void* p) {
    cudaError_t e = cudaHostUnregister(p);
    if (e != cudaSuccess) set_error("cudaHostUnregister", static_cast<int>(e), __FILE__, __LINE__);
    return static_cast<int>(e);
}
#endif

void* t_malloc_device(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    return p;
}

int t_free_device(void* p) {
    cudaError_t e = cudaFree(p);
    if (e != cudaSuccess) set_error("cudaFree", static_cast<int>(e), __FILE__, __LINE__);
    return static_cast<int>(e);
}

// ---------------------------------------------------------------- 传输
int t_memcpy_sync(void* dst, const void* src, size_t bytes, int kind) {
    T_CHECK(cudaMemcpy(dst, src, bytes, to_kind(kind)));
    return 0;
}

int t_memcpy_async(void* dst, const void* src, size_t bytes, int kind, cudaStream_t stream) {
    T_CHECK(cudaMemcpyAsync(dst, src, bytes, to_kind(kind), stream));
    return 0;
}

// ---------------------------------------------------------------- stream / event
void* t_stream_create(void) {
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) return nullptr;
    return static_cast<void*>(s);
}

#ifdef T9_HAVE_STREAM_NONBLOCKING
void* t_stream_create_nonblocking(void) {
    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return nullptr;
    return static_cast<void*>(s);
}
#endif

int t_stream_destroy(void* s) {
    cudaError_t e = cudaStreamDestroy(static_cast<cudaStream_t>(s));
    if (e != cudaSuccess) set_error("cudaStreamDestroy", static_cast<int>(e), __FILE__, __LINE__);
    return static_cast<int>(e);
}

int t_stream_sync(void* s) {
    cudaError_t e = cudaStreamSynchronize(static_cast<cudaStream_t>(s));
    if (e != cudaSuccess) set_error("cudaStreamSynchronize", static_cast<int>(e), __FILE__, __LINE__);
    return static_cast<int>(e);
}

int t_stream_query(void* s) {  // 0=完成 1=未完成 负数=错误 -1000=平台缺符号（非阻塞探测）
    cudaError_t e = cudaStreamQuery(static_cast<cudaStream_t>(s));
    if (e == cudaSuccess) return 0;
#ifdef T9_HAVE_ERROR_NOT_READY
    if (e == cudaErrorNotReady) return 1;
    return -static_cast<int>(e);
#else
    // musaErrorNotReady 缺失（探测未通过）：无法区分未完成与错误，保守按未完成
    return 1;
#endif
}

void* t_event_create(void) {
    cudaEvent_t e = nullptr;
    if (cudaEventCreate(&e) != cudaSuccess) return nullptr;
    return static_cast<void*>(e);
}

int t_event_destroy(void* ev) {
    cudaError_t e = cudaEventDestroy(static_cast<cudaEvent_t>(ev));
    if (e != cudaSuccess) set_error("cudaEventDestroy", static_cast<int>(e), __FILE__, __LINE__);
    return static_cast<int>(e);
}

int t_event_record(void* ev, cudaStream_t stream) {
    T_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(ev), stream));
    return 0;
}

int t_stream_wait_event(void* stream, void* ev) {
    T_CHECK(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(ev), 0));
    return 0;
}

int t_event_elapsed(void* a, void* b, float* ms) {
    T_CHECK(cudaEventElapsedTime(&ms[0], static_cast<cudaEvent_t>(a), static_cast<cudaEvent_t>(b)));
    return 0;
}

int t_device_sync(void) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) set_error("cudaDeviceSynchronize", static_cast<int>(e), __FILE__, __LINE__);
    return static_cast<int>(e);
}

// ------------------------------------------- 计算负载（第 02 期 RMSNorm，零改动复用）
// x,y 为设备指针（或 mapped host 内存的设备别名），w 为设备常驻权重。
int t_rmsnorm(void* x, void* w, void* y, long long rows, long long h, double eps,
              long long tile, int is_fp16, cudaStream_t stream) {
    try {
        mini_infra::launch_rmsnorm_any(x, w, y, is_fp16 != 0, static_cast<int>(rows),
                                       static_cast<int>(h), static_cast<float>(eps),
                                       static_cast<int>(tile), stream);
    } catch (const std::exception& ex) {
        std::snprintf(g_last_error, sizeof(g_last_error), "rmsnorm: %s", ex.what());
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------- 诊断
const char* t_last_error(void) { return g_last_error; }

// 清除运行时 sticky last-error（musaGetLastError/cudaGetLastError 语义：读出即清零）。
// 用途：测试段之间隔离陈旧错误（例如无效句柄探测失败后，后续 kernel 的
// GetLastError 检查会捡到陈旧错误码 —— S4000 会话实测发现，见 KNOWN_ISSUES）。
int t_clear_last_error(void) {
    cudaGetLastError();
    g_last_error[0] = '\0';
    return 0;
}

// 编译期自报告：本库实际编译进来的 API 符号名（MUSA 迁移对照的第一手证据）
const char* t_api_names(void) {
    return "cudaMemcpy,cudaMemcpyAsync,cudaMallocHost,cudaHostAlloc,cudaFreeHost,"
           "cudaHostAllocDefault,cudaHostAllocPortable,cudaHostAllocMapped,"
           "cudaHostRegister,cudaHostUnregister,cudaHostGetDevicePointer,"
           "cudaSetDeviceFlags,cudaDeviceMapHost,cudaStreamCreate,"
           "cudaStreamCreateWithFlags,cudaStreamNonBlocking,cudaStreamDestroy,"
           "cudaStreamSynchronize,cudaStreamQuery,cudaStreamWaitEvent,"
           "cudaEventCreate,cudaEventDestroy,cudaEventRecord,cudaEventElapsedTime,"
           "cudaDeviceSynchronize,cudaMalloc,cudaFree,cudaGetDeviceProperties,"
           "cudaDeviceGetPCIBusId";
}

#ifdef T9_HAVE_PCI_BUS_ID
int t_pci_bus_id(char* buf, int cap) {
    int device = 0;
    T_CHECK(cudaGetDevice(&device));
    T_CHECK(cudaDeviceGetPCIBusId(buf, cap, device));
    return 0;
}
#else
// musaDeviceGetPCIBusId 缺失（探测未通过）——拓扑探测退到 sysfs/lspci，边界记录
int t_pci_bus_id(char* buf, int cap) {
    (void)cap;
    set_error("DeviceGetPCIBusId 符号探测未通过（平台边界）", 0, __FILE__, __LINE__);
    return -99;
}
#endif

}  // extern "C"
