// ============================================================================
// stub_main_emu.cpp —— 本地 CPU 仿真器（无 nvcc/mcc 时对第 22 期 GPU 列的
// 语法 + 数值双重预验证）。
//
// 三层本地检查（上服务器前全部通过）：
//   1. rope_kernels.cuh 全模板实例化编译检查（stub_syntax_check.cpp）；
//   2. 本文件：rope_bench_main.cuh 全量编译（含 JSON/apply/host 真值路径）；
//   3. 本文件执行：kernel 主体在 CPU 上逐 block/thread 串行执行（元素级核
//      无 __syncthreads/__shfl，逐线程重放语义精确），gpuMalloc/Memcpy 映射
//      到 malloc/memcpy，__half 用 _Float16 位精确——selftest 走真实核代码
//      vs host FP64 真值，与服务器 selftest 同判据。
//
// 运行：g++ -std=c++17 -DRPE_HOST_EMU -I../cuda -o /tmp/rope_emu stub_main_emu.cpp
//       && /tmp/rope_emu selftest && /tmp/rope_emu apply-smoke
// ============================================================================

#define RPE_HOST_EMU 1
#define RPE_NO_FP16_INCLUDE 1

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

// ---- CUDA/MUSA 限定符与 builtin stub ----
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__
#define __restrict__

namespace rpe_emu {
thread_local unsigned emu_block = 0;
thread_local unsigned emu_thread = 0;
}

#define blockIdx rpe_emu::emu_block_idx()
#define threadIdx rpe_emu::emu_thread_idx()
#define blockDim rpe_emu::emu_block_dim()
namespace rpe_emu {
struct Dim3 { unsigned x, y, z; };
inline Dim3& emu_block_idx() { static thread_local Dim3 v{0, 0, 0}; return v; }
inline Dim3& emu_thread_idx() { static thread_local Dim3 v{0, 0, 0}; return v; }
inline Dim3 emu_block_dim() { return Dim3{1024, 1, 1}; }   // cast 核 blockDim=1024
}
// device builtin warpSize 的 host 常量（不用宏——run_attrs 里 prop.warpSize
// 是成员访问，宏展开会破坏语法）；g++ 下全局常量与 kernel 体同名解析正确
constexpr unsigned warpSize = 32;

// sincosf（glibc 在严格 C 下不导出 C++ 名，自备包装）
inline void sincosf_emu(float x, float* s, float* c) { *s = sinf(x); *c = cosf(x); }
#define sincosf(x, s, c) sincosf_emu((x), (s), (c))

// ---- __half stand-in：_Float16 位精确（IEEE binary16 舍入与 GPU 一致）----
struct HalfStub { _Float16 v; };
inline float __half2float(HalfStub h) { return static_cast<float>(h.v); }
inline HalfStub __float2half(float f) { return HalfStub{static_cast<_Float16>(f)}; }
#define RPE_HALF_TYPE HalfStub

// ---- runtime API stub：malloc/memcpy 直通 ----
enum gpuError_t { gpuSuccess = 0, gpuErrorGeneric = 1 };
enum gpuMemcpyKind { gpuMemcpyHostToDevice = 1, gpuMemcpyDeviceToHost = 2 };
inline const char* gpuGetErrorString(gpuError_t) { return "emu"; }
struct gpuDeviceProp_t {
    char name[64] = "CPU-EMU";
    int major = 0, minor = 0;
    int multiProcessorCount = 8;
    size_t totalGlobalMem = 1ull << 32;
    int warpSize = 32;               // C API 口径（device builtin 由宏另算）
    size_t sharedMemPerBlock = 28672;
    int regsPerBlock = 65536;
};
inline gpuError_t gpuGetDeviceProperties(gpuDeviceProp_t* p, int) { *p = gpuDeviceProp_t(); return gpuSuccess; }
inline gpuError_t gpuMalloc(void** out, size_t bytes) {
    *out = calloc(bytes, 1);          // 零初始化（EMU 可复现）
    return *out ? gpuSuccess : gpuErrorGeneric;
}
inline gpuError_t gpuFree(void* p) { free(p); return gpuSuccess; }
inline gpuError_t gpuMemcpy(void* dst, const void* src, size_t bytes, gpuMemcpyKind) {
    memcpy(dst, src, bytes);
    return gpuSuccess;
}
struct gpuEventStub {};
using gpuEvent_t = gpuEventStub*;
inline gpuError_t gpuEventCreate(gpuEvent_t* e) { *e = new gpuEventStub(); return gpuSuccess; }
inline gpuError_t gpuEventDestroy(gpuEvent_t e) { delete e; return gpuSuccess; }
inline gpuError_t gpuEventRecord(gpuEvent_t) { return gpuSuccess; }
inline gpuError_t gpuEventSynchronize(gpuEvent_t) { return gpuSuccess; }
inline gpuError_t gpuEventElapsedTime(float* ms, gpuEvent_t, gpuEvent_t) { *ms = 0.0f; return gpuSuccess; }
inline gpuError_t gpuDeviceSynchronize() { return gpuSuccess; }
inline gpuError_t gpuGetLastError() { return gpuSuccess; }
#define GPU_CHECK(expr)                                                        \
    do {                                                                       \
        gpuError_t err__ = (expr);                                             \
        if (err__ != gpuSuccess) {                                             \
            fprintf(stderr, "[emu-gpu-error] %s:%d\n", __FILE__, __LINE__);    \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

// ---- EMU 发射：逐 block × thread 串行重放（元素级核语义精确）----
namespace rpe_emu {
template <typename K, typename... A>
void go(K kernel, unsigned grid, unsigned block, A... args) {
    for (unsigned b = 0; b < grid; ++b) {
        emu_block_idx() = Dim3{b, 0, 0};
        for (unsigned t = 0; t < block; ++t) {
            emu_thread_idx() = Dim3{t, 0, 0};
            kernel(args...);
        }
    }
}
}  // namespace rpe_emu

// ---- 主入口体重命名（头内 main 让位给本文件 driver）----
#define main rpe_bench_original_main
#include "rope_bench_main.cuh"
#undef main

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "selftest";
    if (mode == "selftest") {
        return rpe23::run_selftest(4);
    }
    if (mode == "smoke") {
        // correctness 小矩阵抽查（seq ≤ 512 两 dtype 两形态 + 确定性），
        // 全链走真实核代码 vs host FP64 真值。JSON 落 /tmp（本地工作产物）。
        rpe23::g_seq_cap = 512;
        const int rc1 = rpe23::run_correctness(4, "/tmp/emu-correctness.json");
        printf("[emu] correctness-small rc=%d\n", rc1);
        return rc1;
    }
    if (mode == "cachedtype") {
        // cache dtype 误差放大矩阵在 EMU 上的数值验证（表误差 + e2e）；
        // EMU 串行执行，大 seq 慢——仅验证数值结构，正式数字以 GPU 为准。
        return rpe23::run_cachedtype(4, "/tmp/emu-cache-dtype.json");
    }
    if (mode == "apply") {
        if (argc < 3) {
            fprintf(stderr, "用法: %s apply <desc.json>\n", argv[0]);
            return 1;
        }
        return rpe23::run_apply(argv[2], "/tmp/emu-apply-summary.json");
    }
    fprintf(stderr, "用法: %s {selftest|smoke}\n", argv[0]);
    return 1;
}
