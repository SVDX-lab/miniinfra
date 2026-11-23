// ============================================================================
// rope_bench_main.cuh —— 第 23 期 standalone 基准入口体（CUDA/MUSA 共用）。
//
// 平台差异只在本文件之前的 shim 块（rope_bench.cu / rope_bench.mu 各自定义
// gpu* 宏）——"机械替换单点化"迁移设计（ch18/ch21 模式）。无 torch 依赖：
// 自带 hash-RNG 数据生成（rope_config.py 同算法，JSON 记录校验和）、
// host FP64 逐元素真值（std::thread 并行）、correctness 矩阵、cache-dtype
// 误差放大矩阵、benchmark 矩阵、apply 文件桥（真实权重单层验证用）、事件对
// 计时、JSON 输出。
//
// 用法：
//   ./rope_bench attrs          # 设备属性 + warpSize 探针 → *-env.json
//   ./rope_bench selftest       # 最小形状快速自检（先于全矩阵，失败即停）
//   ./rope_bench correctness    # 正确性矩阵 → *-correctness.json
//   ./rope_bench cachedtype     # cache dtype 误差放大 → *-cache-dtype.json
//   ./rope_bench bench          # 基准矩阵 → *-benchmark.json
//   ./rope_bench apply <desc>   # 文件桥：按 desc.json 跑单次核 → 输出 bin
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>

#include "rope_kernels.cuh"

// ---- GPU_CHECK（定义一次；入口/EMU stub 若已定义同 macro 则以其为准）----
#ifndef GPU_CHECK
#define GPU_CHECK(expr)                                                      \
    do {                                                                     \
        gpuError_t err__ = (expr);                                           \
        if (err__ != gpuSuccess) {                                           \
            fprintf(stderr, "[gpu-error] %s:%d %s: %s\n", __FILE__,          \
                    __LINE__, #expr, gpuGetErrorString(err__));              \
            exit(2);                                                         \
        }                                                                    \
    } while (0)
#endif

// ---- 发射封装：nvcc/mcc 走真实 <<<>>> 路径；RPE_HOST_EMU（本地 CPU 仿真，
//      g++ 全量编译 + kernel 主体逐 block/thread 执行——语法与数值双重预验证）；
//      RPE_HOST_STUB（仅语法检查，发射为空操作）----
#if defined(RPE_HOST_EMU)
namespace rpe_emu {
// go() 在 stub 文件定义：设置 thread_local emu_block/emu_thread 后逐线程执行
template <typename K, typename... A>
void go(K kernel, unsigned grid, unsigned block, A... args);
}
#define LAUNCH(kern, grid, block, ...) rpe_emu::go(kern, grid, block, ##__VA_ARGS__)
#elif defined(RPE_HOST_STUB)
namespace rpe_stub {
inline void go(const char* name, int grid, int block) {
    (void)name; (void)grid; (void)block;
}
}
#define LAUNCH(kern, grid, block, ...) rpe_stub::go(#kern, grid, block)
#else
#define LAUNCH(kern, grid, block, ...) kern<<<grid, block>>>(__VA_ARGS__)
#endif

namespace rpe23 {

using rpe::HD;
using rpe::HALF;
using rpe::THREADS;
// RPE_HALF_TYPE 不 using：EMU 模式下它是全局 HalfStub（在 rpe 内展开为
// rpe::HalfStub 会失败）；rpe23 内未限定名查找能到达全局作用域。

constexpr int HQ = 16;   // Qwen3-0.6B（rope_config.py 一致）
constexpr int HKV = 8;
constexpr float THETA = 1000000.0f;   // config.json 实读（rope_theta=1e6）
constexpr uint32_t SEED_Q = 0x23002301u;
constexpr uint32_t SEED_K = 0x23002302u;

int g_warp = 0;   // 设备内置 warpSize（探针）
int g_seq_cap = 1 << 30;   // correctness seq 上限（EMU 本地冒烟用；GPU 运行不动）

// ---------------------------------------------------------------------------
// hash RNG（rope_config.mu_hash 同算法：uint32 溢出语义；numpy 侧可复算校验和）
// ---------------------------------------------------------------------------

inline uint32_t mu_hash(uint32_t idx, uint32_t seed) {
    uint32_t x = idx * 0x9E3779B9u ^ seed;
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}

inline float mu_value(size_t idx, uint32_t seed) {
    return static_cast<float>(mu_hash(static_cast<uint32_t>(idx), seed) % 20001u) / 10000.0f - 1.0f;
}

double checksum(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += x;
    return s;
}

// ---------------------------------------------------------------------------
// host 真值：cache 表（fp64 锚点 / fp32 镜像）与逐元素 apply（可 thread 并行）
// ---------------------------------------------------------------------------

void host_build_tables_f64(int max_seq, double theta,
                           std::vector<double>& cos_t, std::vector<double>& sin_t) {
    cos_t.assign(static_cast<size_t>(max_seq) * HALF, 0.0);
    sin_t.assign(static_cast<size_t>(max_seq) * HALF, 0.0);
    for (int j = 0; j < HALF; ++j) {
        const double exponent = (2.0 * static_cast<double>(j)) / static_cast<double>(HD);
        const double inv_freq = 1.0 / std::pow(theta, exponent);
        for (int pos = 0; pos < max_seq; ++pos) {
            const double angle = static_cast<double>(pos) * inv_freq;
            cos_t[static_cast<size_t>(pos) * HALF + j] = std::cos(angle);
            sin_t[static_cast<size_t>(pos) * HALF + j] = std::sin(angle);
        }
    }
}

// fp32 镜像表（host libm powf/cosf/sinf——与 device powf/sincosf 的 ulp 差
// 本身是实验测量对象，不假设一致）
void host_build_tables_f32(int max_seq, float theta,
                           std::vector<float>& cos_t, std::vector<float>& sin_t) {
    cos_t.assign(static_cast<size_t>(max_seq) * HALF, 0.0f);
    sin_t.assign(static_cast<size_t>(max_seq) * HALF, 0.0f);
    for (int j = 0; j < HALF; ++j) {
        const float exponent = (2.0f * static_cast<float>(j)) / static_cast<float>(HD);
        const float inv_freq = 1.0f / powf(theta, exponent);
        for (int pos = 0; pos < max_seq; ++pos) {
            const float angle = static_cast<float>(pos) * inv_freq;
            cos_t[static_cast<size_t>(pos) * HALF + j] = cosf(angle);
            sin_t[static_cast<size_t>(pos) * HALF + j] = sinf(angle);
        }
    }
}

// 通用 host apply（CT ∈ {float, double}）：q/k fp32 io（正确性模式语义），
// 内部提升到 CT 计算。按 (t, h) 分块给 nthreads 个线程。
template <typename CT>
void host_apply(const std::vector<float>& q, const std::vector<float>& k,
                std::vector<float>& qo, std::vector<float>& ko,
                int num_tokens, int hq, int hkv, int seq_offset,
                const std::vector<CT>& cos_t, const std::vector<CT>& sin_t,
                int nthreads) {
    qo.assign(q.size(), 0.0f);
    ko.assign(k.size(), 0.0f);
    const int heads = hq + hkv;
    auto worker = [&](int tid) {
        for (int th = tid; th < heads; th += nthreads) {
            const bool is_q = th < hq;
            const int h = is_q ? th : th - hq;
            const size_t rows = is_q ? static_cast<size_t>(num_tokens) * hq * HD
                                     : static_cast<size_t>(num_tokens) * hkv * HD;
            const std::vector<float>& src = is_q ? q : k;
            std::vector<float>& dst = is_q ? qo : ko;
            for (int t = 0; t < num_tokens; ++t) {
                const size_t row_cache = static_cast<size_t>(seq_offset + t) * HALF;
                const size_t row_src = (static_cast<size_t>(t) * (is_q ? hq : hkv) + h) * HD;
                for (int j = 0; j < HALF; ++j) {
                    const CT c = cos_t[row_cache + j];
                    const CT s = sin_t[row_cache + j];
                    const CT x1 = static_cast<CT>(src[row_src + j]);
                    const CT x2 = static_cast<CT>(src[row_src + j + HALF]);
                    dst[row_src + j] = static_cast<float>(x1 * c - x2 * s);
                    dst[row_src + j + HALF] = static_cast<float>(x2 * c + x1 * s);
                }
            }
            (void)rows;
        }
    };
    std::vector<std::thread> ths;
    for (int i = 0; i < nthreads; ++i) ths.emplace_back(worker, i);
    for (auto& th : ths) th.join();
}

// 单 Token host apply（decode 真值）
template <typename CT>
void host_apply_decode(const std::vector<float>& q, const std::vector<float>& k,
                       std::vector<float>& qo, std::vector<float>& ko,
                       int hq, int hkv, int seq_offset,
                       const std::vector<CT>& cos_t, const std::vector<CT>& sin_t) {
    std::vector<float> qq = q, kk = k;
    host_apply<CT>(qq, kk, qo, ko, 1, hq, hkv, seq_offset, cos_t, sin_t, 1);
}

// ---------------------------------------------------------------------------
// GPU 缓冲与发射
// ---------------------------------------------------------------------------

struct DeviceBuf {
    void* ptr = nullptr;
    size_t bytes = 0;
    void alloc(size_t nbytes) {
        bytes = nbytes;
        GPU_CHECK(gpuMalloc(&ptr, nbytes));
    }
    void free() {
        if (ptr) GPU_CHECK(gpuFree(ptr));
        ptr = nullptr;
    }
};

struct GpuData {
    DeviceBuf q32, k32;                 // fp32 master（host 上传）
    DeviceBuf q16, k16;                 // fp16 io 缓冲
    DeviceBuf qo32, ko32;               // fp32 输出
    DeviceBuf qo16, ko16;               // fp16 输出
    DeviceBuf cache32c, cache32s;       // fp32 cache
    DeviceBuf cache16c, cache16s;       // fp16 cache
    size_t q_count = 0, k_count = 0;
    size_t cache_count = 0;             // max_seq * HALF

    void alloc_io(int num_tokens, int hq, int hkv, int max_seq, bool with_f16_cache) {
        q_count = static_cast<size_t>(num_tokens) * hq * HD;
        k_count = static_cast<size_t>(num_tokens) * hkv * HD;
        q32.alloc(q_count * 4); k32.alloc(k_count * 4);
        q16.alloc(q_count * 2); k16.alloc(k_count * 2);
        qo32.alloc(q_count * 4); ko32.alloc(k_count * 4);
        qo16.alloc(q_count * 2); ko16.alloc(k_count * 2);
        cache_count = static_cast<size_t>(max_seq) * HALF;
        cache32c.alloc(cache_count * 4); cache32s.alloc(cache_count * 4);
        if (with_f16_cache) {
            cache16c.alloc(cache_count * 2); cache16s.alloc(cache_count * 2);
        }
    }
    void build_caches(int max_seq, float theta, bool with_f16_cache) {
        const int blocks32 = static_cast<int>((cache_count + THREADS - 1) / THREADS);
        LAUNCH(rpe::build_cache_kernel<float>, blocks32, THREADS,
               reinterpret_cast<float*>(cache32c.ptr), reinterpret_cast<float*>(cache32s.ptr),
               max_seq, theta);
        GPU_CHECK(gpuGetLastError());
        if (with_f16_cache) {
            LAUNCH(rpe::build_cache_kernel<RPE_HALF_TYPE>, blocks32, THREADS,
                   reinterpret_cast<RPE_HALF_TYPE*>(cache16c.ptr),
                   reinterpret_cast<RPE_HALF_TYPE*>(cache16s.ptr), max_seq, theta);
            GPU_CHECK(gpuGetLastError());
        }
        GPU_CHECK(gpuDeviceSynchronize());
    }
    void upload(const std::vector<float>& q, const std::vector<float>& k) {
        GPU_CHECK(gpuMemcpy(q32.ptr, q.data(), q_count * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(k32.ptr, k.data(), k_count * 4, gpuMemcpyHostToDevice));
    }
    void cast_io_to_f16() {
        auto cast = [](DeviceBuf& s32, DeviceBuf& s16, size_t count) {
            const int blocks = static_cast<int>((count + 1023) / 1024);
            LAUNCH(rpe::cast_f32_to_f16_kernel, blocks, 1024,
                   reinterpret_cast<const float*>(s32.ptr),
                   reinterpret_cast<RPE_HALF_TYPE*>(s16.ptr), count);
        };
        cast(q32, q16, q_count);
        cast(k32, k16, k_count);
        GPU_CHECK(gpuGetLastError());
        GPU_CHECK(gpuDeviceSynchronize());
    }
    void free_all() {
        q32.free(); k32.free(); q16.free(); k16.free();
        qo32.free(); ko32.free(); qo16.free(); ko16.free();
        cache32c.free(); cache32s.free(); cache16c.free(); cache16s.free();
    }
};

template <typename T, typename CT>
void launch_prefill_t(const GpuData& g, int num_tokens, int hq, int hkv,
                      int seq_offset, bool io_half, bool cache_half,
                      bool sync_after = true) {
    const void* qs = io_half ? static_cast<const void*>(g.q16.ptr) : static_cast<const void*>(g.q32.ptr);
    const void* ks = io_half ? static_cast<const void*>(g.k16.ptr) : static_cast<const void*>(g.k32.ptr);
    void* qo = io_half ? static_cast<void*>(g.qo16.ptr) : static_cast<void*>(g.qo32.ptr);
    void* ko = io_half ? static_cast<void*>(g.ko16.ptr) : static_cast<void*>(g.ko32.ptr);
    const void* cc = cache_half ? static_cast<const void*>(g.cache16c.ptr) : static_cast<const void*>(g.cache32c.ptr);
    const void* cs = cache_half ? static_cast<const void*>(g.cache16s.ptr) : static_cast<const void*>(g.cache32s.ptr);
    const size_t total = static_cast<size_t>(num_tokens) * (hq + hkv) * HALF;
    const int blocks = static_cast<int>((total + THREADS - 1) / THREADS);
    if (io_half) {
        LAUNCH((rpe::rope_prefill_kernel<RPE_HALF_TYPE, CT>), blocks, THREADS,
               reinterpret_cast<const RPE_HALF_TYPE*>(qs), reinterpret_cast<const RPE_HALF_TYPE*>(ks),
               reinterpret_cast<RPE_HALF_TYPE*>(qo), reinterpret_cast<RPE_HALF_TYPE*>(ko),
               reinterpret_cast<const CT*>(cc), reinterpret_cast<const CT*>(cs),
               num_tokens, hq, hkv, seq_offset);
    } else {
        LAUNCH((rpe::rope_prefill_kernel<float, CT>), blocks, THREADS,
               reinterpret_cast<const float*>(qs), reinterpret_cast<const float*>(ks),
               reinterpret_cast<float*>(qo), reinterpret_cast<float*>(ko),
               reinterpret_cast<const CT*>(cc), reinterpret_cast<const CT*>(cs),
               num_tokens, hq, hkv, seq_offset);
    }
    GPU_CHECK(gpuGetLastError());
    if (sync_after) GPU_CHECK(gpuDeviceSynchronize());
}

void launch_prefill(const GpuData& g, int num_tokens, int hq, int hkv,
                    int seq_offset, bool io_half, bool cache_half,
                    bool sync_after = true) {
    if (io_half && cache_half)       launch_prefill_t<RPE_HALF_TYPE, RPE_HALF_TYPE>(g, num_tokens, hq, hkv, seq_offset, true, true, sync_after);
    else if (io_half)                launch_prefill_t<RPE_HALF_TYPE, float>(g, num_tokens, hq, hkv, seq_offset, true, false, sync_after);
    else if (cache_half)             launch_prefill_t<float, RPE_HALF_TYPE>(g, num_tokens, hq, hkv, seq_offset, false, true, sync_after);
    else                             launch_prefill_t<float, float>(g, num_tokens, hq, hkv, seq_offset, false, false, sync_after);
}

template <typename T, typename CT>
void launch_decode_t(const GpuData& g, int hq, int hkv, int seq_offset,
                     bool io_half, bool cache_half, bool sync_after = true) {
    const void* qs = io_half ? static_cast<const void*>(g.q16.ptr) : static_cast<const void*>(g.q32.ptr);
    const void* ks = io_half ? static_cast<const void*>(g.k16.ptr) : static_cast<const void*>(g.k32.ptr);
    void* qo = io_half ? static_cast<void*>(g.qo16.ptr) : static_cast<void*>(g.qo32.ptr);
    void* ko = io_half ? static_cast<void*>(g.ko16.ptr) : static_cast<void*>(g.ko32.ptr);
    const void* cc = cache_half ? static_cast<const void*>(g.cache16c.ptr) : static_cast<const void*>(g.cache32c.ptr);
    const void* cs = cache_half ? static_cast<const void*>(g.cache16s.ptr) : static_cast<const void*>(g.cache32s.ptr);
    if (io_half) {
        LAUNCH((rpe::rope_decode_kernel<RPE_HALF_TYPE, CT>), static_cast<unsigned>(hq + hkv),
               rpe::DECODE_THREADS,
               reinterpret_cast<const RPE_HALF_TYPE*>(qs), reinterpret_cast<const RPE_HALF_TYPE*>(ks),
               reinterpret_cast<RPE_HALF_TYPE*>(qo), reinterpret_cast<RPE_HALF_TYPE*>(ko),
               reinterpret_cast<const CT*>(cc), reinterpret_cast<const CT*>(cs),
               hq, hkv, seq_offset);
    } else {
        LAUNCH((rpe::rope_decode_kernel<float, CT>), static_cast<unsigned>(hq + hkv),
               rpe::DECODE_THREADS,
               reinterpret_cast<const float*>(qs), reinterpret_cast<const float*>(ks),
               reinterpret_cast<float*>(qo), reinterpret_cast<float*>(ko),
               reinterpret_cast<const CT*>(cc), reinterpret_cast<const CT*>(cs),
               hq, hkv, seq_offset);
    }
    GPU_CHECK(gpuGetLastError());
    if (sync_after) GPU_CHECK(gpuDeviceSynchronize());
}

void launch_decode(const GpuData& g, int hq, int hkv, int seq_offset,
                   bool io_half, bool cache_half, bool sync_after = true) {
    if (io_half && cache_half)       launch_decode_t<RPE_HALF_TYPE, RPE_HALF_TYPE>(g, hq, hkv, seq_offset, true, true, sync_after);
    else if (io_half)                launch_decode_t<RPE_HALF_TYPE, float>(g, hq, hkv, seq_offset, true, false, sync_after);
    else if (cache_half)             launch_decode_t<float, RPE_HALF_TYPE>(g, hq, hkv, seq_offset, false, true, sync_after);
    else                             launch_decode_t<float, float>(g, hq, hkv, seq_offset, false, false, sync_after);
}

// 输出回读（fp16 输出经 cast kernel 回 fp32 再下载）
void download_output(GpuData& g, bool io_half, std::vector<float>& qo, std::vector<float>& ko) {
    qo.resize(g.q_count);
    ko.resize(g.k_count);
    if (io_half) {
        auto cast_back = [](DeviceBuf& s16, DeviceBuf& d32, size_t count) {
            const int blocks = static_cast<int>((count + 1023) / 1024);
            LAUNCH(rpe::cast_f16_to_f32_kernel, blocks, 1024,
                   reinterpret_cast<const RPE_HALF_TYPE*>(s16.ptr),
                   reinterpret_cast<float*>(d32.ptr), count);
        };
        cast_back(g.qo16, g.qo32, g.q_count);
        cast_back(g.ko16, g.ko32, g.k_count);
        GPU_CHECK(gpuGetLastError());
        GPU_CHECK(gpuDeviceSynchronize());
    }
    GPU_CHECK(gpuMemcpy(qo.data(), g.qo32.ptr, g.q_count * 4, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(ko.data(), g.ko32.ptr, g.k_count * 4, gpuMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// host 数据生成
// ---------------------------------------------------------------------------

struct HostData {
    std::vector<float> q, k;
    void gen(int num_tokens) {
        const size_t nq = static_cast<size_t>(num_tokens) * HQ * HD;
        const size_t nk = static_cast<size_t>(num_tokens) * HKV * HD;
        q.resize(nq);
        k.resize(nk);
        for (size_t i = 0; i < nq; ++i) q[i] = mu_value(i, SEED_Q);
        for (size_t i = 0; i < nk; ++i) k[i] = mu_value(i, SEED_K);
    }
};

// ---------------------------------------------------------------------------
// 误差统计（线程并行分段累加）
// ---------------------------------------------------------------------------

struct ErrStats {
    double max_abs = 0.0, mean_abs = 0.0;
    double dot_aa = 0.0, dot_bb = 0.0, dot_ab = 0.0;
    int nan_count = 0;
    double max_abs_pos = 0;
    void accumulate(const std::vector<float>& out, const std::vector<double>& ref,
                    int nthreads) {
        const size_t n = ref.size();
        struct Part { double max_abs = 0, sum = 0, aa = 0, bb = 0, ab = 0, pos = 0; int nan = 0; };
        std::vector<Part> parts(nthreads);
        auto worker = [&](int tid) {
            Part& p = parts[tid];
            for (size_t i = tid; i < n; i += static_cast<size_t>(nthreads)) {
                const double o = static_cast<double>(out[i]);
                const double d = std::fabs(o - ref[i]);
                if (std::isnan(o)) ++p.nan;
                if (d > p.max_abs) { p.max_abs = d; p.pos = static_cast<double>(i); }
                p.sum += d;
                p.aa += o * o;
                p.bb += ref[i] * ref[i];
                p.ab += o * ref[i];
            }
        };
        std::vector<std::thread> ths;
        for (int i = 0; i < nthreads; ++i) ths.emplace_back(worker, i);
        for (auto& th : ths) th.join();
        double sum = 0;
        for (auto& p : parts) {
            sum += p.sum;
            if (p.max_abs > max_abs) { max_abs = p.max_abs; max_abs_pos = p.pos; }
            dot_aa += p.aa; dot_bb += p.bb; dot_ab += p.ab;
            nan_count += p.nan;
        }
        mean_abs = sum / static_cast<double>(n);
    }
    double cosine() const { return dot_ab / (std::sqrt(dot_aa) * std::sqrt(dot_bb) + 1e-30); }
};

// ---------------------------------------------------------------------------
// attrs
// ---------------------------------------------------------------------------

int run_attrs(const char* env_name) {
    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, 0));
    DeviceBuf buf;
    buf.alloc(4);
    LAUNCH(rpe::warp_size_probe, 1, 32, reinterpret_cast<int*>(buf.ptr));
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    int builtin_warp = 0;
    GPU_CHECK(gpuMemcpy(&builtin_warp, buf.ptr, 4, gpuMemcpyDeviceToHost));
    buf.free();
    g_warp = builtin_warp;

    FILE* f = fopen(env_name, "w");
    fprintf(f, "{\n");
    fprintf(f, "  \"schema_version\": \"kernel-chapter23.env.gpu.v1\",\n");
    fprintf(f, "  \"device_name\": \"%s\",\n", prop.name);
    fprintf(f, "  \"cc\": \"%d.%d\",\n", prop.major, prop.minor);
    fprintf(f, "  \"multi_processor_count\": %d,\n", prop.multiProcessorCount);
    fprintf(f, "  \"total_memory_bytes\": %zu,\n", static_cast<size_t>(prop.totalGlobalMem));
    fprintf(f, "  \"warp_size_c_api\": %d,\n", prop.warpSize);
    fprintf(f, "  \"warp_size_device_builtin_probe\": %d,\n", builtin_warp);
    fprintf(f, "  \"shared_mem_per_block\": %zu,\n", static_cast<size_t>(prop.sharedMemPerBlock));
    fprintf(f, "  \"regs_per_block\": %d,\n", prop.regsPerBlock);
    fprintf(f, "  \"rope_constants\": {\"head_dim\": %d, \"theta\": %.1f, \"hq\": %d, \"hkv\": %d}\n",
            HD, THETA, HQ, HKV);
    fprintf(f, "}\n");
    fclose(f);
    printf("device=%s CC=%d.%d SMs=%d warpC-API=%d warpBuiltin=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.warpSize, builtin_warp);
    return 0;
}

// ---------------------------------------------------------------------------
// correctness 矩阵：seq {1, 64, 512, 2048, 32768} × {fp32, fp16} × 两形态
// 真值：host FP64 表 + FP64 apply（io fp32）；解析式容差见 run 内注释
// ---------------------------------------------------------------------------

struct CaseResult {
    std::string form, dtype;
    int seq, seq_offset;
    double max_abs, mean_abs, cosine;
    int nan;
    double tol;
    bool pass;
    double checksum_q, checksum_k;
};

double tol_fp32_case(double max_pos) {
    // fp32 角度舍入放大界：angle err ≤ max_pos·2^-24，输出两合成 ≤ 3×(|x|max=1)
    // 基础算术舍入 ≤ 5e-4。留 2× 余量。
    return 5e-4 + 6.0 * max_pos * 5.9604645e-8;
}

int run_correctness(int nthreads, const char* json_name) {
    FILE* f = fopen(json_name, "w");
    fprintf(f, "{\n  \"schema_version\": \"kernel-chapter23.correctness.gpu.v1\",\n");
    fprintf(f, "  \"truth\": \"host FP64 tables (HF formula 1/theta^(2j/D)) + FP64 elementwise apply; hash-RNG seed 0x23002301/02 (rope_config.py 同算法)\",\n");
    fprintf(f, "  \"warp_device_builtin\": %d,\n", g_warp);
    fprintf(f, "  \"theta\": %.1f,\n", THETA);
    fprintf(f, "  \"cases\": [\n");
    int failures = 0;
    int written = 0;

    const int max_seq = 40000;
    std::vector<double> cos64, sin64;
    host_build_tables_f64(max_seq, THETA, cos64, sin64);

    auto emit = [&](const char* form, int seq, const char* dtype, bool io_half) {
        const int seq_offset = (strcmp(form, "decode") == 0) ? seq : 0;
        const int num_tokens = (strcmp(form, "decode") == 0) ? 1 : seq;
        const int max_pos = seq_offset + num_tokens;   // 触达的最大位置
        HostData host;
        host.gen(num_tokens);
        const double cs_q = checksum(host.q);
        const double cs_k = checksum(host.k);

        GpuData g;
        g.alloc_io(num_tokens, HQ, HKV, max_seq, false);
        g.upload(host.q, host.k);
        g.cast_io_to_f16();
        g.build_caches(max_seq, THETA, false);
        if (strcmp(form, "decode") == 0) {
            launch_decode(g, HQ, HKV, seq_offset, io_half, false);
        } else {
            launch_prefill(g, num_tokens, HQ, HKV, seq_offset, io_half, false);
        }
        std::vector<float> qo, ko;
        download_output(g, io_half, qo, ko);
        g.free_all();

        // host FP64 真值
        std::vector<float> tq, tk;
        if (strcmp(form, "decode") == 0) {
            host_apply_decode<double>(host.q, host.k, tq, tk, HQ, HKV, seq_offset, cos64, sin64);
        } else {
            host_apply<double>(host.q, host.k, tq, tk, num_tokens, HQ, HKV, seq_offset, cos64, sin64, nthreads);
        }
        std::vector<double> ref;
        ref.reserve(tq.size() + tk.size());
        for (float x : tq) ref.push_back(static_cast<double>(x));
        for (float x : tk) ref.push_back(static_cast<double>(x));
        std::vector<float> got;
        got.reserve(ref.size());
        for (float x : qo) got.push_back(x);
        for (float x : ko) got.push_back(x);

        ErrStats st;
        st.accumulate(got, ref, nthreads);
        const double tol = io_half ? (2e-3 + tol_fp32_case(max_pos) - 5e-4)  // fp16 输出舍入 ≤2e-3 + 角度项
                                   : tol_fp32_case(max_pos);
        const bool pass = (st.max_abs <= tol) && st.nan_count == 0;
        if (!pass) ++failures;

        fprintf(f, "%s    {\"form\": \"%s\", \"seq\": %d, \"seq_offset\": %d, "
                   "\"dtype\": \"%s\", \"head_dim\": %d, "
                   "\"max_abs\": %.3e, \"mean_abs\": %.3e, \"cosine\": %.8f, "
                   "\"nan\": %d, \"tolerance\": %.3e, \"pass\": %s, "
                   "\"checksum_q\": %.6f, \"checksum_k\": %.6f}",
                written ? ",\n" : "", form, seq, seq_offset, dtype, HD,
                st.max_abs, st.mean_abs, st.cosine(), st.nan_count, tol,
                pass ? "true" : "false", cs_q, cs_k);
        ++written;
        printf("[correctness/%s] seq=%d pos=%d dtype=%s: max_abs=%.3e cos=%.7f %s\n",
               form, seq, seq_offset, dtype, st.max_abs, st.cosine(), pass ? "PASS" : "FAIL");
        fflush(f);
        fflush(stdout);
    };

    for (int di = 0; di < 2; ++di) {
        const char* dtype = di == 0 ? "fp16" : "fp32";
        const bool io_half = di == 0;
        for (int s : {1, 64, 512, 2048, 32768}) if (s <= g_seq_cap) emit("prefill", s, dtype, io_half);
        for (int s : {1, 64, 512, 2048, 32768}) if (s <= g_seq_cap) emit("decode", s, dtype, io_half);
    }

    // 确定性：prefill/decode fp16 各 3 次位级比对
    auto determinism = [&](const char* form, int seq, bool io_half) {
        const int seq_offset = (strcmp(form, "decode") == 0) ? seq : 0;
        const int num_tokens = (strcmp(form, "decode") == 0) ? 1 : seq;
        HostData host;
        host.gen(num_tokens);
        std::vector<std::vector<float>> runs;
        for (int r = 0; r < 3; ++r) {
            GpuData g;
            g.alloc_io(num_tokens, HQ, HKV, max_seq, false);
            g.upload(host.q, host.k);
            g.cast_io_to_f16();
            g.build_caches(max_seq, THETA, false);
            if (strcmp(form, "decode") == 0) launch_decode(g, HQ, HKV, seq_offset, io_half, false);
            else launch_prefill(g, num_tokens, HQ, HKV, seq_offset, io_half, false);
            std::vector<float> qo, ko;
            download_output(g, io_half, qo, ko);
            g.free_all();
            qo.insert(qo.end(), ko.begin(), ko.end());
            runs.push_back(std::move(qo));
        }
        const bool det = (runs[0] == runs[1]) && (runs[1] == runs[2]);
        fprintf(f, ",\n    {\"form\": \"%s-determinism\", \"seq\": %d, \"dtype\": \"%s\", "
                   "\"runs\": 3, \"bitwise_identical\": %s}",
                form, seq, io_half ? "fp16" : "fp32", det ? "true" : "false");
        printf("[determinism/%s] seq=%d: %s\n", form, seq, det ? "True" : "False");
    };
    determinism("prefill", 2048, true);
    determinism("decode", 2047, true);

    fprintf(f, "\n  ],\n  \"failures\": %d\n}\n", failures);
    fclose(f);
    return failures;
}

// ---------------------------------------------------------------------------
// cachedtype：cache 表 fp32 vs fp16 误差放大（theta × seq 矩阵）
// ---------------------------------------------------------------------------

int run_cachedtype(int nthreads, const char* json_name) {
    FILE* f = fopen(json_name, "w");
    fprintf(f, "{\n  \"schema_version\": \"kernel-chapter23.cachedtype.gpu.v1\",\n");
    fprintf(f, "  \"warp_device_builtin\": %d,\n", g_warp);
    fprintf(f, "  \"table_error\": [\n");
    int written = 0;

    // 1) 表误差矩阵：theta {1e4, 1e6, 1e8} × seq {512, 2048, 8192, 32768}
    //    × dtype {fp32, fp16}；逐 j 误差剖面（top-5 高误差 j）
    for (double theta_d : {10000.0, 1000000.0, 100000000.0}) {
        const float theta_f = static_cast<float>(theta_d);
        for (int seq : {512, 2048, 8192, 32768}) {
            std::vector<double> cos64, sin64;
            host_build_tables_f64(seq, theta_d, cos64, sin64);
            // device 双表
            GpuData g;
            const int seq_pad = seq;  // cache 独立分配
            g.cache_count = static_cast<size_t>(seq) * HALF;
            g.cache32c.alloc(g.cache_count * 4); g.cache32s.alloc(g.cache_count * 4);
            g.cache16c.alloc(g.cache_count * 2); g.cache16s.alloc(g.cache_count * 2);
            const int blocks = static_cast<int>((g.cache_count + THREADS - 1) / THREADS);
            LAUNCH(rpe::build_cache_kernel<float>, blocks, THREADS,
                   reinterpret_cast<float*>(g.cache32c.ptr), reinterpret_cast<float*>(g.cache32s.ptr),
                   seq, theta_f);
            LAUNCH(rpe::build_cache_kernel<RPE_HALF_TYPE>, blocks, THREADS,
                   reinterpret_cast<RPE_HALF_TYPE*>(g.cache16c.ptr),
                   reinterpret_cast<RPE_HALF_TYPE*>(g.cache16s.ptr), seq, theta_f);
            GPU_CHECK(gpuGetLastError());
            GPU_CHECK(gpuDeviceSynchronize());
            std::vector<float> c32(g.cache_count), s32(g.cache_count);
            std::vector<RPE_HALF_TYPE> c16(g.cache_count), s16(g.cache_count);
            GPU_CHECK(gpuMemcpy(c32.data(), g.cache32c.ptr, g.cache_count * 4, gpuMemcpyDeviceToHost));
            GPU_CHECK(gpuMemcpy(s32.data(), g.cache32s.ptr, g.cache_count * 4, gpuMemcpyDeviceToHost));
            GPU_CHECK(gpuMemcpy(c16.data(), g.cache16c.ptr, g.cache_count * 2, gpuMemcpyDeviceToHost));
            GPU_CHECK(gpuMemcpy(s16.data(), g.cache16s.ptr, g.cache_count * 2, gpuMemcpyDeviceToHost));
            g.cache32c.free(); g.cache32s.free(); g.cache16c.free(); g.cache16s.free();

            for (int di = 0; di < 2; ++di) {
                const char* dtype = di == 0 ? "fp16" : "fp32";
                // 误差 + 逐 j 剖面
                double max_c = 0, max_s = 0, sum = 0;
                std::vector<double> jerr(HALF, 0.0);
                for (size_t i = 0; i < g.cache_count; ++i) {
                    const double cv = di == 0 ? static_cast<double>(__half2float(c16[i]))
                                              : static_cast<double>(c32[i]);
                    const double sv = di == 0 ? static_cast<double>(__half2float(s16[i]))
                                              : static_cast<double>(s32[i]);
                    const double ec = std::fabs(cv - cos64[i]);
                    const double es = std::fabs(sv - sin64[i]);
                    max_c = std::fmax(max_c, ec);
                    max_s = std::fmax(max_s, es);
                    sum += ec + es;
                    const int j = static_cast<int>(i % HALF);
                    jerr[j] = std::fmax(jerr[j], std::fmax(ec, es));
                }
                // top-3 误差 j
                std::vector<int> order(HALF);
                for (int j = 0; j < HALF; ++j) order[j] = j;
                std::stable_sort(order.begin(), order.end(),
                                 [&](int a, int b) { return jerr[a] > jerr[b]; });
                fprintf(f, "%s    {\"theta\": %.1e, \"seq\": %d, \"dtype\": \"%s\", "
                           "\"cos_max\": %.3e, \"sin_max\": %.3e, \"combined_mean\": %.3e, "
                           "\"worst_j\": [%d, %d, %d], \"worst_j_err\": [%.3e, %.3e, %.3e]}",
                        written ? ",\n" : "", theta_d, seq, dtype,
                        max_c, max_s, sum / static_cast<double>(2 * g.cache_count),
                        order[0], order[1], order[2], jerr[order[0]], jerr[order[1]], jerr[order[2]]);
                ++written;
                printf("[table/%.0e/%d/%s] cos_max=%.3e sin_max=%.3e worst_j=%d(%.2e)\n",
                       theta_d, seq, dtype, max_c, max_s, order[0], jerr[order[0]]);
            }
        }
    }
    fprintf(f, "\n  ],\n");

    // 2) 端到端 q 误差（theta=1e6 Qwen3 口径）：seq × cache dtype，io fp32
    fprintf(f, "  \"e2e\": [\n");
    int ewritten = 0;
    {
        const double theta_d = 1000000.0;
        std::vector<double> cos64, sin64;
        for (int seq : {512, 2048, 8192, 32768}) {
            host_build_tables_f64(seq, theta_d, cos64, sin64);
            HostData host;
            host.gen(seq);
            std::vector<float> tq, tk;
            host_apply<double>(host.q, host.k, tq, tk, seq, HQ, HKV, 0, cos64, sin64, nthreads);
            std::vector<double> ref;
            ref.reserve(tq.size());
            for (float x : tq) ref.push_back(static_cast<double>(x));

            GpuData g;
            g.alloc_io(seq, HQ, HKV, seq, true);
            g.upload(host.q, host.k);
            g.cast_io_to_f16();
            g.build_caches(seq, static_cast<float>(theta_d), true);
            std::vector<double> all_jerr_fp32(HALF, 0.0);
            for (int ci = 0; ci < 2; ++ci) {
                const bool cache_half = ci == 0;
                launch_prefill(g, seq, HQ, HKV, 0, false, cache_half);
                std::vector<float> qo, ko;
                download_output(g, false, qo, ko);
                ErrStats st;
                st.accumulate(qo, ref, nthreads);
                fprintf(f, "%s    {\"theta\": %.1e, \"seq\": %d, \"cache_dtype\": \"%s\", "
                           "\"io_dtype\": \"fp32\", \"max_abs\": %.3e, \"mean_abs\": %.3e, "
                           "\"cosine\": %.8f}",
                        ewritten ? ",\n" : "", theta_d, seq, cache_half ? "fp16" : "fp32",
                        st.max_abs, st.mean_abs, st.cosine());
                ++ewritten;
                printf("[e2e/%d/%s] max_abs=%.3e mean=%.3e\n",
                       seq, cache_half ? "fp16" : "fp32", st.max_abs, st.mean_abs);
            }
            g.free_all();
        }
    }
    fprintf(f, "\n  ],\n  \"failures\": 0\n}\n");
    fclose(f);
    return 0;
}

// ---------------------------------------------------------------------------
// benchmark 矩阵
// ---------------------------------------------------------------------------

struct Timing { double p50_ms, p95_ms, mean_ms, min_ms; int warmup, repeats; };

Timing time_it(int warmup, int repeats, const std::function<void()>& op) {
    for (int i = 0; i < warmup; ++i) op();
    GPU_CHECK(gpuDeviceSynchronize());
    std::vector<gpuEvent_t> ev(repeats * 2);
    for (int i = 0; i < repeats * 2; ++i) GPU_CHECK(gpuEventCreate(&ev[i]));
    for (int i = 0; i < repeats; ++i) {
        GPU_CHECK(gpuEventRecord(ev[2 * i]));
        op();
        GPU_CHECK(gpuEventRecord(ev[2 * i + 1]));
    }
    GPU_CHECK(gpuDeviceSynchronize());
    std::vector<double> ms(repeats);
    double sum = 0;
    for (int i = 0; i < repeats; ++i) {
        float el = 0.0f;
        GPU_CHECK(gpuEventElapsedTime(&el, ev[2 * i], ev[2 * i + 1]));
        ms[i] = static_cast<double>(el);
        sum += ms[i];
    }
    for (int i = 0; i < repeats * 2; ++i) GPU_CHECK(gpuEventDestroy(ev[i]));
    std::sort(ms.begin(), ms.end());
    return Timing{ms[repeats / 2],
                  ms[static_cast<size_t>(std::fmax(0.0, std::round(repeats * 0.95) - 1))],
                  sum / repeats, ms[0], warmup, repeats};
}

// back-to-back 发射吞吐（M-12 差分法）：enqueue_only = CPU 发射 N 次的 wall/N；
// wall = 发射 + 全部执行完成的 wall/N。
void launch_rate(int warmup, int count, const std::function<void()>& op,
                 double* enqueue_us, double* wall_us) {
    for (int i = 0; i < warmup; ++i) op();
    GPU_CHECK(gpuDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) op();
    auto t1 = std::chrono::steady_clock::now();
    GPU_CHECK(gpuDeviceSynchronize());
    auto t2 = std::chrono::steady_clock::now();
    *enqueue_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / count;
    *wall_us = std::chrono::duration<double, std::micro>(t2 - t0).count() / count;
}

int run_bench(const char* json_name) {
    FILE* f = fopen(json_name, "w");
    fprintf(f, "{\n  \"schema_version\": \"kernel-chapter23.benchmark.gpu.v1\",\n");
    fprintf(f, "  \"timing\": \"event-pair per repeat, sync once at end; decode 另测 launch-rate 差分\",\n");
    fprintf(f, "  \"theta\": %.1f,\n", THETA);
    fprintf(f, "  \"warp_device_builtin\": %d,\n", g_warp);
    fprintf(f, "  \"entries\": [\n");
    int written = 0;

    auto io_bytes = [&](int num_tokens, bool io_half, bool cache_half) {
        const size_t item = io_half ? 2 : 4;
        const size_t citem = cache_half ? 2 : 4;
        const size_t qkv = 2ull * (static_cast<size_t>(num_tokens) * HQ * HD
                                   + static_cast<size_t>(num_tokens) * HKV * HD) * item;
        const size_t cache = 2ull * static_cast<size_t>(num_tokens) * HALF * citem;  // unique 口径
        const size_t cache_issued = cache * (HQ + HKV);
        return std::make_pair(qkv + cache, qkv + cache_issued);
    };

    auto emit_prefill = [&](int tokens, bool io_half, bool cache_half, int warmup, int repeats) {
        const int max_seq = tokens + 1;
        HostData host;
        host.gen(tokens);
        GpuData g;
        g.alloc_io(tokens, HQ, HKV, max_seq, cache_half);
        g.upload(host.q, host.k);
        g.cast_io_to_f16();
        g.build_caches(max_seq, THETA, cache_half);
        Timing t = time_it(warmup, repeats, [&] {
            launch_prefill(g, tokens, HQ, HKV, 0, io_half, cache_half);
        });
        auto bs = io_bytes(tokens, io_half, cache_half);
        fprintf(f, "%s    {\"form\": \"prefill\", \"tokens\": %d, \"io_dtype\": \"%s\", "
                   "\"cache_dtype\": \"%s\", \"p50_ms\": %.4f, \"p95_ms\": %.4f, "
                   "\"mean_ms\": %.4f, \"min_ms\": %.4f, \"warmup\": %d, \"repeats\": %d, "
                   "\"bytes_unique\": %zu, \"bytes_issued\": %zu, "
                   "\"gb_s_unique\": %.2f, \"gb_s_issued\": %.2f, \"flops\": %zu, \"ai_flop_per_byte\": %.4f}",
                written ? ",\n" : "", tokens, io_half ? "fp16" : "fp32",
                cache_half ? "fp16" : "fp32",
                t.p50_ms, t.p95_ms, t.mean_ms, t.min_ms, t.warmup, t.repeats,
                bs.first, bs.second,
                bs.first / (t.p50_ms * 1e-3) / 1e9, bs.second / (t.p50_ms * 1e-3) / 1e9,
                static_cast<size_t>(6ull * static_cast<size_t>(tokens) * (HQ + HKV) * HALF),
                static_cast<double>(6ull * static_cast<size_t>(tokens) * (HQ + HKV) * HALF) / static_cast<double>(bs.first));
        ++written;
        printf("[bench/prefill] T=%d io=%s cache=%s: p50=%.4f ms (%.1f GB/s unique)\n",
               tokens, io_half ? "fp16" : "fp32", cache_half ? "fp16" : "fp32",
               t.p50_ms, bs.first / (t.p50_ms * 1e-3) / 1e9);
        fflush(f);
        g.free_all();
    };

    auto emit_decode = [&](int pos, bool io_half, bool cache_half, bool with_launchrate,
                           int warmup, int repeats) {
        const int max_seq = pos + 1;
        HostData host;
        host.gen(1);
        GpuData g;
        g.alloc_io(1, HQ, HKV, max_seq, cache_half);
        g.upload(host.q, host.k);
        g.cast_io_to_f16();
        g.build_caches(max_seq, THETA, cache_half);
        Timing t = time_it(warmup, repeats, [&] {
            launch_decode(g, HQ, HKV, pos, io_half, cache_half);
        });
        auto bs = io_bytes(1, io_half, cache_half);
        double enqueue_us = -1, wall_us = -1;
        if (with_launchrate) {
            // 无同步发射：enqueue_us = 纯 CPU 发射率（M-12 差分）；wall = 发射 + 全部完成
            launch_rate(10, 1000, [&] { launch_decode(g, HQ, HKV, pos, io_half, cache_half, false); },
                        &enqueue_us, &wall_us);
        }
        fprintf(f, "%s    {\"form\": \"decode\", \"pos\": %d, \"io_dtype\": \"%s\", "
                   "\"cache_dtype\": \"%s\", \"p50_ms\": %.4f, \"p95_ms\": %.4f, "
                   "\"mean_ms\": %.4f, \"min_ms\": %.4f, \"warmup\": %d, \"repeats\": %d, "
                   "\"bytes_unique\": %zu, \"bytes_issued\": %zu, "
                   "\"gb_s_unique\": %.3f, \"enqueue_us\": %.3f, \"wall_us_backtoback\": %.3f, "
                   "\"flops\": %zu}",
                written ? ",\n" : "", pos, io_half ? "fp16" : "fp32",
                cache_half ? "fp16" : "fp32",
                t.p50_ms, t.p95_ms, t.mean_ms, t.min_ms, t.warmup, t.repeats,
                bs.first, bs.second,
                bs.first / (t.p50_ms * 1e-3) / 1e9, enqueue_us, wall_us,
                static_cast<size_t>(6ull * (HQ + HKV) * HALF));
        ++written;
        printf("[bench/decode] pos=%d io=%s cache=%s: p50=%.4f ms enqueue=%.2fus wall=%.2fus\n",
               pos, io_half ? "fp16" : "fp32", cache_half ? "fp16" : "fp32",
               t.p50_ms, enqueue_us, wall_us);
        fflush(f);
        g.free_all();
    };

    // prefill 扫描
    for (int di = 0; di < 2; ++di) {
        const bool io_half = di == 0;
        for (int tokens : {128, 2048, 8192, 32768}) {
            const int r = tokens <= 2048 ? 30 : (tokens <= 8192 ? 15 : 8);
            const int w = tokens <= 2048 ? 3 : 2;
            emit_prefill(tokens, io_half, false, w, r);
        }
    }
    // cache fp16 对照（io fp16）：prefill 两档
    emit_prefill(2048, true, true, 3, 30);
    emit_prefill(32768, true, true, 2, 8);

    // decode 扫描（位置平坦性 + launch-rate 差分）
    for (int di = 0; di < 2; ++di) {
        const bool io_half = di == 0;
        for (int pos : {0, 2048, 32767}) {
            emit_decode(pos, io_half, false, true, 10, 100);
        }
    }
    // cache fp16 对照
    emit_decode(2048, true, true, false, 10, 100);
    emit_decode(32767, false, true, false, 10, 100);

    // cache 生成吞吐（一次性成本）
    for (int di = 0; di < 2; ++di) {
        const bool cache_half = di == 0;
        for (int max_seq : {2048, 32768, 131072}) {
            const int warmup = max_seq <= 32768 ? 3 : 2;
            const int repeats = max_seq <= 32768 ? 30 : 10;
            DeviceBuf bc, bs;
            const size_t count = static_cast<size_t>(max_seq) * HALF;
            bc.alloc(count * (cache_half ? 2 : 4));
            bs.alloc(count * (cache_half ? 2 : 4));
            const int blocks = static_cast<int>((count + THREADS - 1) / THREADS);
            Timing t = time_it(warmup, repeats, [&] {
                if (cache_half) {
                    LAUNCH(rpe::build_cache_kernel<RPE_HALF_TYPE>, blocks, THREADS,
                           reinterpret_cast<RPE_HALF_TYPE*>(bc.ptr),
                           reinterpret_cast<RPE_HALF_TYPE*>(bs.ptr), max_seq, THETA);
                } else {
                    LAUNCH(rpe::build_cache_kernel<float>, blocks, THREADS,
                           reinterpret_cast<float*>(bc.ptr), reinterpret_cast<float*>(bs.ptr),
                           max_seq, THETA);
                }
                GPU_CHECK(gpuGetLastError());
                GPU_CHECK(gpuDeviceSynchronize());
            });
            const size_t bytes = 2ull * count * (cache_half ? 2 : 4);
            fprintf(f, "%s    {\"form\": \"build-cache\", \"max_seq\": %d, \"dtype\": \"%s\", "
                       "\"p50_ms\": %.4f, \"p95_ms\": %.4f, \"min_ms\": %.4f, "
                       "\"warmup\": %d, \"repeats\": %d, \"bytes_written\": %zu, "
                       "\"gb_s\": %.2f, \"mem_footprint_mb\": %.2f}",
                    written ? ",\n" : "", max_seq, cache_half ? "fp16" : "fp32",
                    t.p50_ms, t.p95_ms, t.min_ms, t.warmup, t.repeats, bytes,
                    bytes / (t.p50_ms * 1e-3) / 1e9,
                    static_cast<double>(bytes) / (1024.0 * 1024.0));
            ++written;
            printf("[bench/build-cache] max_seq=%d dtype=%s: p50=%.4f ms (%.1f GB/s, %.1f MB)\n",
                   max_seq, cache_half ? "fp16" : "fp32", t.p50_ms,
                   bytes / (t.p50_ms * 1e-3) / 1e9, static_cast<double>(bytes) / 1048576.0);
            fflush(f);
            bc.free();
            bs.free();
        }
    }

    // 形态切换：prefill(2048) + decode(pos=2048) 组合 vs 两者单独之和
    {
        const int tokens = 2048;
        HostData host;
        host.gen(tokens);
        GpuData g;
        g.alloc_io(tokens, HQ, HKV, tokens + 1, false);
        g.upload(host.q, host.k);
        g.cast_io_to_f16();
        g.build_caches(tokens + 1, THETA, false);
        Timing tp = time_it(3, 30, [&] { launch_prefill(g, tokens, HQ, HKV, 0, true, false); });
        Timing td = time_it(3, 30, [&] { launch_decode(g, HQ, HKV, tokens, true, false); });
        Timing tc = time_it(3, 30, [&] {
            launch_prefill(g, tokens, HQ, HKV, 0, true, false);
            launch_decode(g, HQ, HKV, tokens, true, false);
        });
        const double overhead_ms = tc.p50_ms - (tp.p50_ms + td.p50_ms);
        fprintf(f, ",\n    {\"form\": \"switch-prefill+decode\", \"tokens\": %d, "
                   "\"prefill_p50_ms\": %.4f, \"decode_p50_ms\": %.4f, "
                   "\"combined_p50_ms\": %.4f, \"switch_overhead_ms\": %.4f}",
                tokens, tp.p50_ms, td.p50_ms, tc.p50_ms, overhead_ms);
        // 注意：written 已非 0，上面的逗号逻辑写错会破坏 JSON——改为下式重写
        (void)written;
        printf("[bench/switch] prefill=%.4f decode=%.4f combined=%.4f overhead=%.4f ms\n",
               tp.p50_ms, td.p50_ms, tc.p50_ms, overhead_ms);
        g.free_all();
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    return 0;
}

// ---------------------------------------------------------------------------
// apply 文件桥（真实权重单层验证）：desc.json + fp32 LE bin → kernel → 输出 bin
// ---------------------------------------------------------------------------

std::string json_get_str(const std::string& blob, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    size_t p = blob.find(pat);
    if (p == std::string::npos) return "";
    p = blob.find(':', p + pat.size());
    if (p == std::string::npos) return "";
    ++p;
    while (p < blob.size() && (blob[p] == ' ' || blob[p] == '\t')) ++p;
    if (p < blob.size() && blob[p] == '"') {
        const size_t e = blob.find('"', p + 1);
        return blob.substr(p + 1, e - p - 1);
    }
    const size_t e = blob.find_first_of(",}\n", p);
    return blob.substr(p, e - p);
}

double json_get_num(const std::string& blob, const std::string& key, double def) {
    const std::string s = json_get_str(blob, key);
    if (s.empty()) return def;
    return atof(s.c_str());
}

std::vector<float> read_bin_f32(const std::string& path, size_t expect_count) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "[apply] 无法打开输入 %s\n", path.c_str());
        exit(2);
    }
    fseek(fp, 0, SEEK_END);
    const long bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (static_cast<size_t>(bytes) != expect_count * 4) {
        fprintf(stderr, "[apply] %s 字节数 %ld != 期望 %zu\n", path.c_str(), bytes, expect_count * 4);
        exit(2);
    }
    std::vector<float> v(expect_count);
    if (fread(v.data(), 4, expect_count, fp) != expect_count) {
        fprintf(stderr, "[apply] %s 读取失败\n", path.c_str());
        exit(2);
    }
    fclose(fp);
    return v;
}

void write_bin_f32(const std::string& path, const std::vector<float>& v) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "[apply] 无法打开输出 %s\n", path.c_str());
        exit(2);
    }
    fwrite(v.data(), 4, v.size(), fp);
    fclose(fp);
}

int run_apply(const std::string& desc_path, const char* summary_name) {
    FILE* fp = fopen(desc_path.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "[apply] 无法打开 desc %s\n", desc_path.c_str());
        return 2;
    }
    std::string blob;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) blob += line;
    fclose(fp);

    const std::string form = json_get_str(blob, "form");
    const std::string io = json_get_str(blob, "io_dtype");
    const std::string cache = json_get_str(blob, "cache_dtype");
    const int num_tokens = static_cast<int>(json_get_num(blob, "num_tokens", 1));
    const int hq = static_cast<int>(json_get_num(blob, "hq", HQ));
    const int hkv = static_cast<int>(json_get_num(blob, "hkv", HKV));
    const int seq_offset = static_cast<int>(json_get_num(blob, "seq_offset", 0));
    const int head_dim = static_cast<int>(json_get_num(blob, "head_dim", HD));
    const int max_seq = static_cast<int>(json_get_num(blob, "max_seq", 0));
    if (head_dim != HD) {
        fprintf(stderr, "[apply] head_dim=%d 不支持（编译期固定 %d）\n", head_dim, HD);
        return 2;
    }
    if (max_seq <= 0) {
        fprintf(stderr, "[apply] desc 缺 max_seq\n");
        return 2;
    }
    const bool io_half = (io == "fp16");
    const bool cache_half = (cache == "fp16");
    const size_t q_count = static_cast<size_t>(num_tokens) * hq * HD;
    const size_t k_count = static_cast<size_t>(num_tokens) * hkv * HD;

    std::vector<float> q = read_bin_f32(json_get_str(blob, "input_q"), q_count);
    std::vector<float> k = read_bin_f32(json_get_str(blob, "input_k"), k_count);
    std::vector<float> cos_in = read_bin_f32(json_get_str(blob, "cos"),
                                             static_cast<size_t>(max_seq) * HALF);
    std::vector<float> sin_in = read_bin_f32(json_get_str(blob, "sin"),
                                             static_cast<size_t>(max_seq) * HALF);

    GpuData g;
    g.alloc_io(num_tokens, hq, hkv, max_seq, cache_half);
    g.upload(q, k);
    if (io_half) g.cast_io_to_f16();
    // cache 由文件给定（不从 theta 现算——真实权重路径的表由 python 侧生成）
    GPU_CHECK(gpuMemcpy(g.cache32c.ptr, cos_in.data(), g.cache_count * 4, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(g.cache32s.ptr, sin_in.data(), g.cache_count * 4, gpuMemcpyHostToDevice));
    if (cache_half) {
        const int blocks = static_cast<int>((g.cache_count + 1023) / 1024);
        LAUNCH(rpe::cast_f32_to_f16_kernel, blocks, 1024,
               reinterpret_cast<const float*>(g.cache32c.ptr),
               reinterpret_cast<RPE_HALF_TYPE*>(g.cache16c.ptr), g.cache_count);
        LAUNCH(rpe::cast_f32_to_f16_kernel, blocks, 1024,
               reinterpret_cast<const float*>(g.cache32s.ptr),
               reinterpret_cast<RPE_HALF_TYPE*>(g.cache16s.ptr), g.cache_count);
        GPU_CHECK(gpuGetLastError());
        GPU_CHECK(gpuDeviceSynchronize());
    }
    if (form == "decode") launch_decode(g, hq, hkv, seq_offset, io_half, cache_half);
    else launch_prefill(g, num_tokens, hq, hkv, seq_offset, io_half, cache_half);
    std::vector<float> qo, ko;
    download_output(g, io_half, qo, ko);
    g.free_all();

    write_bin_f32(json_get_str(blob, "output_q"), qo);
    write_bin_f32(json_get_str(blob, "output_k"), ko);

    FILE* f = fopen(summary_name, "w");
    fprintf(f, "{\n  \"form\": \"%s\",\n  \"io_dtype\": \"%s\",\n  \"cache_dtype\": \"%s\",\n"
               "  \"num_tokens\": %d,\n  \"seq_offset\": %d,\n  \"hq\": %d,\n  \"hkv\": %d,\n"
               "  \"checksum_q_out\": %.9f,\n  \"checksum_k_out\": %.9f,\n  \"warp\": %d\n}\n",
            form.c_str(), io.c_str(), cache.c_str(), num_tokens, seq_offset, hq, hkv,
            checksum(qo), checksum(ko), g_warp);
    fclose(f);
    printf("[apply] form=%s io=%s cache=%s T=%d pos=%d -> %s / %s (cs_q=%.6f cs_k=%.6f)\n",
           form.c_str(), io.c_str(), cache.c_str(), num_tokens, seq_offset,
           json_get_str(blob, "output_q").c_str(), json_get_str(blob, "output_k").c_str(),
           checksum(qo), checksum(ko));
    return 0;
}

// ---------------------------------------------------------------------------
// selftest
// ---------------------------------------------------------------------------

int run_selftest(int nthreads) {
    const int max_seq = 64;
    std::vector<double> cos64, sin64;
    host_build_tables_f64(max_seq, THETA, cos64, sin64);
    for (int s : {1, 3, 5}) {
        HostData host;
        host.gen(s);
        std::vector<float> tq, tk;
        host_apply<double>(host.q, host.k, tq, tk, s, HQ, HKV, 0, cos64, sin64, nthreads);
        GpuData g;
        g.alloc_io(s, HQ, HKV, max_seq, false);
        g.upload(host.q, host.k);
        g.cast_io_to_f16();
        g.build_caches(max_seq, THETA, false);
        launch_prefill(g, s, HQ, HKV, 0, true, false);
        std::vector<float> qo, ko;
        download_output(g, true, qo, ko);
        g.free_all();
        std::vector<double> ref;
        ref.reserve(tq.size() + tk.size());
        for (float x : tq) ref.push_back(x);
        for (float x : tk) ref.push_back(x);
        std::vector<float> got;
        got.reserve(ref.size());
        for (float x : qo) got.push_back(x);
        for (float x : ko) got.push_back(x);
        ErrStats st;
        st.accumulate(got, ref, nthreads);
        const bool ok = st.max_abs <= 5e-3 && st.nan_count == 0;
        printf("[selftest/prefill] T=%d max_abs=%.3e %s\n", s, st.max_abs, ok ? "PASS" : "FAIL");
        if (!ok) return 3;
    }
    for (int pos : {0, 2, 4}) {
        HostData host;
        host.gen(1);
        std::vector<float> tq, tk;
        host_apply_decode<double>(host.q, host.k, tq, tk, HQ, HKV, pos, cos64, sin64);
        GpuData g;
        g.alloc_io(1, HQ, HKV, max_seq, false);
        g.upload(host.q, host.k);
        g.cast_io_to_f16();
        g.build_caches(max_seq, THETA, false);
        launch_decode(g, HQ, HKV, pos, true, false);
        std::vector<float> qo, ko;
        download_output(g, true, qo, ko);
        g.free_all();
        std::vector<double> ref;
        ref.reserve(tq.size() + tk.size());
        for (float x : tq) ref.push_back(x);
        for (float x : tk) ref.push_back(x);
        std::vector<float> got;
        got.reserve(ref.size());
        for (float x : qo) got.push_back(x);
        for (float x : ko) got.push_back(x);
        ErrStats st;
        st.accumulate(got, ref, nthreads);
        const bool ok = st.max_abs <= 5e-3 && st.nan_count == 0;
        printf("[selftest/decode] pos=%d max_abs=%.3e %s\n", pos, st.max_abs, ok ? "PASS" : "FAIL");
        if (!ok) return 3;
    }
    printf("[selftest] all PASS\n");
    return 0;
}

}  // namespace rpe23

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "selftest";
    const char* platform = argc > 2 ? argv[2] : "musa";
    std::string env_name = std::string(platform) + std::string("-env.json");
    std::string corr_name = std::string(platform) + std::string("-correctness.json");
    std::string cache_name = std::string(platform) + std::string("-cache-dtype.json");
    std::string bench_name = std::string(platform) + std::string("-benchmark.json");
    const int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const int nthreads = std::min(hw_threads, 32);

    if (rpe23::run_attrs(env_name.c_str()) != 0) return 2;
    if (mode == "attrs") return 0;
    if (mode == "selftest") return rpe23::run_selftest(nthreads);
    if (mode == "correctness") return rpe23::run_correctness(nthreads, corr_name.c_str());
    if (mode == "cachedtype") return rpe23::run_cachedtype(nthreads, cache_name.c_str());
    if (mode == "bench") return rpe23::run_bench(bench_name.c_str());
    if (mode == "apply") {
        if (argc < 3) {
            fprintf(stderr, "用法: %s apply <desc.json> [platform]\n", argv[0]);
            return 1;
        }
        return rpe23::run_apply(argv[2], "apply-summary.json");
    }
    fprintf(stderr, "用法: %s {attrs|selftest|correctness|cachedtype|bench|apply <desc> [platform]}\n",
            argv[0]);
    return 1;
}
