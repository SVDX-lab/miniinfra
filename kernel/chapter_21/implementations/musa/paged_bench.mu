// ============================================================================
// paged_bench.mu —— 第 21 期 MUSA 列 standalone 二进制（ch18/ch21 模式）。
//
// 与 CUDA 列共用同一份核主体：implementations/cuda/paged_kernels.cuh
// （"机械替换单点化"迁移设计：API 差异集中在下方 gpu* shim）。
// 无 torch 依赖：自带 hash-RNG 数据（与 paged_config.mu_value 同算法）、
// 块表构造（identity / Fisher-Yates rand）、host scatter、CPU FP32 真值
// （std::thread 按 (b,h) 并行，分页池口径）、correctness / bench 矩阵、
// 事件对计时、JSON 输出——S4000 主机恢复后 run_session.sh 一键跑完。
//
// 用法：
//   ./paged_bench attrs         # 设备属性 + warpSize 双口径探针 → musa-env.json
//   ./paged_bench selftest      # 最小形状快速自检（N ∈ {1,3,65}）
//   ./paged_bench correctness   # 全 correctness 矩阵 → musa-correctness.json
//   ./paged_bench bench         # 基准矩阵 → musa-benchmark.json
//
// M-43 预算：decode 核静态 smem ≈3.2KB、prefill 核 ≈20.2/24.2KB —— 均在
//   MUSA 静态 shared 上限 28,672B 之内（ep21 实测约束，预算按此设计）。
// ============================================================================

#define USE_MUSA 1
#define ATTN_NO_FP16_INCLUDE 1

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>

#if defined(USE_MUSA)
  #include <musa_runtime.h>
  #if defined(__has_include)
    #if __has_include(<musa_fp16.h>)
      #include <musa_fp16.h>
      #define HAVE_MUSA_FP16_H 1
    #endif
  #endif
  #define gpuError_t             musaError_t
  #define gpuSuccess             musaSuccess
  #define gpuGetErrorString      musaGetErrorString
  #define gpuMalloc              musaMalloc
  #define gpuFree                musaFree
  #define gpuMemcpy              musaMemcpy
  #define gpuMemcpyHostToDevice  musaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost  musaMemcpyDeviceToHost
  #define gpuEvent_t             musaEvent_t
  #define gpuEventCreate         musaEventCreate
  #define gpuEventDestroy        musaEventDestroy
  #define gpuEventRecord         musaEventRecord
  #define gpuEventSynchronize    musaEventSynchronize
  #define gpuEventElapsedTime    musaEventElapsedTime
  #define gpuDeviceSynchronize   musaDeviceSynchronize
  #define gpuGetLastError        musaGetLastError
  #define gpuDeviceProp_t        musaDeviceProp
  #define gpuGetDeviceProperties musaGetDeviceProperties
#else
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  #define gpuError_t             cudaError_t
  #define gpuSuccess             cudaSuccess
  #define gpuGetErrorString      cudaGetErrorString
  #define gpuMalloc              cudaMalloc
  #define gpuFree                cudaFree
  #define gpuMemcpy              cudaMemcpy
  #define gpuMemcpyHostToDevice  cudaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost  cudaMemcpyDeviceToHost
  #define gpuEvent_t             cudaEvent_t
  #define gpuEventCreate         cudaEventCreate
  #define gpuEventDestroy        cudaEventDestroy
  #define gpuEventRecord         cudaEventRecord
  #define gpuEventSynchronize    cudaEventSynchronize
  #define gpuEventElapsedTime    cudaEventElapsedTime
  #define gpuDeviceSynchronize   cudaDeviceSynchronize
  #define gpuGetLastError        cudaGetLastError
  #define gpuDeviceProp_t        cudaDeviceProp
  #define gpuGetDeviceProperties cudaGetDeviceProperties
#endif

#include "../cuda/paged_kernels.cuh"   // pkg:: 核 + 探针

#define GPU_CHECK(expr)                                                      \
    do {                                                                     \
        gpuError_t err__ = (expr);                                           \
        if (err__ != gpuSuccess) {                                           \
            fprintf(stderr, "[gpu-error] %s:%d %s: %s\n", __FILE__,          \
                    __LINE__, #expr, gpuGetErrorString(err__));              \
            exit(2);                                                         \
        }                                                                    \
    } while (0)

namespace pb22 {

using pkg::D;             // 宿主代码裸用 D（ep21 `using atn::D` 同模式）
constexpr int HQ = 16;    // Qwen3-0.6B（与 paged_config.py 一致）
constexpr int HKV = 8;
constexpr int GROUP = HQ / HKV;
constexpr int BS_DEFAULT = 128;
constexpr float SCALE = 0.08838834764831845f;  // 1/sqrt(128)

constexpr uint32_t SEED_Q = 0x51002201u;
constexpr uint32_t SEED_K = 0x51002202u;
constexpr uint32_t SEED_V = 0x51002203u;
constexpr uint32_t SEED_T = 0x51002204u;   // 块表 Fisher-Yates 随机源

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
    return static_cast<float>(mu_hash(static_cast<uint32_t>(idx), seed) % 20001u)
           / 10000.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// host 数据与块表（分页合同实现，与 paged_config.py 块表语义一一对应）
// ---------------------------------------------------------------------------

struct HostCase {
    int b = 1;
    int n = 0;               // seq_len（每请求同长）
    int bs = BS_DEFAULT;
    int num_blocks = 0;
    int max_blocks = 0;      // 表宽 = ceil(n/bs)（bench 不留 padding；correctness +2）
    std::vector<float> q;    // [B*HQ*D]
    std::vector<float> klog; // [B*HKV*n*D] 逻辑连续
    std::vector<float> vlog;
    std::vector<float> kpool;  // [num_blocks*HKV*bs*D]
    std::vector<float> vpool;
    std::vector<int> table;    // [B*max_blocks]（correctness 留 -1 padding 行）
    std::vector<int> lens;     // [B]
    std::vector<float> kcontig;  // gather 期望输出（= klog）
    std::vector<float> vcontig;

    void build(int b_, int n_, int bs_, const char* mode, int pad_rows) {
        b = b_; n = n_; bs = bs_;
        const int nb_req = (n + bs - 1) / bs;
        max_blocks = nb_req + pad_rows;
        num_blocks = b * nb_req + 2;
        const size_t q_count = static_cast<size_t>(b) * HQ * D;
        const size_t kv_count = static_cast<size_t>(b) * HKV * n * D;
        q.resize(q_count); klog.resize(kv_count); vlog.resize(kv_count);
        for (size_t i = 0; i < q_count; ++i)
            q[i] = mu_value(i, SEED_Q);
        for (size_t i = 0; i < kv_count; ++i) {
            klog[i] = mu_value(i, SEED_K);
            vlog[i] = mu_value(i, SEED_V);
        }
        // 块表
        table.assign(static_cast<size_t>(b) * max_blocks, -1);
        if (std::strcmp(mode, "identity") == 0) {
            for (int bi = 0; bi < b; ++bi) {
                const int base = bi * nb_req;   // 请求独占连续段（B>1 identity 形态）
                for (int t = 0; t < nb_req; ++t)
                    table[static_cast<size_t>(bi) * max_blocks + t] = base + t;
            }
        } else {  // rand：全局 Fisher-Yates 置换后按请求切片（跨请求不重叠）
            std::vector<int> perm(num_blocks);
            for (int i = 0; i < num_blocks; ++i) perm[i] = i;
            for (int i = num_blocks - 1; i > 0; --i) {
                const int j = static_cast<int>(mu_hash(static_cast<size_t>(i), SEED_T)
                                               % static_cast<uint32_t>(i + 1));
                std::swap(perm[i], perm[j]);
            }
            int off = 0;
            for (int bi = 0; bi < b; ++bi) {
                for (int t = 0; t < nb_req; ++t)
                    table[static_cast<size_t>(bi) * max_blocks + t] = perm[off++];
            }
        }
        // scatter 逻辑 → 池
        const size_t pool_count = static_cast<size_t>(num_blocks) * HKV * bs * D;
        kpool.assign(pool_count, 0.0f);
        vpool.assign(pool_count, 0.0f);
        for (int bi = 0; bi < b; ++bi) {
            for (int kvh = 0; kvh < HKV; ++kvh) {
                for (int tok = 0; tok < n; ++tok) {
                    const int pblk = table[static_cast<size_t>(bi) * max_blocks
                                           + tok / bs];
                    const int ot = tok % bs;
                    const size_t src = ((static_cast<size_t>(bi) * HKV + kvh) * n
                                        + tok) * D;
                    const size_t dst = ((static_cast<size_t>(pblk) * HKV + kvh) * bs
                                        + ot) * D;
                    for (int d = 0; d < D; ++d) {
                        kpool[dst + d] = klog[src + d];
                        vpool[dst + d] = vlog[src + d];
                    }
                }
            }
        }
        lens.assign(b, n);
        kcontig = klog;   // gather 期望 = 逻辑连续
        vcontig = vlog;
    }
};

// ---------------------------------------------------------------------------
// CPU FP32 真值（分页池口径；std::thread 按 (b,h) 并行）
// ---------------------------------------------------------------------------

void cpu_paged_decode_head(const HostCase& c, float* o, int bh) {
    const int b = bh / HQ;
    const int h = bh % HQ;
    const int kvh = h / GROUP;
    const float* qb = c.q.data() + static_cast<size_t>(bh) * D;
    const int* trow = c.table.data() + static_cast<size_t>(b) * c.max_blocks;
    float* ob = o + static_cast<size_t>(bh) * D;
    float m = -INFINITY, l = 0.0f;
    std::vector<float> score(c.n), prob(c.n);
    for (int j = 0; j < c.n; ++j) {
        const int pblk = trow[j / c.bs];
        const int ot = j % c.bs;
        const float* kb = c.kpool.data()
            + ((static_cast<size_t>(pblk) * HKV + kvh) * c.bs + ot) * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += qb[d] * kb[d];
        score[j] = dot * SCALE;
        m = score[j] > m ? score[j] : m;
    }
    for (int j = 0; j < c.n; ++j) {
        prob[j] = expf(score[j] - m);
        l += prob[j];
    }
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < c.n; ++j) {
            const int pblk = trow[j / c.bs];
            const int ot = j % c.bs;
            const float* vb = c.vpool.data()
                + ((static_cast<size_t>(pblk) * HKV + kvh) * c.bs + ot) * D;
            acc += (prob[j] / l) * vb[d];
        }
        ob[d] = acc;
    }
}

void cpu_paged_decode_truth(const HostCase& c, std::vector<float>& o, int nthreads) {
    o.assign(static_cast<size_t>(c.b) * HQ * D, 0.0f);
    std::vector<std::thread> ths;
    const int total = c.b * HQ;
    int done = 0;
    for (int t = 0; t < nthreads && done < total; ++t) {
        ths.emplace_back([&, done]() {
            for (int bh = done; bh < total; bh += nthreads)
                cpu_paged_decode_head(c, o.data(), bh);
        });
        ++done;
    }
    for (auto& th : ths) th.join();
}

void cpu_contig_decode_head(const HostCase& c, float* o, int bh) {
    const int b = bh / HQ;
    const int h = bh % HQ;
    const int kvh = h / GROUP;
    const float* qb = c.q.data() + static_cast<size_t>(bh) * D;
    const float* kb = c.klog.data()
        + (static_cast<size_t>(b) * HKV + kvh) * c.n * D;
    const float* vb = c.vlog.data()
        + (static_cast<size_t>(b) * HKV + kvh) * c.n * D;
    float* ob = o + static_cast<size_t>(bh) * D;
    float m = -INFINITY, l = 0.0f;
    std::vector<float> score(c.n), prob(c.n);
    for (int j = 0; j < c.n; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += qb[d] * kb[j * D + d];
        score[j] = dot * SCALE;
        m = score[j] > m ? score[j] : m;
    }
    for (int j = 0; j < c.n; ++j) {
        prob[j] = expf(score[j] - m);
        l += prob[j];
    }
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < c.n; ++j) acc += (prob[j] / l) * vb[j * D + d];
        ob[d] = acc;
    }
}

void cpu_contig_decode_truth(const HostCase& c, std::vector<float>& o, int nthreads) {
    o.assign(static_cast<size_t>(c.b) * HQ * D, 0.0f);
    std::vector<std::thread> ths;
    const int total = c.b * HQ;
    int done = 0;
    for (int t = 0; t < nthreads && done < total; ++t) {
        ths.emplace_back([&, done]() {
            for (int bh = done; bh < total; bh += nthreads)
                cpu_contig_decode_head(c, o.data(), bh);
        });
        ++done;
    }
    for (auto& th : ths) th.join();
}

// CPU gather（池 → 连续；供 gather 核位级对账）
void cpu_gather(const HostCase& c, std::vector<float>& kg, std::vector<float>& vg) {
    kg.assign(c.klog.size(), 0.0f);
    vg.assign(c.vlog.size(), 0.0f);
    for (int bi = 0; bi < c.b; ++bi) {
        for (int kvh = 0; kvh < HKV; ++kvh) {
            for (int tok = 0; tok < c.n; ++tok) {
                const int pblk = c.table[static_cast<size_t>(bi) * c.max_blocks
                                         + tok / c.bs];
                const int ot = tok % c.bs;
                const size_t src = ((static_cast<size_t>(pblk) * HKV + kvh) * c.bs
                                    + ot) * D;
                const size_t dst = ((static_cast<size_t>(bi) * HKV + kvh) * c.n
                                    + tok) * D;
                for (int d = 0; d < D; ++d) {
                    kg[dst + d] = c.kpool[src + d];
                    vg[dst + d] = c.vpool[src + d];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU 缓冲
// ---------------------------------------------------------------------------

struct DeviceBuf {
    void* ptr = nullptr;
    size_t bytes = 0;
    void alloc(size_t nbytes) { bytes = nbytes; GPU_CHECK(gpuMalloc(&ptr, nbytes)); }
    void free() { if (ptr) GPU_CHECK(gpuFree(ptr)); ptr = nullptr; }
};

__global__ void cast_f32_to_f16_kernel(const float* src, ATTN_HALF_TYPE* dst,
                                       size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2half(src[i]);
}

__global__ void cast_f16_to_f32_kernel(const ATTN_HALF_TYPE* src, float* dst,
                                       size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __half2float(src[i]);
}

struct GpuData {
    DeviceBuf q32, kcontig32, vcontig32, kpool32, vpool32, table32, lens32;
    DeviceBuf q16, kcontig16, vcontig16, kpool16, vpool16;
    DeviceBuf o16, o32;
    size_t q_count = 0, kv_count = 0, pool_count = 0, o_count = 0;
    int b = 1, n = 0, bs = BS_DEFAULT, num_blocks = 0, max_blocks = 0;

    void alloc(const HostCase& c) {
        b = c.b; n = c.n; bs = c.bs; num_blocks = c.num_blocks; max_blocks = c.max_blocks;
        q_count = static_cast<size_t>(c.b) * HQ * D;
        kv_count = static_cast<size_t>(c.b) * HKV * c.n * D;
        pool_count = static_cast<size_t>(c.num_blocks) * HKV * c.bs * D;
        o_count = q_count;
        q32.alloc(q_count * 4);
        kcontig32.alloc(kv_count * 4); vcontig32.alloc(kv_count * 4);
        kpool32.alloc(pool_count * 4); vpool32.alloc(pool_count * 4);
        table32.alloc(c.table.size() * 4);
        lens32.alloc(c.lens.size() * 4);
        q16.alloc(q_count * 2);
        kcontig16.alloc(kv_count * 2); vcontig16.alloc(kv_count * 2);
        kpool16.alloc(pool_count * 2); vpool16.alloc(pool_count * 2);
        o16.alloc(o_count * 2); o32.alloc(o_count * 4);
    }
    void upload(const HostCase& c) {
        GPU_CHECK(gpuMemcpy(q32.ptr, c.q.data(), q_count * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(kcontig32.ptr, c.klog.data(), kv_count * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(vcontig32.ptr, c.vlog.data(), kv_count * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(kpool32.ptr, c.kpool.data(), pool_count * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(vpool32.ptr, c.vpool.data(), pool_count * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(table32.ptr, c.table.data(), c.table.size() * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(lens32.ptr, c.lens.data(), c.lens.size() * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuDeviceSynchronize());
    }
    void cast_to_f16() {
        auto cast = [](DeviceBuf& s32, DeviceBuf& s16, size_t count) {
            const int blocks = static_cast<int>((count + 1023) / 1024);
            cast_f32_to_f16_kernel<<<blocks, 1024>>>(
                reinterpret_cast<const float*>(s32.ptr),
                reinterpret_cast<ATTN_HALF_TYPE*>(s16.ptr), count);
        };
        cast(q32, q16, q_count);
        cast(kcontig32, kcontig16, kv_count);
        cast(vcontig32, vcontig16, kv_count);
        cast(kpool32, kpool16, pool_count);
        cast(vpool32, vpool16, pool_count);
        GPU_CHECK(gpuGetLastError());
        GPU_CHECK(gpuDeviceSynchronize());
    }
    void download_output(bool is_half, float* host_o) {
        if (is_half) {
            const int blocks = static_cast<int>((o_count + 1023) / 1024);
            cast_f16_to_f32_kernel<<<blocks, 1024>>>(
                reinterpret_cast<const ATTN_HALF_TYPE*>(o16.ptr),
                reinterpret_cast<float*>(o32.ptr), o_count);
            GPU_CHECK(gpuGetLastError());
            GPU_CHECK(gpuDeviceSynchronize());
            GPU_CHECK(gpuMemcpy(host_o, o32.ptr, o_count * 4, gpuMemcpyDeviceToHost));
        } else {
            GPU_CHECK(gpuMemcpy(host_o, o32.ptr, o_count * 4, gpuMemcpyDeviceToHost));
        }
    }
    void free_all() {
        q32.free(); kcontig32.free(); vcontig32.free();
        kpool32.free(); vpool32.free(); table32.free(); lens32.free();
        q16.free(); kcontig16.free(); vcontig16.free(); kpool16.free(); vpool16.free();
        o16.free(); o32.free();
    }
};

int g_warp = 0;

void probe_warp() {
    if (g_warp > 0) return;
    DeviceBuf buf;
    buf.alloc(4);
    pkg::warp_size_probe<<<1, 32>>>(reinterpret_cast<int*>(buf.ptr));
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    int host = 0;
    GPU_CHECK(gpuMemcpy(&host, buf.ptr, 4, gpuMemcpyDeviceToHost));
    buf.free();
    g_warp = host;
}

// decode 发射（T 双实例；is_half 决定缓冲源；块表内容在 upload 时已定）
template <typename T>
void launch_paged_decode(GpuData& g, bool is_half, float scale) {
    const T* qp = is_half ? reinterpret_cast<const T*>(g.q16.ptr)
                          : reinterpret_cast<const T*>(g.q32.ptr);
    const T* kp = is_half ? reinterpret_cast<const T*>(g.kpool16.ptr)
                          : reinterpret_cast<const T*>(g.kpool32.ptr);
    const T* vp = is_half ? reinterpret_cast<const T*>(g.vpool16.ptr)
                          : reinterpret_cast<const T*>(g.vpool32.ptr);
    T* op = is_half ? reinterpret_cast<T*>(g.o16.ptr) : reinterpret_cast<T*>(g.o32.ptr);
    const int* tbl = reinterpret_cast<const int*>(g.table32.ptr);
    const int* lens = reinterpret_cast<const int*>(g.lens32.ptr);
    const unsigned grid = static_cast<unsigned>(g.b * HQ);
    if (g_warp == 128) {
        pkg::paged_decode_kernel<T, 128><<<grid, pkg::THREADS>>>(
            qp, kp, vp, tbl, lens, op, HQ, HKV, g.bs, g.max_blocks, scale);
    } else {
        pkg::paged_decode_kernel<T, 32><<<grid, pkg::THREADS>>>(
            qp, kp, vp, tbl, lens, op, HQ, HKV, g.bs, g.max_blocks, scale);
    }
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
}

template <typename T>
void launch_contig_decode(GpuData& g, bool is_half, float scale) {
    const T* qp = is_half ? reinterpret_cast<const T*>(g.q16.ptr)
                          : reinterpret_cast<const T*>(g.q32.ptr);
    const T* kp = is_half ? reinterpret_cast<const T*>(g.kcontig16.ptr)
                          : reinterpret_cast<const T*>(g.kcontig32.ptr);
    const T* vp = is_half ? reinterpret_cast<const T*>(g.vcontig16.ptr)
                          : reinterpret_cast<const T*>(g.vcontig32.ptr);
    T* op = is_half ? reinterpret_cast<T*>(g.o16.ptr) : reinterpret_cast<T*>(g.o32.ptr);
    const int* lens = reinterpret_cast<const int*>(g.lens32.ptr);
    const unsigned grid = static_cast<unsigned>(g.b * HQ);
    if (g_warp == 128) {
        pkg::contig_decode_kernel<T, 128><<<grid, pkg::THREADS>>>(
            qp, kp, vp, lens, op, HQ, HKV, g.n, scale);
    } else {
        pkg::contig_decode_kernel<T, 32><<<grid, pkg::THREADS>>>(
            qp, kp, vp, lens, op, HQ, HKV, g.n, scale);
    }
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
}

template <typename T>
void launch_gather(GpuData& g, bool is_half, DeviceBuf& kout, DeviceBuf& vout) {
    const T* kp = is_half ? reinterpret_cast<const T*>(g.kpool16.ptr)
                          : reinterpret_cast<const T*>(g.kpool32.ptr);
    const T* vp = is_half ? reinterpret_cast<const T*>(g.vpool16.ptr)
                          : reinterpret_cast<const T*>(g.vpool32.ptr);
    T* ko = reinterpret_cast<T*>(kout.ptr);
    T* vo = reinterpret_cast<T*>(vout.ptr);
    const int* tbl = reinterpret_cast<const int*>(g.table32.ptr);
    const int* lens = reinterpret_cast<const int*>(g.lens32.ptr);
    const long long total = 2LL * g.b * HKV * g.n * D;
    const unsigned grid = static_cast<unsigned>((total + pkg::THREADS - 1) / pkg::THREADS);
    pkg::gather_kv_kernel<T><<<grid, pkg::THREADS>>>(
        kp, vp, tbl, lens, ko, vo, g.b, HKV, g.bs, g.max_blocks, g.n, total);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// 工具：误差与 JSON
// ---------------------------------------------------------------------------

struct ErrStats { double max_abs = 0.0, mean_abs = 0.0, rel = 0.0; };

ErrStats compare(const std::vector<float>& got, const std::vector<float>& want) {
    ErrStats e;
    double sum = 0.0, wmax = 1e-30;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - want[i]);
        e.max_abs = d > e.max_abs ? d : e.max_abs;
        sum += d;
        const double w = std::fabs(static_cast<double>(want[i]));
        wmax = w > wmax ? w : wmax;
    }
    e.mean_abs = sum / static_cast<double>(got.size());
    e.rel = e.max_abs / wmax;
    return e;
}

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void write_json(const char* path, const std::string& content) {
    FILE* f = std::fopen(path, "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); exit(3); }
    std::fputs(content.c_str(), f);
    std::fclose(f);
    std::printf("[json] wrote %s\n", path);
}

// ---------------------------------------------------------------------------
// 命令：attrs / selftest / correctness / bench
// ---------------------------------------------------------------------------

int cmd_attrs() {
    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, 0));
    probe_warp();
    char buf[4096];
    std::snprintf(buf, sizeof(buf),
        "{\n  \"backend\": \"musa\",\n  \"gpu_name\": \"%s\",\n"
        "  \"multi_processor_count\": %d,\n"
        "  \"shared_mem_per_block\": %zu,\n"
        "  \"warp_size_builtin\": %d,\n"
        "  \"warp_size_probe_kernel\": %d,\n"
        "  \"total_global_mem_gb\": %.1f,\n"
        "  \"kernel_smem_budget_note\": \"decode ~3.2KB / prefill ~20.2KB(fp16) "
        "—— MUSA 静态 smem 上限 28672B 内（M-43）\",\n"
        "  \"episode\": 22\n}\n",
        prop.name, prop.multiProcessorCount,
        static_cast<size_t>(prop.sharedMemPerBlock),
        static_cast<int>(prop.warpSize), g_warp,
        static_cast<double>(prop.totalGlobalMem) / 1e9);
    write_json("musa-env.json", buf);
    std::printf("%s\n", buf);
    return 0;
}

int cmd_selftest() {
    probe_warp();
    int failures = 0;
    const int cases[][2] = {{1, 16}, {3, 16}, {65, 128}};
    for (const auto& tc : cases) {
        HostCase c;
        c.build(1, tc[0], tc[1], "rand", 1);
        std::vector<float> truth;
        cpu_paged_decode_truth(c, truth, 8);
        GpuData g;
        g.alloc(c); g.upload(c); g.cast_to_f16();
        launch_paged_decode<ATTN_HALF_TYPE>(g, true, SCALE);
        std::vector<float> got(g.o_count);
        g.download_output(true, got.data());
        const ErrStats e = compare(got, truth);
        const bool ok = e.rel < 2e-3;
        if (!ok) ++failures;
        std::printf("[selftest] N=%d bs=%d fp16: max=%.3e rel=%.2e %s\n",
                    tc[0], tc[1], e.max_abs, e.rel, ok ? "OK" : "FAIL");
        g.free_all();
    }
    std::printf("== selftest 总结：%d 失败 ==\n", failures);
    return failures ? 1 : 0;
}

int cmd_correctness() {
    probe_warp();
    int failures = 0;
    std::string json = "{\n  \"backend\": \"musa\",\n  \"schema_version\": "
        "\"kernel-chapter22.correctness.v1\",\n  \"value_transparency_claim\": "
        "\"paged kernel 与 contig kernel 输出位级一致；vs CPU FP32 真值容差报告\",\n"
        "  \"cases\": [\n";
    bool first = true;
    long cases_total = 0, cases_ok = 0;

    auto run_case = [&](int b, int n, int bs, const char* mode, bool is_half,
                        bool gather_check) {
        HostCase c;
        c.build(b, n, bs, mode, 2);
        std::vector<float> truth_paged, truth_contig;
        cpu_paged_decode_truth(c, truth_paged, 16);
        cpu_contig_decode_truth(c, truth_contig, 16);
        GpuData g;
        g.alloc(c); g.upload(c);
        if (is_half) g.cast_to_f16();
        // paged kernel
        std::vector<float> o_paged(g.o_count);
        if (is_half) {
            launch_paged_decode<ATTN_HALF_TYPE>(g, true, SCALE);
            g.download_output(true, o_paged.data());
        } else {
            launch_paged_decode<float>(g, false, SCALE);
            g.download_output(false, o_paged.data());
        }
        // contig kernel（复用 o 缓冲）
        std::vector<float> o_contig(g.o_count);
        if (is_half) {
            launch_contig_decode<ATTN_HALF_TYPE>(g, true, SCALE);
            g.download_output(true, o_contig.data());
        } else {
            launch_contig_decode<float>(g, false, SCALE);
            g.download_output(false, o_contig.data());
        }
        const ErrStats e_p = compare(o_paged, truth_paged);
        const ErrStats e_c = compare(o_contig, truth_contig);
        const bool bit = bit_equal(o_paged, o_contig);
        const double tol = is_half ? 2e-3 : 5e-5;
        const bool ok = e_p.rel < tol && e_c.rel < tol && bit;
        // gather 位级检查（fp32 主口径一次即可）
        bool gather_ok = true;
        if (gather_check && !is_half) {
            DeviceBuf kout, vout;
            kout.alloc(c.klog.size() * 4); vout.alloc(c.vlog.size() * 4);
            launch_gather<float>(g, false, kout, vout);
            std::vector<float> kg(c.klog.size()), vg(c.vlog.size());
            GPU_CHECK(gpuMemcpy(kg.data(), kout.ptr, c.klog.size() * 4, gpuMemcpyDeviceToHost));
            GPU_CHECK(gpuMemcpy(vg.data(), vout.ptr, c.vlog.size() * 4, gpuMemcpyDeviceToHost));
            kout.free(); vout.free();
            gather_ok = bit_equal(kg, c.kcontig) && bit_equal(vg, c.vcontig);
        }
        ++cases_total;
        if (ok && gather_ok) ++cases_ok; else ++failures;
        char entry[1024];
        std::snprintf(entry, sizeof(entry),
            "%s    {\"case\": \"n%d_b%d_bs%d_%s_%s\", \"dtype\": \"%s\", "
            "\"paged_vs_truth\": {\"max_abs\": %.3e, \"rel\": %.3e}, "
            "\"contig_vs_truth\": {\"max_abs\": %.3e, \"rel\": %.3e}, "
            "\"bit_paged_vs_contig\": %s, \"gather_bitexact\": %s, \"pass\": %s}",
            first ? "" : ",\n", n, b, bs, mode, is_half ? "fp16" : "fp32",
            is_half ? "fp16" : "fp32",
            e_p.max_abs, e_p.rel, e_c.max_abs, e_c.rel,
            bit ? "true" : "false", gather_ok ? "true" : "false",
            (ok && gather_ok) ? "true" : "false");
        json += entry;
        first = false;
        std::printf("[correct] n%d_b%d_bs%d_%s_%s: paged(max=%.2e rel=%.2e) "
                    "contig(max=%.2e) bit=%d gather=%d %s\n",
                    n, b, bs, mode, is_half ? "fp16" : "fp32",
                    e_p.max_abs, e_p.rel, e_c.max_abs,
                    bit ? 1 : 0, gather_ok ? 1 : 0, ok ? "OK" : "FAIL");
        g.free_all();
    };

    // fp16 主矩阵
    for (int n : {1, 65, 511, 2048}) {
        for (int b : {1, 4}) {
            for (int bs : {16, 128}) {
                for (const char* mode : {"identity", "rand"}) {
                    run_case(b, n, bs, mode, true, false);
                }
            }
        }
    }
    // fp32 子集（含 gather 位级检查）
    for (int n : {65, 511}) {
        run_case(1, n, 128, "rand", false, true);
        run_case(1, n, 128, "identity", false, false);
    }
    // fp16 大形状
    run_case(1, 8191, 128, "rand", true, false);

    json += "\n  ],\n";
    char sum[256];
    std::snprintf(sum, sizeof(sum),
                  "  \"summary\": {\"total\": %ld, \"pass\": %ld, \"fail\": %ld},\n"
                  "  \"tolerance\": {\"fp16_rel\": 2e-3, \"fp32_rel\": 5e-5}\n}\n",
                  cases_total, cases_ok, static_cast<long>(failures));
    json += sum;
    write_json("musa-correctness.json", json);
    std::printf("== correctness 总结：%ld/%ld 通过，%d 失败 ==\n",
                cases_ok, cases_total, failures);
    return failures ? 1 : 0;
}

double bench_once_paged(GpuData& g, bool is_half, int warmup, int repeats) {
    gpuEvent_t ev0, ev1;
    GPU_CHECK(gpuEventCreate(&ev0));
    GPU_CHECK(gpuEventCreate(&ev1));
    std::vector<float> dummy(g.o_count);
    for (int i = 0; i < warmup; ++i) {
        if (is_half) launch_paged_decode<ATTN_HALF_TYPE>(g, true, SCALE);
        else launch_paged_decode<float>(g, false, SCALE);
    }
    std::vector<float> times;
    for (int i = 0; i < repeats; ++i) {
        GPU_CHECK(gpuEventRecord(ev0));
        if (is_half) launch_paged_decode<ATTN_HALF_TYPE>(g, true, SCALE);
        else launch_paged_decode<float>(g, false, SCALE);
        GPU_CHECK(gpuEventRecord(ev1));
        GPU_CHECK(gpuEventSynchronize(ev1));
        float ms = 0.0f;
        GPU_CHECK(gpuEventElapsedTime(&ms, ev0, ev1));
        times.push_back(ms);
    }
    GPU_CHECK(gpuEventDestroy(ev0));
    GPU_CHECK(gpuEventDestroy(ev1));
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

double bench_once_contig(GpuData& g, bool is_half, int warmup, int repeats) {
    gpuEvent_t ev0, ev1;
    GPU_CHECK(gpuEventCreate(&ev0));
    GPU_CHECK(gpuEventCreate(&ev1));
    for (int i = 0; i < warmup; ++i) {
        if (is_half) launch_contig_decode<ATTN_HALF_TYPE>(g, true, SCALE);
        else launch_contig_decode<float>(g, false, SCALE);
    }
    std::vector<float> times;
    for (int i = 0; i < repeats; ++i) {
        GPU_CHECK(gpuEventRecord(ev0));
        if (is_half) launch_contig_decode<ATTN_HALF_TYPE>(g, true, SCALE);
        else launch_contig_decode<float>(g, false, SCALE);
        GPU_CHECK(gpuEventRecord(ev1));
        GPU_CHECK(gpuEventSynchronize(ev1));
        float ms = 0.0f;
        GPU_CHECK(gpuEventElapsedTime(&ms, ev0, ev1));
        times.push_back(ms);
    }
    GPU_CHECK(gpuEventDestroy(ev0));
    GPU_CHECK(gpuEventDestroy(ev1));
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

int cmd_bench() {
    probe_warp();
    std::string json = "{\n  \"backend\": \"musa\",\n  \"schema_version\": "
        "\"kernel-chapter22.benchmark.v1\",\n  \"warp_device_builtin\": "
        + std::to_string(g_warp) + ",\n  \"entries\": [\n";
    bool first = true;
    auto emit = [&](const std::string& s) {
        json += (first ? "" : ",\n") + s;
        first = false;
    };
    const int warm = 2;
    auto plan = [](int n, int b) { return (n <= 512) ? 30 : ((n <= 2048) ? 20 : 10); };

    for (int n : {512, 2048, 8192}) {
        for (int b : {1, 4}) {
            const int rep = plan(n, b);
            const int bs = BS_DEFAULT;
            // identity 表（顺序档）
            {
                HostCase c;
                c.build(b, n, bs, "identity", 0);
                GpuData g;
                g.alloc(c); g.upload(c); g.cast_to_f16();
                const double tc = bench_once_contig(g, true, warm, rep);
                const double tp = bench_once_paged(g, true, warm, rep);
                const double bytes = 2.0 * b * HKV * n * D * 2 + 2.0 * b * HQ * D * 2;
                char b1[512], b2[512];
                std::snprintf(b1, sizeof(b1),
                    "    {\"variant\": \"contig_kernel\", \"batch\": %d, \"kv_len\": %d, "
                    "\"dtype\": \"fp16\", \"block_size\": %d, \"table_mode\": \"none\", "
                    "\"p50_ms\": %.4f, \"achieved_gbs_unique\": %.1f}",
                    b, n, bs, tc, bytes / (tc / 1000.0) / 1e9);
                std::snprintf(b2, sizeof(b2),
                    "    {\"variant\": \"paged_kernel\", \"batch\": %d, \"kv_len\": %d, "
                    "\"dtype\": \"fp16\", \"block_size\": %d, \"table_mode\": \"seq\", "
                    "\"p50_ms\": %.4f, \"achieved_gbs_unique\": %.1f}",
                    b, n, bs, tp, bytes / (tp / 1000.0) / 1e9);
                emit(b1);
                emit(b2);
                std::printf("[bench] contig N=%d B=%d: %.4f ms | paged-seq: %.4f ms\n",
                            n, b, tc, tp);
                g.free_all();
            }
            // rand 表
            {
                HostCase c;
                c.build(b, n, bs, "rand", 0);
                GpuData g;
                g.alloc(c); g.upload(c); g.cast_to_f16();
                const double tp = bench_once_paged(g, true, warm, rep);
                char b3[512];
                const double bytes = 2.0 * b * HKV * n * D * 2 + 2.0 * b * HQ * D * 2;
                std::snprintf(b3, sizeof(b3),
                    "    {\"variant\": \"paged_kernel\", \"batch\": %d, \"kv_len\": %d, "
                    "\"dtype\": \"fp16\", \"block_size\": %d, \"table_mode\": \"rand\", "
                    "\"p50_ms\": %.4f, \"achieved_gbs_unique\": %.1f}",
                    b, n, bs, tp, bytes / (tp / 1000.0) / 1e9);
                emit(b3);
                std::printf("[bench] paged-rand N=%d B=%d: %.4f ms\n", n, b, tp);
                g.free_all();
            }
        }
    }
    // fp32 抽查（N=2048 B=1）
    {
        HostCase c;
        c.build(1, 2048, BS_DEFAULT, "rand", 0);
        GpuData g;
        g.alloc(c); g.upload(c);
        const double tc = bench_once_contig(g, false, 2, 10);
        const double tp = bench_once_paged(g, false, 2, 10);
        char b4[512];
        std::snprintf(b4, sizeof(b4),
            "    {\"variant\": \"contig_kernel\", \"batch\": 1, \"kv_len\": 2048, "
            "\"dtype\": \"fp32\", \"block_size\": %d, \"table_mode\": \"none\", "
            "\"p50_ms\": %.4f},\n    {\"variant\": \"paged_kernel\", \"batch\": 1, "
            "\"kv_len\": 2048, \"dtype\": \"fp32\", \"block_size\": %d, "
            "\"table_mode\": \"rand\", \"p50_ms\": %.4f}",
            BS_DEFAULT, tc, BS_DEFAULT, tp);
        emit(b4);
        std::printf("[bench] fp32 N=2048: contig %.4f ms | paged-rand %.4f ms\n", tc, tp);
        g.free_all();
    }
    json += "\n  ]\n}\n";
    write_json("musa-benchmark.json", json);
    return 0;
}

}  // namespace pb22

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法: %s {attrs|selftest|correctness|bench}\n", argv[0]);
        return 1;
    }
    std::string cmd = argv[1];
    if (cmd == "attrs") return pb22::cmd_attrs();
    if (cmd == "selftest") return pb22::cmd_selftest();
    if (cmd == "correctness") return pb22::cmd_correctness();
    if (cmd == "bench") return pb22::cmd_bench();
    std::fprintf(stderr, "未知命令: %s\n", cmd.c_str());
    return 1;
}
