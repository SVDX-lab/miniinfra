// ============================================================================
// attention_bench.mu —— 第 20 期 MUSA 列 standalone 二进制（ch18 模式）。
//
// 与 CUDA 列共用同一份核主体：implementations/cuda/attention_kernels.cuh
// （API 差异集中在下方 gpu* shim——ch18 "机械替换单点化"迁移设计）。
// 无 torch 依赖：自带 hash-RNG 数据生成（与 attention_config.mu_hash 同算法，
// manifest 记录校验和）、CPU FP32 逐元素真值（std::thread 并行）、
// correctness 矩阵、benchmark 矩阵、事件对计时、JSON 输出。
//
// 用法：
//   ./attention_bench attrs         # 设备属性 + warpSize 双口径探针 → musa-env.json
//   ./attention_bench selftest      # 最小形状快速自检（S/N ∈ {1,3,5}）
//   ./attention_bench correctness   # 全 correctness 矩阵 → musa-correctness.json
//   ./attention_bench bench         # 基准矩阵 → musa-benchmark.json
// ============================================================================

#define USE_MUSA 1          // MUSA 入口文件必须自带（ch18 模式；缺它则 #else 分支找 cuda_runtime.h——首败实录）
#define ATTN_NO_FP16_INCLUDE 1

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

// ---------------------------------------------------------------------------
// 平台 shim（ch18 验证过的机械替换单点化；M-34：传输族 cuda*→musa* 全量可用）
// ---------------------------------------------------------------------------
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
  #define gpuDeviceProp_t        musaDeviceProp
  #define gpuGetDeviceProperties musaGetDeviceProperties
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
#else
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  #define gpuError_t             cudaError_t
  #define gpuSuccess             cudaSuccess
  #define gpuGetErrorString      cudaGetErrorString
  #define gpuDeviceProp_t        cudaDeviceProp
  #define gpuGetDeviceProperties cudaGetDeviceProperties
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
#endif

#include "attention_kernels.cuh"   // atn:: 双形态核 + 物化变体 + 探针

#define GPU_CHECK(expr)                                                      \
    do {                                                                     \
        gpuError_t err__ = (expr);                                           \
        if (err__ != gpuSuccess) {                                           \
            fprintf(stderr, "[gpu-error] %s:%d %s: %s\n", __FILE__,          \
                    __LINE__, #expr, gpuGetErrorString(err__));              \
            exit(2);                                                         \
        }                                                                    \
    } while (0)

namespace atn21 {

using atn::D;
constexpr int HQ = 16;   // Qwen3-0.6B（与 attention_config.py 一致）
constexpr int HKV = 8;
constexpr int GROUP = HQ / HKV;
constexpr float SCALE = 0.08838834764831845f;  // 1/sqrt(128)

// ---------------------------------------------------------------------------
// hash RNG（与 attention_config.mu_hash 逐位一致：uint32 溢出语义）
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

constexpr uint32_t SEED_Q = 0x51002101u;
constexpr uint32_t SEED_K = 0x51002102u;
constexpr uint32_t SEED_V = 0x51002103u;

// 简单加法校验和（与 numpy 侧逐位可复算；用于跨平台数据一致性抽查）
struct Checksums { double q, k, v; };

Checksums checksum3(const std::vector<float>& q, const std::vector<float>& k,
                    const std::vector<float>& v) {
    Checksums c{0.0, 0.0, 0.0};
    for (float x : q) c.q += x;
    for (float x : k) c.k += x;
    for (float x : v) c.v += x;
    return c;
}

// ---------------------------------------------------------------------------
// CPU FP32 真值（逐元素循环；std::thread 按 (b,h) 并行）
// ---------------------------------------------------------------------------

void cpu_prefill_head(const float* q, const float* k, const float* v, float* o,
                      int hq, int hkv, int s, float scale, int bh) {
    const int b = bh / hq;
    const int h = bh % hq;
    const int kvh = h / GROUP;
    const float* qb = q + static_cast<size_t>(bh) * s * D;
    const float* kb = k + (static_cast<size_t>(b) * hkv + kvh) * s * D;
    const float* vb = v + (static_cast<size_t>(b) * hkv + kvh) * s * D;
    float* ob = o + static_cast<size_t>(bh) * s * D;
    std::vector<float> score(s), prob(s);
    for (int qi = 0; qi < s; ++qi) {
        float m = -INFINITY;
        for (int j = 0; j <= qi; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += qb[qi * D + d] * kb[j * D + d];
            score[j] = dot * scale;
            m = score[j] > m ? score[j] : m;
        }
        float l = 0.0f;
        for (int j = 0; j <= qi; ++j) {
            prob[j] = expf(score[j] - m);
            l += prob[j];
        }
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int j = 0; j <= qi; ++j) acc += (prob[j] / l) * vb[j * D + d];
            ob[qi * D + d] = acc;
        }
    }
}

void cpu_prefill_truth(const float* q, const float* k, const float* v, float* o,
                       int b, int hq, int s, int nthreads) {
    std::vector<std::thread> ths;
    const int total = b * hq;
    int done = 0;
    for (int t = 0; t < nthreads && done < total; ++t) {
        ths.emplace_back([&, done]() {
            for (int bh = done; bh < total; bh += nthreads) {
                cpu_prefill_head(q, k, v, o, hq, HKV, s, SCALE, bh);
            }
        });
        ++done;
    }
    for (auto& th : ths) th.join();
}

void cpu_decode_head(const float* q, const float* kc, const float* vc, float* o,
                     int hq, int n, int bh) {
    const int b = bh / hq;
    const int h = bh % hq;
    const int kvh = h / GROUP;
    const float* qb = q + static_cast<size_t>(bh) * D;
    const float* kb = kc + (static_cast<size_t>(b) * HKV + kvh) * n * D;
    const float* vb = vc + (static_cast<size_t>(b) * HKV + kvh) * n * D;
    float* ob = o + static_cast<size_t>(bh) * D;
    float m = -INFINITY, l = 0.0f;
    std::vector<float> score(n), prob(n);
    for (int j = 0; j < n; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += qb[d] * kb[j * D + d];
        score[j] = dot * SCALE;
        m = score[j] > m ? score[j] : m;
    }
    for (int j = 0; j < n; ++j) {
        prob[j] = expf(score[j] - m);
        l += prob[j];
    }
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < n; ++j) acc += (prob[j] / l) * vb[j * D + d];
        ob[d] = acc;
    }
}

void cpu_decode_truth(const float* q, const float* kc, const float* vc, float* o,
                      int b, int hq, int n, int nthreads) {
    std::vector<std::thread> ths;
    const int total = b * hq;
    int done = 0;
    for (int t = 0; t < nthreads && done < total; ++t) {
        ths.emplace_back([&, done]() {
            for (int bh = done; bh < total; bh += nthreads) {
                cpu_decode_head(q, kc, vc, o, hq, n, bh);
            }
        });
        ++done;
    }
    for (auto& th : ths) th.join();
}

// ---------------------------------------------------------------------------
// GPU 缓冲与发射（T = float / ATTN_HALF_TYPE 双实例）
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
    DeviceBuf q32, k32, v32;   // fp32 master（host 上传）
    DeviceBuf q16, k16, v16;   // 转换后的 io 缓冲
    DeviceBuf o16, o32;        // 输出（fp16 io 时 o32 供回读）
    DeviceBuf score_buf, attn_buf, lrow_buf;   // 物化变体（预分配，bench 不进计时）
    size_t q_count, kv_count, o_count;
    size_t score_count = 0;

    void alloc_prefill(int b, int s) {
        q_count = static_cast<size_t>(b) * HQ * s * D;
        kv_count = static_cast<size_t>(b) * HKV * s * D;
        o_count = q_count;
        q32.alloc(q_count * 4); k32.alloc(kv_count * 4); v32.alloc(kv_count * 4);
        q16.alloc(q_count * 2); k16.alloc(kv_count * 2); v16.alloc(kv_count * 2);
        o16.alloc(o_count * 2); o32.alloc(o_count * 4);
    }
    void alloc_decode(int b, int n) {
        q_count = static_cast<size_t>(b) * HQ * D;
        kv_count = static_cast<size_t>(b) * HKV * n * D;
        o_count = q_count;
        q32.alloc(q_count * 4); k32.alloc(kv_count * 4); v32.alloc(kv_count * 4);
        q16.alloc(q_count * 2); k16.alloc(kv_count * 2); v16.alloc(kv_count * 2);
        o16.alloc(o_count * 2); o32.alloc(o_count * 4);
    }
    void alloc_materialized(int b, int s, size_t elem_size) {
        score_count = static_cast<size_t>(b) * HQ * s * s;
        score_buf.alloc(score_count * elem_size);
        attn_buf.alloc(score_count * elem_size);
        lrow_buf.alloc(static_cast<size_t>(b) * HQ * s * 4);
    }
    void upload(const std::vector<float>& q, const std::vector<float>& k,
                const std::vector<float>& v) {
        GPU_CHECK(gpuMemcpy(q32.ptr, q.data(), q_count * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(k32.ptr, k.data(), kv_count * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(v32.ptr, v.data(), kv_count * 4, gpuMemcpyHostToDevice));
    }
    void cast_to_f16() {
        auto cast = [](DeviceBuf& s32, DeviceBuf& s16, size_t count) {
            const int blocks = static_cast<int>((count + 1023) / 1024);
            cast_f32_to_f16_kernel<<<blocks, 1024>>>(
                reinterpret_cast<const float*>(s32.ptr),
                reinterpret_cast<ATTN_HALF_TYPE*>(s16.ptr), count);
        };
        cast(q32, q16, q_count);
        cast(k32, k16, kv_count);
        cast(v32, v16, kv_count);
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
        q32.free(); k32.free(); v32.free();
        q16.free(); k16.free(); v16.free();
        o16.free(); o32.free();
        score_buf.free(); attn_buf.free(); lrow_buf.free();
    }
};

int g_warp = 0;   // 设备内置 warpSize（探针），0 = 未探测

template <typename T>
void launch_prefill_t(const GpuData& g, int b, int hq, int hkv, int s,
                      float scale, bool materialized, bool is_half) {
    // fp32 io 路径读 fp32 master 缓冲；fp16 io 读转换缓冲
    const void* qsrc = is_half ? static_cast<const void*>(g.q16.ptr)
                               : static_cast<const void*>(g.q32.ptr);
    const void* ksrc = is_half ? static_cast<const void*>(g.k16.ptr)
                               : static_cast<const void*>(g.k32.ptr);
    const void* vsrc = is_half ? static_cast<const void*>(g.v16.ptr)
                               : static_cast<const void*>(g.v32.ptr);
    void* odst = is_half ? static_cast<void*>(g.o16.ptr) : static_cast<void*>(g.o32.ptr);
    const T* qp = reinterpret_cast<const T*>(qsrc);
    const T* kp = reinterpret_cast<const T*>(ksrc);
    const T* vp = reinterpret_cast<const T*>(vsrc);
    if (materialized) {
        dim3 grid(static_cast<unsigned>(s), static_cast<unsigned>(b * hq));
        atn::mat_scores_kernel<T><<<grid, atn::THREADS>>>(
            qp, kp, reinterpret_cast<T*>(g.score_buf.ptr), hq, hkv, s, scale);
        GPU_CHECK(gpuGetLastError());
        if (g_warp == 128) {
            atn::mat_softmax_kernel<T, 128><<<grid, atn::THREADS>>>(
                reinterpret_cast<const T*>(g.score_buf.ptr),
                reinterpret_cast<T*>(g.attn_buf.ptr),
                reinterpret_cast<float*>(g.lrow_buf.ptr), hq, s);
        } else {
            atn::mat_softmax_kernel<T, 32><<<grid, atn::THREADS>>>(
                reinterpret_cast<const T*>(g.score_buf.ptr),
                reinterpret_cast<T*>(g.attn_buf.ptr),
                reinterpret_cast<float*>(g.lrow_buf.ptr), hq, s);
        }
        GPU_CHECK(gpuGetLastError());
        atn::mat_pv_kernel<T><<<grid, atn::THREADS>>>(
            reinterpret_cast<const T*>(g.attn_buf.ptr),
            reinterpret_cast<const float*>(g.lrow_buf.ptr), vp,
            reinterpret_cast<T*>(odst), hq, hkv, s);
        GPU_CHECK(gpuGetLastError());
        GPU_CHECK(gpuDeviceSynchronize());
        return;
    }
    const unsigned tiles = static_cast<unsigned>((s + atn::BR - 1) / atn::BR);
    atn::gqa_prefill_kernel<T><<<dim3(tiles, static_cast<unsigned>(b * hq)),
                                 atn::THREADS>>>(
        qp, kp, vp, reinterpret_cast<T*>(odst), hq, hkv, s, scale);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
}

void launch_prefill(const GpuData& g, int b, int s, float scale,
                    bool is_half, bool materialized) {
    if (is_half) launch_prefill_t<ATTN_HALF_TYPE>(g, b, HQ, HKV, s, scale, materialized, true);
    else launch_prefill_t<float>(g, b, HQ, HKV, s, scale, materialized, false);
}

template <typename T, int WARP>
void launch_decode_t(const GpuData& g, int b, int hq, int hkv, int n, float scale,
                     bool is_half) {
    const void* qsrc = is_half ? static_cast<const void*>(g.q16.ptr)
                               : static_cast<const void*>(g.q32.ptr);
    const void* ksrc = is_half ? static_cast<const void*>(g.k16.ptr)
                               : static_cast<const void*>(g.k32.ptr);
    const void* vsrc = is_half ? static_cast<const void*>(g.v16.ptr)
                               : static_cast<const void*>(g.v32.ptr);
    void* odst = is_half ? static_cast<void*>(g.o16.ptr) : static_cast<void*>(g.o32.ptr);
    atn::gqa_decode_kernel<T, WARP><<<static_cast<unsigned>(b * hq), atn::THREADS>>>(
        reinterpret_cast<const T*>(qsrc), reinterpret_cast<const T*>(ksrc),
        reinterpret_cast<const T*>(vsrc), reinterpret_cast<T*>(odst),
        hq, hkv, n, scale);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
}

void launch_decode(const GpuData& g, int b, int n, float scale, bool is_half) {
    if (is_half) {
        if (g_warp == 128) launch_decode_t<ATTN_HALF_TYPE, 128>(g, b, HQ, HKV, n, scale, true);
        else launch_decode_t<ATTN_HALF_TYPE, 32>(g, b, HQ, HKV, n, scale, true);
    } else {
        if (g_warp == 128) launch_decode_t<float, 128>(g, b, HQ, HKV, n, scale, false);
        else launch_decode_t<float, 32>(g, b, HQ, HKV, n, scale, false);
    }
}

// ---------------------------------------------------------------------------
// 数据生成（host，fp32 master）
// ---------------------------------------------------------------------------

struct HostData {
    std::vector<float> q, k, v;
    void gen_prefill(int b, int s) {
        const size_t nq = static_cast<size_t>(b) * HQ * s * D;
        const size_t nkv = static_cast<size_t>(b) * HKV * s * D;
        q.resize(nq); k.resize(nkv); v.resize(nkv);
        for (size_t i = 0; i < nq; ++i) q[i] = mu_value(i, SEED_Q);
        for (size_t i = 0; i < nkv; ++i) k[i] = mu_value(i, SEED_K);
        for (size_t i = 0; i < nkv; ++i) v[i] = mu_value(i, SEED_V);
    }
    void gen_decode(int b, int n) {
        const size_t nq = static_cast<size_t>(b) * HQ * D;
        const size_t nkv = static_cast<size_t>(b) * HKV * n * D;
        q.resize(nq); k.resize(nkv); v.resize(nkv);
        for (size_t i = 0; i < nq; ++i) q[i] = mu_value(i, SEED_Q);
        for (size_t i = 0; i < nkv; ++i) k[i] = mu_value(i, SEED_K);
        for (size_t i = 0; i < nkv; ++i) v[i] = mu_value(i, SEED_V);
    }
};

// ---------------------------------------------------------------------------
// 误差统计
// ---------------------------------------------------------------------------

struct ErrStats {
    double max_abs = 0.0, mean_abs = 0.0, rel_denom = 0.0;
    double dot_aa = 0.0, dot_bb = 0.0, dot_ab = 0.0;
    int nan_count = 0;
    double max_abs_pos = 0;
    void accumulate(const float* out, const float* ref, size_t count) {
        double sum_abs = 0.0;
        for (size_t i = 0; i < count; ++i) {
            const double diff = fabs(static_cast<double>(out[i]) - ref[i]);
            if (std::isnan(static_cast<double>(out[i]))) ++nan_count;
            if (diff > max_abs) { max_abs = diff; max_abs_pos = static_cast<double>(i); }
            sum_abs += diff;
            rel_denom += fabs(ref[i]) > 1e-6 ? fabs(ref[i]) : 1e-6;
            dot_aa += static_cast<double>(out[i]) * out[i];
            dot_bb += ref[i] * ref[i];
            dot_ab += static_cast<double>(out[i]) * ref[i];
        }
        mean_abs = sum_abs / static_cast<double>(count);
    }
    double cosine() const { return dot_ab / (sqrt(dot_aa) * sqrt(dot_bb) + 1e-30); }
    double mean_rel() const { return mean_abs / (rel_denom / 1.0); }
};

// ---------------------------------------------------------------------------
// JSON 小工具（手工拼装，固定 schema）
// ---------------------------------------------------------------------------

void json_open(FILE* f, const char* name) { fprintf(f, "\"%s\": ", name); }

// ---------------------------------------------------------------------------
// correctness 矩阵
// ---------------------------------------------------------------------------

int run_correctness(int nthreads) {
    FILE* f = fopen("musa-correctness.json", "w");
    fprintf(f, "{\n  \"schema_version\": \"kernel-chapter21.correctness.musa.v1\",\n");
    fprintf(f, "  \"truth\": \"binary-internal CPU FP32 逐元素循环（hash-RNG seed 0x51002101..03，与 attention_config.py 同算法）\",\n");
    fprintf(f, "  \"warp_device_builtin\": %d,\n", g_warp);
    fprintf(f, "  \"cases\": [\n");
    int failures = 0;
    int written = 0;
    auto emit = [&](const char* form, int b, int seq, const char* dtype,
                    bool is_half, bool materialized) {
        HostData host;
        if (strcmp(form, "prefill") == 0) host.gen_prefill(b, seq);
        else host.gen_decode(b, seq);
        Checksums cs = checksum3(host.q, host.k, host.v);

        GpuData g;
        if (strcmp(form, "prefill") == 0) g.alloc_prefill(b, seq);
        else g.alloc_decode(b, seq);
        if (materialized) g.alloc_materialized(b, seq, is_half ? 2 : 4);
        g.upload(host.q, host.k, host.v);
        g.cast_to_f16();

        // CPU truth
        std::vector<float> truth(g.o_count);
        if (strcmp(form, "prefill") == 0) {
            cpu_prefill_truth(host.q.data(), host.k.data(), host.v.data(),
                              truth.data(), b, HQ, seq, nthreads);
        } else {
            cpu_decode_truth(host.q.data(), host.k.data(), host.v.data(),
                             truth.data(), b, HQ, seq, nthreads);
        }

        // kernel
        if (strcmp(form, "prefill") == 0) {
            launch_prefill(g, b, seq, SCALE, is_half, materialized);
        } else {
            launch_decode(g, b, seq, SCALE, is_half);
        }
        std::vector<float> out(g.o_count);
        g.download_output(is_half, out.data());
        g.free_all();

        ErrStats st;
        st.accumulate(out.data(), truth.data(), g.o_count);
        // 容差分层：fp32 io ≤ 5e-4；fp16 io ≤ 2e-2（输出 O(0.3)）
        const double tol = is_half ? 2e-2 : 5e-4;
        const bool pass = (st.max_abs <= tol) && st.nan_count == 0;
        if (!pass) ++failures;

        fprintf(f, "%s    {\"form\": \"%s%s\", \"batch\": %d, \"seq\": %d, "
                   "\"dtype\": \"%s\", \"max_abs\": %.3e, \"mean_abs\": %.3e, "
                   "\"cosine\": %.8f, \"nan\": %d, \"tolerance\": %.1e, "
                   "\"pass\": %s, \"checksum_q\": %.9f, \"checksum_k\": %.9f, "
                   "\"checksum_v\": %.6f, \"out_absmax\": %.6f}",
                written ? ",\n" : "", form,
                materialized ? "-materialized" : "", b, seq, dtype,
                st.max_abs, st.mean_abs, st.cosine(), st.nan_count, tol,
                pass ? "true" : "false", cs.q, cs.k, cs.v,
                [&]{ double m = 0; for (float x : out) m = fmax(m, fabs(x)); return m; }());
        ++written;
        printf("[correctness/%s] B=%d seq=%d dtype=%s%s: max_abs=%.3e cos=%.7f %s\n",
               form, b, seq, dtype, materialized ? "-mat" : "",
               st.max_abs, st.cosine(), pass ? "PASS" : "FAIL");
        fflush(f);
    };

    // 主矩阵：两形态 × seq {64,512,2048} × batch {1,4} × {fp16, fp32}
    for (int di = 0; di < 2; ++di) {
        const char* dtype = di == 0 ? "fp16" : "fp32";
        const bool is_half = di == 0;
        for (int s : {64, 512, 2048}) {
            for (int b : {1, 4}) emit("prefill", b, s, dtype, is_half, false);
        }
        for (int n : {64, 512, 2048}) {
            for (int b : {1, 4}) emit("decode", b, n, dtype, is_half, false);
        }
    }
    // 边界形状（fp16）：S/N ∈ {1, 63, 65}
    for (int s : {1, 63, 65}) emit("prefill", 1, s, "fp16", true, false);
    for (int n : {1, 63, 65}) emit("decode", 1, n, "fp16", true, false);
    // 物化对照变体（fp16）：S ∈ {512, 2048} × B {1, 4}
    for (int s : {512, 2048}) {
        for (int b : {1, 4}) emit("prefill", b, s, "fp16", true, true);
    }

    fprintf(f, "\n  ],\n  \"failures\": %d\n}\n", failures);
    fclose(f);
    return failures;
}

// ---------------------------------------------------------------------------
// benchmark 矩阵（事件对逐样本、末尾一次同步）
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
                  ms[static_cast<size_t>(fmax(0.0, round(repeats * 0.95) - 1))],
                  sum / repeats, ms[0], warmup, repeats};
}

int run_bench() {
    FILE* f = fopen("musa-benchmark.json", "w");
    fprintf(f, "{\n  \"schema_version\": \"kernel-chapter21.benchmark.musa.v1\",\n");
    fprintf(f, "  \"timing\": \"event-pair per repeat, sync once at end\",\n");
    fprintf(f, "  \"warp_device_builtin\": %d,\n", g_warp);
    fprintf(f, "  \"entries\": [\n");
    int written = 0;
    auto bench = [&](const char* form, int b, int seq, const char* dtype,
                     bool is_half, bool materialized, int warmup, int repeats) {
        HostData host;
        if (strcmp(form, "prefill") == 0) host.gen_prefill(b, seq);
        else host.gen_decode(b, seq);
        GpuData g;
        if (strcmp(form, "prefill") == 0) g.alloc_prefill(b, seq);
        else g.alloc_decode(b, seq);
        if (materialized) g.alloc_materialized(b, seq, is_half ? 2 : 4);
        g.upload(host.q, host.k, host.v);
        g.cast_to_f16();
        Timing t{0, 0, 0, 0, warmup, repeats};
        if (strcmp(form, "prefill") == 0) {
            t = time_it(warmup, repeats, [&] {
                launch_prefill(g, b, seq, SCALE, is_half, materialized);
            });
        } else {
            t = time_it(warmup, repeats, [&] {
                launch_decode(g, b, seq, SCALE, is_half);
            });
        }
        g.free_all();
        fprintf(f, "%s    {\"form\": \"%s%s\", \"batch\": %d, \"seq\": %d, "
                   "\"dtype\": \"%s\", \"p50_ms\": %.4f, \"p95_ms\": %.4f, "
                   "\"mean_ms\": %.4f, \"min_ms\": %.4f, \"warmup\": %d, "
                   "\"repeats\": %d}",
                written ? ",\n" : "", form,
                materialized ? "-materialized" : "", b, seq, dtype,
                t.p50_ms, t.p95_ms, t.mean_ms, t.min_ms, t.warmup, t.repeats);
        ++written;
        printf("[bench/%s] B=%d seq=%d dtype=%s%s: p50=%.4f ms\n",
               form, b, seq, dtype, materialized ? "-mat" : "", t.p50_ms);
        fflush(f);
    };

    for (int di = 0; di < 2; ++di) {
        const char* dtype = di == 0 ? "fp16" : "fp32";
        const bool is_half = di == 0;
        for (int s : {64, 512, 2048, 8192}) {
            const int r = s <= 512 ? 30 : (s <= 2048 ? 15 : 5);
            const int w = s <= 512 ? 3 : (s <= 2048 ? 2 : 1);
            for (int b : {1, 4}) bench("prefill", b, s, dtype, is_half, false, w, r);
        }
        for (int n : {64, 512, 2048, 8192, 32768}) {
            const int r = n <= 2048 ? 30 : 15;
            const int w = n <= 2048 ? 3 : 2;
            for (int b : {1, 4}) bench("decode", b, n, dtype, is_half, false, w, r);
        }
    }
    for (int s : {512, 2048}) {
        for (int b : {1, 4}) bench("prefill", b, s, "fp16", true, true, 2, 10);
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    return 0;
}

// ---------------------------------------------------------------------------
// attrs（设备属性 + warpSize 双口径）
// ---------------------------------------------------------------------------

int run_attrs() {
    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, 0));
    DeviceBuf buf;
    buf.alloc(4);
    atn::warp_size_probe<<<1, 32>>>(
        reinterpret_cast<int*>(buf.ptr));
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    int builtin_warp = 0;
    GPU_CHECK(gpuMemcpy(&builtin_warp, buf.ptr, 4, gpuMemcpyDeviceToHost));
    buf.free();
    g_warp = builtin_warp;

    FILE* f = fopen("musa-env.json", "w");
    fprintf(f, "{\n");
    fprintf(f, "  \"device_name\": \"%s\",\n", prop.name);
    fprintf(f, "  \"cc\": \"%d.%d\",\n", prop.major, prop.minor);
    fprintf(f, "  \"multi_processor_count\": %d,\n", prop.multiProcessorCount);
    fprintf(f, "  \"total_memory_bytes\": %zu,\n", static_cast<size_t>(prop.totalGlobalMem));
    fprintf(f, "  \"warp_size_c_api\": %d,\n", prop.warpSize);
    fprintf(f, "  \"warp_size_device_builtin_probe\": %d,\n", builtin_warp);
    fprintf(f, "  \"shared_mem_per_block\": %zu,\n", static_cast<size_t>(prop.sharedMemPerBlock));
    fprintf(f, "  \"regs_per_block\": %d\n", prop.regsPerBlock);
    fprintf(f, "}\n");
    fclose(f);
    printf("device=%s CC=%d.%d SMs=%d warpC-API=%d warpBuiltin=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.warpSize, builtin_warp);
    return 0;
}

}  // namespace atn21

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "selftest";
    if (atn21::run_attrs() != 0) return 2;
    const int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const int nthreads = std::min(hw_threads, 32);

    if (mode == "attrs") return 0;
    if (mode == "selftest") {
        // 最小形状快速自检（先于全矩阵：失败即停，不浪费服务器时间）
        for (int s : {1, 3, 5}) {
            atn21::HostData host;
            host.gen_prefill(1, s);
            atn21::GpuData g;
            g.alloc_prefill(1, s);
            g.upload(host.q, host.k, host.v);
            g.cast_to_f16();
            std::vector<float> truth(g.o_count);
            atn21::cpu_prefill_truth(host.q.data(), host.k.data(), host.v.data(),
                                     truth.data(), 1, atn21::HQ, s, nthreads);
            atn21::launch_prefill(g, 1, s, atn21::SCALE, true, false);
            std::vector<float> out(g.o_count);
            g.download_output(true, out.data());
            g.free_all();
            atn21::ErrStats st;
            st.accumulate(out.data(), truth.data(), g.o_count);
            const bool ok = st.max_abs <= 2e-2 && st.nan_count == 0;
            printf("[selftest/prefill] S=%d max_abs=%.3e %s\n", s, st.max_abs,
                   ok ? "PASS" : "FAIL");
            if (!ok) return 3;
        }
        for (int n : {1, 3, 5}) {
            atn21::HostData host;
            host.gen_decode(1, n);
            atn21::GpuData g;
            g.alloc_decode(1, n);
            g.upload(host.q, host.k, host.v);
            g.cast_to_f16();
            std::vector<float> truth(g.o_count);
            atn21::cpu_decode_truth(host.q.data(), host.k.data(), host.v.data(),
                                    truth.data(), 1, atn21::HQ, n, nthreads);
            atn21::launch_decode(g, 1, n, atn21::SCALE, true);
            std::vector<float> out(g.o_count);
            g.download_output(true, out.data());
            g.free_all();
            atn21::ErrStats st;
            st.accumulate(out.data(), truth.data(), g.o_count);
            const bool ok = st.max_abs <= 2e-2 && st.nan_count == 0;
            printf("[selftest/decode] N=%d max_abs=%.3e %s\n", n, st.max_abs,
                   ok ? "PASS" : "FAIL");
            if (!ok) return 3;
        }
        printf("[selftest] all PASS\n");
        return 0;
    }
    if (mode == "correctness") return atn21::run_correctness(nthreads);
    if (mode == "bench") return atn21::run_bench();
    fprintf(stderr, "用法: %s {attrs|selftest|correctness|bench}\n", argv[0]);
    return 1;
}
