// ============================================================================
// rope_kernels.cuh —— 第 23 期 RoPE 双形态核（可移植主体，CUDA/MUSA 共用）。
//
// 包含方需先提供平台运行时与 <__half> 定义并定义 gpu* shim：
//   CUDA 侧 rope_bench.cu：#include <cuda_runtime.h> + <cuda_fp16.h>；
//   MUSA 侧 rope_bench.mu：#define USE_MUSA 1 + <musa_runtime.h> + musa_fp16.h
//   （ATTN_NO_FP16_INCLUDE 风格防御开关：RPE_NO_FP16_INCLUDE）。
//
// 语义（与 implementations/cpu/rope_reference.py 严格一致，RoPE-NeXt
// half-split，HF rotate_half 等价）：
//   inv_freq[j] = 1 / theta^(2j/D)      j ∈ [0, HALF)
//   pair (x[j], x[j+HALF])，角度 = pos · inv_freq[j]：
//     out[j]       = x[j]·c − x[j+HALF]·s
//     out[j+HALF]  = x[j+HALF]·c + x[j]·s
//
// 三份核：
//   1. build_cache_kernel<CT>   cos/sin compact cache [max_seq, HALF] 生成
//      （每线程一个角度 sincosf；CT = float / __half 存储双实例）。
//   2. rope_prefill_kernel<T,CT> prefill 批量形态：并行维 =
//      num_tokens × (hq+hkv) × HALF，tid 除法映射 (t, h, j)，查表一行。
//   3. rope_decode_kernel<T,CT>  decode 单 Token 形态：grid = (hq+hkv,)
//      block = HALF=64；q 头走 blockIdx.x < hq 分支，k 头走 k 指针——
//      双形态的可见差异：并行维从 tokens 维塌缩为头维。
//
// 数值口径：io T（float/half）读入提升 fp32；角度来自 cache 表（fp32 存储
// 或 fp16 存储 → __half2float）；片上 fp32 算术；输出按 io dtype 舍入。
// 无归约 / 无 shared / 无原子 → 位级确定。D=128 编译期固定（Qwen3 口径）。
// M-43 约束：本核零静态 shared，天然低于 28,672B 上限。
// ============================================================================

#ifndef ROPE_KERNELS_CUH
#define ROPE_KERNELS_CUH

#ifndef RPE_NO_FP16_INCLUDE
#include <cuda_fp16.h>
#endif

#include <cmath>

namespace rpe {

constexpr int HD = 128;        // head_dim（Qwen3-0.6B，编译期固定）
constexpr int HALF = HD / 2;   // 64：pair 数 = compact cache 第二维
constexpr int THREADS = 128;   // prefill / build_cache block 线程数
constexpr int DECODE_THREADS = HALF;  // decode block = 64 线程（每线程 1 pair）

// ---- fp16 类型可参数化（musa_fp16.h / cuda_fp16.h 同名 __half）----
#ifndef RPE_HALF_TYPE
#define RPE_HALF_TYPE __half
#endif

// ---- 可移植 min ----
__device__ __forceinline__ int rpe_min(int a, int b) { return a < b ? a : b; }

// ---- sincosf 缺失时的回退开关（build.sh 回退链第 2 档；以实机为准）----
#ifdef RPE_NO_SINCOS
__device__ __forceinline__ void rpe_sincosf(float x, float* s, float* c) {
    *s = sinf(x);
    *c = cosf(x);
}
#else
__device__ __forceinline__ void rpe_sincosf(float x, float* s, float* c) {
    sincosf(x, s, c);
}
#endif

// ---- dtype traits：io T / cache CT → 片上 float ----
template <typename X> struct RToFloat;
template <> struct RToFloat<float> {
    __device__ static float to(float x) { return x; }
};
template <> struct RToFloat<RPE_HALF_TYPE> {
    __device__ static float to(RPE_HALF_TYPE x) { return __half2float(x); }
};
template <typename X> struct RFromFloat;
template <> struct RFromFloat<float> {
    __device__ static float from(float x) { return x; }
};
template <> struct RFromFloat<RPE_HALF_TYPE> {
    __device__ static RPE_HALF_TYPE from(float x) { return __float2half(x); }
};

// ---- 设备 warpSize 探针（M-11/M-20 双口径惯例，attrs 模式用）----
__global__ void warp_size_probe(int* out) {
    if (threadIdx.x == 0) *out = static_cast<int>(warpSize);
}

// ============================================================================
// 1. cache 生成：cos/sin compact [max_seq, HALF]，每线程一个角度
//    inv_freq 镜像 HF 公式：1 / theta^(2j/HD)（先幂后倒数，与
//    rope_reference.build_cache / chapter_01 RotaryEmbedding 同序）
// ============================================================================

template <typename CT>
__global__ void build_cache_kernel(
    CT* __restrict__ cos_cache, CT* __restrict__ sin_cache,
    const int max_seq, const float theta) {
    const size_t total = static_cast<size_t>(max_seq) * HALF;
    const size_t idx = static_cast<size_t>(blockIdx.x) * THREADS + threadIdx.x;
    if (idx >= total) return;
    const int pos = static_cast<int>(idx / HALF);
    const int j = static_cast<int>(idx % HALF);
    const float exponent = (2.0f * static_cast<float>(j)) / static_cast<float>(HD);
    const float inv_freq = 1.0f / powf(theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    float c, s;
    rpe_sincosf(angle, &s, &c);
    cos_cache[idx] = RFromFloat<CT>::from(c);
    sin_cache[idx] = RFromFloat<CT>::from(s);
}

// ============================================================================
// 2. prefill 批量形态：q [T, HQ, HD]，k [T, HKV, HD]（token 主序连续）
//    grid = ceil(T·(hq+hkv)·HALF / THREADS)，block = THREADS
// ============================================================================

template <typename T, typename CT>
__global__ void rope_prefill_kernel(
    const T* __restrict__ q, const T* __restrict__ k,
    T* __restrict__ q_out, T* __restrict__ k_out,
    const CT* __restrict__ cos_cache, const CT* __restrict__ sin_cache,
    const int num_tokens, const int hq, const int hkv, const int seq_offset) {
    const int heads = hq + hkv;
    const size_t total = static_cast<size_t>(num_tokens) * heads * HALF;
    const size_t idx = static_cast<size_t>(blockIdx.x) * THREADS + threadIdx.x;
    if (idx >= total) return;

    const int j = static_cast<int>(idx % HALF);
    const int h = static_cast<int>((idx / HALF) % heads);
    const int t = static_cast<int>(idx / (static_cast<size_t>(HALF) * heads));
    const int row = seq_offset + t;
    const float c = RToFloat<CT>::to(cos_cache[static_cast<size_t>(row) * HALF + j]);
    const float s = RToFloat<CT>::to(sin_cache[static_cast<size_t>(row) * HALF + j]);

    const bool is_q = h < hq;
    const int hh = is_q ? h : h - hq;
    const T* src = (is_q ? q + (static_cast<size_t>(t) * hq + hh) * HD
                         : k + (static_cast<size_t>(t) * hkv + hh) * HD);
    T* dst = (is_q ? q_out + (static_cast<size_t>(t) * hq + hh) * HD
                   : k_out + (static_cast<size_t>(t) * hkv + hh) * HD);
    const float x1 = RToFloat<T>::to(src[j]);
    const float x2 = RToFloat<T>::to(src[j + HALF]);
    dst[j] = RFromFloat<T>::from(x1 * c - x2 * s);
    dst[j + HALF] = RFromFloat<T>::from(x2 * c + x1 * s);
}

// ============================================================================
// 3. decode 单 Token 形态：q [HQ, HD]，k [HKV, HD]，位置 = seq_offset
//    grid = (hq+hkv,)，block = HALF=64（每 block 一个头，每线程一个 pair）
//    —— 与 prefill 的可见差异：token 维消失，grid 从 T·(hq+hkv) 塌缩为
//    (hq+hkv)，总线程 24×64 = 1536（prefill T=2048 时为 2048×24×64/128 的
//    数千 block）——launch-bound 的结构性来源。
// ============================================================================

template <typename T, typename CT>
__global__ void rope_decode_kernel(
    const T* __restrict__ q, const T* __restrict__ k,
    T* __restrict__ q_out, T* __restrict__ k_out,
    const CT* __restrict__ cos_cache, const CT* __restrict__ sin_cache,
    const int hq, const int hkv, const int seq_offset) {
    const int h = static_cast<int>(blockIdx.x);
    const int j = static_cast<int>(threadIdx.x);
    if (j >= HALF) return;
    const float c = RToFloat<CT>::to(cos_cache[static_cast<size_t>(seq_offset) * HALF + j]);
    const float s = RToFloat<CT>::to(sin_cache[static_cast<size_t>(seq_offset) * HALF + j]);

    if (h < hq) {
        const T* src = q + static_cast<size_t>(h) * HD;
        T* dst = q_out + static_cast<size_t>(h) * HD;
        const float x1 = RToFloat<T>::to(src[j]);
        const float x2 = RToFloat<T>::to(src[j + HALF]);
        dst[j] = RFromFloat<T>::from(x1 * c - x2 * s);
        dst[j + HALF] = RFromFloat<T>::from(x2 * c + x1 * s);
    } else {
        const int hh = h - hq;
        const T* src = k + static_cast<size_t>(hh) * HD;
        T* dst = k_out + static_cast<size_t>(hh) * HD;
        const float x1 = RToFloat<T>::to(src[j]);
        const float x2 = RToFloat<T>::to(src[j + HALF]);
        dst[j] = RFromFloat<T>::from(x1 * c - x2 * s);
        dst[j + HALF] = RFromFloat<T>::from(x2 * c + x1 * s);
    }
}

// ============================================================================
// fp32 ↔ fp16 转换（io fp16 路径的缓冲准备/回读，ch21 模式）
// ============================================================================

__global__ void cast_f32_to_f16_kernel(const float* src, RPE_HALF_TYPE* dst,
                                       size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2half(src[i]);
}

__global__ void cast_f16_to_f32_kernel(const RPE_HALF_TYPE* src, float* dst,
                                       size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __half2float(src[i]);
}

}  // namespace rpe

#endif  // ROPE_KERNELS_CUH
