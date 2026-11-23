// ============================================================================
// attention_kernels.cuh —— 第 21 期 naive Attention 双形态核（可移植主体）。
//
// 被 CUDA（attention_ext.cu，torch 扩展）与 MUSA（attention_bench.mu，
// standalone 二进制）共用；fp16/fp32 io 由模板 T 选择（片上 fp32 累加）。
// 包含方需先提供 <__half> 定义：CUDA 侧本文件自含 include cuda_fp16.h；
// MUSA 侧 .mu 先包含 <musa_fp16.h> 并定义 ATTN_NO_FP16_INCLUDE。
//
// 形态一 gqa_prefill_kernel（tile + online softmax，causal）：
//   grid (ceil(S/BR), B·HQ)，block THREADS=128。
//   每 block 负责 BR=16 个 query；沿 key 按 BC=128 分块推进：
//     score 相：thread j (j<ncols) 串行算 key j 与 16 个 query 的点积 → s_s；
//     per-row online softmax 相：thread r<BR 串行扫 tile（m_new → alpha/l 更新；
//       该行本 tile 无有效 key 则整行跳过——-inf 守卫）；
//     重归一 + PV 相：thread d 拥有维度 d，行优先扫 tile 内 key
//       （acc[r][d] = acc[r][d]·alpha[r] + Σ exp(s_s−m_new)·v[c][d]，v 行合并访存）；
//     causal early-termination：key tile 循环上界 = ceil(min(q0+BR,S)/BC)。
//   不物化 S×S 矩阵：scores/attn 全程在 shared/寄存器（online softmax 的
//   tiling——完整 FlashAttention 的多 block 重缩放不做，归算子篇）。
//
// 形态二 gqa_decode_kernel（单 query，chunked 遍历 + 稳定 softmax）：
//   grid (B·HQ,)，block 128。thread d 拥有输出维度 d；KV 按 chunk=128 分块：
//   score 相（thread j 一行）→ block max / block sum（block_reduce，
//   WARP 模板双实例 32/128——M-11/M-20 warpSize 双口径）→ alpha/l 更新 →
//   PV 相 → 下一 chunk。GQA：kv 头 = h / (HQ/HKV)。
//
// 物化对照变体（量化 online softmax 省下的 HBM 流量）：
//   mat_scores（写 S×S scores）→ mat_softmax（读 scores 写 attn + l）→
//   mat_pv（读 attn/V 写 o）。三 kernel、S×S 两遍满阵读写。
//
// 数值口径：io T（fp16 或 fp32），片上 fp32；expf（非 fast-math 近似）；
// 固定归约顺序、无原子 → 位级确定（同平台同 warpSize 路径）。
// ============================================================================

#ifndef ATTENTION_KERNELS_CUH
#define ATTENTION_KERNELS_CUH

#ifndef ATTN_NO_FP16_INCLUDE
#include <cuda_fp16.h>
#endif

#include <cmath>

namespace atn {

constexpr int D = 128;        // head_dim（Qwen3-0.6B，编译期固定）
constexpr int BR = 16;        // prefill query tile 高
constexpr int BC = 128;       // key tile 宽 = decode chunk = block 线程数
constexpr int THREADS = 128;

// ---- fp16 类型可参数化：默认 __half（cuda_fp16.h / musa_fp16.h 同名）；
//      musa_fp16.h 不可用时 .mu 侧可 -DATTN_HALF_TYPE=... 注入 fallback ----
#ifndef ATTN_HALF_TYPE
#define ATTN_HALF_TYPE __half
#endif

// ---- 可移植 min（不依赖平台 device builtin 名）----
__device__ __forceinline__ int atn_min(int a, int b) { return a < b ? a : b; }

// ---- dtype traits：io T → 片上 float ----
template <typename T> struct IO;
template <> struct IO<float> {
    __device__ static float to(float x) { return x; }
    __device__ static float from(float x) { return x; }
};
template <> struct IO<ATTN_HALF_TYPE> {
    __device__ static float to(ATTN_HALF_TYPE x) { return __half2float(x); }
    __device__ static ATTN_HALF_TYPE from(float x) { return __float2half(x); }
};

// ---- block 归约（warpSize-generic：WARP 编译期双实例 32/128，host 按设备
//      probe 选择；warp 内蝶形 + 跨 warp shared 蝶形，THREADS % WARP == 0）----
template <int WARP, bool IS_MAX>
__device__ __forceinline__ float block_reduce(float val, float* red) {
    const unsigned FULL = 0xffffffffu;
#pragma unroll
    for (int off = WARP >> 1; off > 0; off >>= 1) {
        const float other = __shfl_xor_sync(FULL, val, off);
        val = IS_MAX ? fmaxf(val, other) : val + other;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid % WARP;
    const int wid = tid / WARP;
    constexpr int NW = THREADS / WARP;
    if (NW == 1) return val;
    if (lane == 0) red[wid] = val;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < NW) ? red[lane] : (IS_MAX ? -INFINITY : 0.0f);
#pragma unroll
        for (int off = WARP >> 1; off > 0; off >>= 1) {
            const float other = __shfl_xor_sync(FULL, v, off);
            v = IS_MAX ? fmaxf(v, other) : v + other;
        }
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}

// ---- 设备 warpSize 探针（M-17/M-20 模式：host 据此选 WARP 实例）----
__global__ void warp_size_probe(int* out) {
    if (threadIdx.x == 0) *out = static_cast<int>(warpSize);
}

// ============================================================================
// 形态一：prefill（tile + online softmax，causal）
// ============================================================================

template <typename T>
__global__ void gqa_prefill_kernel(
    const T* __restrict__ q,   // [B, HQ, S, D]
    const T* __restrict__ k,   // [B, HKV, S, D]
    const T* __restrict__ v,   // [B, HKV, S, D]
    T* __restrict__ o,         // [B, HQ, S, D]
    const int hq, const int hkv, const int s, const float scale) {
    const int bh = static_cast<int>(blockIdx.y);   // b*HQ + h
    const int b = bh / hq;
    const int h = bh % hq;
    const int kvh = h / (hq / hkv);
    const int tid = static_cast<int>(threadIdx.x);
    const int q0 = static_cast<int>(blockIdx.x) * BR;
    if (q0 >= s) return;                            // host 保证 grid 不产生空 tile

    const int q_end = atn_min(q0 + BR, s);              // 有效 query 行边界
    const int nr = q_end - q0;
    const int k_tiles = (q_end + BC - 1) / BC;      // causal early-termination

    __shared__ T q_s[BR][D];                        // 4KB(fp16)/8KB(fp32)
    __shared__ float s_s[BR][BC];                   // 8KB
    // 注：无独立 p_s 缓冲——PV 相从 s_s/m_s 重算 exp（同一 expf 输入 → 位级同值，
    // 仅省一次 shared 写读）。原因：MUSA 静态 shared 上限 28672B（mcc 实测报错
    // "shared memory (28884) exceeds limit (28672)"，CUDA 为 48KB）——p_s 的
    // 8KB 恰好把 fp16 实例顶爆（M-43 台账素材）。
    __shared__ float acc[BR][D];                    // 8KB
    __shared__ float m_s[BR];
    __shared__ float l_s[BR];
    __shared__ float alpha_s[BR];

    // 协作加载 q-tile（合并访存）
    const T* q_ptr = q + (static_cast<size_t>(bh) * s + q0) * D;
    for (int i = tid; i < nr * D; i += THREADS) {
        q_s[i / D][i % D] = q_ptr[i];
    }
    for (int r = 0; r < BR; ++r) {
        if (tid == r) { m_s[r] = -INFINITY; l_s[r] = 0.0f; }
        for (int d0 = tid; d0 < D; d0 += THREADS) acc[r][d0] = 0.0f;
    }
    __syncthreads();

    const size_t kv_base = (static_cast<size_t>(b) * hkv + kvh) * s;
    for (int t = 0; t < k_tiles; ++t) {
        const int c0 = t * BC;
        const int cmax = atn_min(c0 + BC, s);
        const int ncols = cmax - c0;

        // ---- score 相：thread j 串行算 key (c0+j) 与 nr 个 query 的点积 ----
        if (tid < ncols) {
            const int j = c0 + tid;
            const T* krow = k + (kv_base + j) * D;
            float dots[BR];
#pragma unroll 4
            for (int r = 0; r < BR; ++r) dots[r] = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                const float kd = IO<T>::to(krow[dd]);
                for (int r = 0; r < nr; ++r) {
                    dots[r] += IO<T>::to(q_s[r][dd]) * kd;
                }
            }
            for (int r = 0; r < nr; ++r) s_s[r][tid] = dots[r] * scale;
        }
        __syncthreads();

        // ---- per-row online softmax 相（thread r 串行扫本 tile）----
        if (tid < BR && tid < nr) {
            const int r = tid;
            const int qpos = q0 + r;
            alpha_s[r] = 1.0f;
            float m_new = -INFINITY;
            for (int c = 0; c < ncols; ++c) {
                if (c0 + c <= qpos) m_new = fmaxf(m_new, s_s[r][c]);
            }
            if (m_new > -INFINITY) {                 // 本 tile 该行有有效 key
                const float a = (m_s[r] == -INFINITY) ? 0.0f
                                                      : expf(m_s[r] - m_new);
                float l_new = 0.0f;
                for (int c = 0; c < ncols; ++c) {
                    if (c0 + c <= qpos) {
                        l_new += expf(s_s[r][c] - m_new);
                    }
                }
                alpha_s[r] = a;
                l_s[r] = l_s[r] * a + l_new;
                m_s[r] = m_new;
            }
            // 无有效 key：m/l 不动、alpha=1；PV 相的索引守卫保证该行不被累加
        }
        __syncthreads();

        // ---- 重归一 + PV 相：thread d 拥有维度 d，行优先扫 key（v 合并访存）----
        // p 从 s_s/m_s 重算（同输入 expf → 位级同值，无 p_s 存储）
        for (int r = 0; r < nr; ++r) {
            acc[r][tid] *= alpha_s[r];
        }
        for (int c = 0; c < ncols; ++c) {
            const int j = c0 + c;
            const float vv = IO<T>::to(v[(kv_base + j) * D + tid]);
            for (int r = 0; r < nr; ++r) {
                if (j <= q0 + r) {                   // causal 索引守卫
                    acc[r][tid] += expf(s_s[r][c] - m_s[r]) * vv;
                }
            }
        }
        __syncthreads();                             // 下一 tile 复用 s_s/p_s
    }

    // ---- 归一化写 o（l>0 由 causal ≥1 key 保证；防御式守卫）----
    T* o_ptr = o + (static_cast<size_t>(bh) * s + q0) * D;
    for (int r = 0; r < nr; ++r) {
        const float inv_l = (l_s[r] > 0.0f) ? 1.0f / l_s[r] : 0.0f;
        o_ptr[r * D + tid] = IO<T>::from(acc[r][tid] * inv_l);
    }
}

// ============================================================================
// 形态二：decode（单 query，chunked 遍历 + 稳定 softmax）
// ============================================================================

template <typename T, int WARP>
__global__ void gqa_decode_kernel(
    const T* __restrict__ q,   // [B, HQ, D]
    const T* __restrict__ k,   // [B, HKV, N, D]
    const T* __restrict__ v,   // [B, HKV, N, D]
    T* __restrict__ o,         // [B, HQ, D]
    const int hq, const int hkv, const int n, const float scale) {
    const int bh = static_cast<int>(blockIdx.x);   // b*HQ + h
    const int b = bh / hq;
    const int h = bh % hq;
    const int kvh = h / (hq / hkv);
    const int tid = static_cast<int>(threadIdx.x);

    __shared__ T q_s[D];
    __shared__ float acc[D];
    __shared__ float sc[BC];
    __shared__ float p[BC];
    // max 与 sum 归约用独立 scratch：两次 block_reduce 背靠背调用时，
    // 前一次的 red[0] 读取发生在其末尾 barrier 之后、无中间 barrier，
    // 共用 buffer 会与后一次的写入竞争（kernel_sim 阶段评审抓出）。
    __shared__ float red_max[THREADS];
    __shared__ float red_sum[THREADS];

    q_s[tid] = q[static_cast<size_t>(bh) * D + tid];
    acc[tid] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    __syncthreads();

    const size_t kv_base = (static_cast<size_t>(b) * hkv + kvh) * n;
    for (int c0 = 0; c0 < n; c0 += BC) {
        const int nv = atn_min(BC, n - c0);

        // ---- score 相：thread j 一行（j < nv）----
        if (tid < nv) {
            const T* krow = k + (kv_base + c0 + tid) * D;
            float dot = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                dot += IO<T>::to(q_s[dd]) * IO<T>::to(krow[dd]);
            }
            sc[tid] = dot * scale;
        }
        __syncthreads();

        // ---- block max + alpha ----
        float val = (tid < nv) ? sc[tid] : -INFINITY;
        const float m_tile = block_reduce<WARP, true>(val, red_max);
        const float m_new = fmaxf(m, m_tile);
        const float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);

        // ---- p + block sum ----
        const float pv = (tid < nv) ? expf(sc[tid] - m_new) : 0.0f;
        if (tid < nv) p[tid] = pv;
        const float p_sum = block_reduce<WARP, false>(pv, red_sum);
        l = l * alpha + p_sum;
        acc[tid] *= alpha;
        __syncthreads();                             // p 写完再被 PV 相读

        // ---- PV 相：thread d 拥有维度 d（v 行合并访存）----
        for (int c = 0; c < nv; ++c) {
            const float vv = IO<T>::to(v[(kv_base + c0 + c) * D + tid]);
            acc[tid] += p[c] * vv;
        }
        m = m_new;
        __syncthreads();                             // 下一 chunk 复用 sc/p
    }

    const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    o[static_cast<size_t>(bh) * D + tid] = IO<T>::from(acc[tid] * inv_l);
}

// ============================================================================
// 物化对照变体（三 kernel，S×S 满阵写读——量化 online softmax 的省流量）
// ============================================================================

// K1：scores[bh, r, j] = (q·k_j)·scale（j ≤ r），否则 0；满阵写出。
template <typename T>
__global__ void mat_scores_kernel(
    const T* __restrict__ q, const T* __restrict__ k,
    T* __restrict__ scores,                        // [B*HQ, S, S]
    const int hq, const int hkv, const int s, const float scale) {
    const int bh = static_cast<int>(blockIdx.y);
    const int b = bh / hq;
    const int h = bh % hq;
    const int kvh = h / (hq / hkv);
    const int r = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const T* qrow = q + (static_cast<size_t>(bh) * s + r) * D;
    const size_t kbase = (static_cast<size_t>(b) * hkv + kvh) * s;
    T* out = scores + (static_cast<size_t>(bh) * s + r) * s;
    for (int j = tid; j < s; j += THREADS) {
        if (j <= r) {
            const T* krow = k + (kbase + j) * D;
            float dot = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                dot += IO<T>::to(qrow[dd]) * IO<T>::to(krow[dd]);
            }
            out[j] = IO<T>::from(dot * scale);
        } else {
            out[j] = IO<T>::from(0.0f);
        }
    }
}

// K2：行 softmax（j ≤ r 掩码）→ attn + 每行 l（fp32）。
template <typename T, int WARP>
__global__ void mat_softmax_kernel(
    const T* __restrict__ scores, T* __restrict__ attn,
    float* __restrict__ lrow,                      // [B*HQ, S]
    const int hq, const int s) {
    (void)hq;
    const int bh = static_cast<int>(blockIdx.y);
    const int r = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float red_max[THREADS];
    __shared__ float red_sum[THREADS];
    const size_t base = (static_cast<size_t>(bh) * s + r) * s;

    float local = -INFINITY;
    for (int j = tid; j <= r; j += THREADS) {
        local = fmaxf(local, IO<T>::to(scores[base + j]));
    }
    const float m = block_reduce<WARP, true>(local, red_max);

    float lsum = 0.0f;
    for (int j = tid; j < s; j += THREADS) {
        const float p = (j <= r) ? expf(IO<T>::to(scores[base + j]) - m) : 0.0f;
        attn[base + j] = IO<T>::from(p);
        lsum += p;
    }
    const float l = block_reduce<WARP, false>(lsum, red_sum);
    if (tid == 0) lrow[static_cast<size_t>(bh) * s + r] = l;
}

// K3：PV + 归一化：thread d 拥有维度 d。
template <typename T>
__global__ void mat_pv_kernel(
    const T* __restrict__ attn, const float* __restrict__ lrow,
    const T* __restrict__ v, T* __restrict__ o,
    const int hq, const int hkv, const int s) {
    const int bh = static_cast<int>(blockIdx.y);
    const int b = bh / hq;
    const int h = bh % hq;
    const int kvh = h / (hq / hkv);
    const int r = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const float l = lrow[static_cast<size_t>(bh) * s + r];
    const size_t abase = (static_cast<size_t>(bh) * s + r) * s;
    const size_t vbase = (static_cast<size_t>(b) * hkv + kvh) * s;
    float accv = 0.0f;
    for (int j = 0; j <= r; ++j) {
        const float p = IO<T>::to(attn[abase + j]);
        const float vv = IO<T>::to(v[(vbase + j) * D + tid]);
        accv += p * vv;
    }
    const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    o[(static_cast<size_t>(bh) * s + r) * D + tid] = IO<T>::from(accv * inv_l);
}

}  // namespace atn

#endif  // ATTENTION_KERNELS_CUH
