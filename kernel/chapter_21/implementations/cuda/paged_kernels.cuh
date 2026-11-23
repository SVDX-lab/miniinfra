// ============================================================================
// paged_kernels.cuh —— 第 22 期分页 KV 核（可移植主体，CUDA/MUSA 共用）。
//
// 被共享方先定义（与 ch21 同约定）：
//   CUDA 侧本文件自含 <cuda_fp16.h>；MUSA 侧 .mu 先包含 <musa_fp16.h> 并
//   #define ATTN_NO_FP16_INCLUDE（ch21/M-43 模式）。
//
// 块表语义合同（paged_config.py 模块头为单源；此处为实现侧引用）：
//   pool:  k/v_pool [num_blocks, HKV, BS, D]——物理块 p 的 (kv头 g, 块内偏移 o)
//          行 = pool[p, g, o, :]，D 维连续；
//   token t → tb = t / BS; ot = t % BS; pblk = block_table[b * max_blocks + tb]；
//   block_table: [B, max_blocks] int32，有效区间 [0, ceil(seq_len_b/BS))，
//          padding 表项合同值 -1，kernel 不读取（host 侧 validate_tables 预检）；
//   seq_lens: [B] int32，decode 每 query 只 attends [0, seq_len_b)。
//
// 四个核（2×2：{decode, prefill} × {paged, contig-控制}）+ gather-only 核：
//   paged_decode_kernel<T,WARP>   decode 单 query + 块表寻址（本期主核）
//   contig_decode_kernel<T,WARP>  同核去块表化（控制组；除地址计算外逐行相同）
//   paged_prefill_kernel<T>       ch21 prefill 结构（tile + online softmax）
//                                 + K/V 行块表寻址（可选对照项）
//   contig_prefill_kernel<T>      ch21 gqa_prefill_kernel 原样（prefill 控制组）
//   gather_kv_kernel<T>           纯 gather（K+V 一遍：池→连续；归因分层用：
//                                 单独测"块表查找+取数"的访存代价）
//
// 归因分层设计（任务书：块表查找 vs gather 本身 vs 占用率）：
//   contig 核（无表）→ paged 核+恒等表（+表查找/整除取模）→ paged 核+随机表
//   （+局部性损失）→ B 扫描（占用率）。三者输出位级一致（值透明）。
//
// 数值口径：io T（fp16/fp32），片上 fp32；expf；固定归约顺序、无原子 →
//   位级确定。无动态 smem（静态预算 ≤ 24.2KB fp32 / 20.2KB fp16 —— MUSA
//   静态 smem 上限 28,672B 之内，M-43 约束）。
// ============================================================================

#ifndef PAGED_KERNELS_CUH
#define PAGED_KERNELS_CUH

#ifndef ATTN_NO_FP16_INCLUDE
#include <cuda_fp16.h>
#endif

#include <cmath>

namespace pkg {

constexpr int D = 128;        // head_dim（Qwen3-0.6B，编译期固定）
constexpr int BC = 128;       // decode chunk = key tile 宽 = block 线程数
constexpr int BR = 16;        // prefill query tile 高（ch21 同参数）
constexpr int THREADS = 128;

#ifndef ATTN_HALF_TYPE
#define ATTN_HALF_TYPE __half
#endif

__device__ __forceinline__ int pkg_min(int a, int b) { return a < b ? a : b; }

// ---- dtype traits：io T → 片上 float（ch21 同式）----
template <typename T> struct IO;
template <> struct IO<float> {
    __device__ static float to(float x) { return x; }
    __device__ static float from(float x) { return x; }
};
template <> struct IO<ATTN_HALF_TYPE> {
    __device__ static float to(ATTN_HALF_TYPE x) { return __half2float(x); }
    __device__ static ATTN_HALF_TYPE from(float x) { return __float2half(x); }
};

// ---- block 归约（warpSize-generic 双实例 32/128 + host 探针选择，M-11/M-20）----
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

__global__ void warp_size_probe(int* out) {
    if (threadIdx.x == 0) *out = static_cast<int>(warpSize);
}

// ---- 池行地址（合同映射；所有核共用同一算式——位级透明的地址层）----
__device__ __forceinline__ size_t pool_row_offset(int hkv, int kvh, int bs,
                                                  int pblk, int ot) {
    return (static_cast<size_t>(pblk) * hkv + kvh) * bs * D
           + static_cast<size_t>(ot) * D;
}

// ============================================================================
// 核 1：paged decode（单 query + 块表寻址）
// ============================================================================

template <typename T, int WARP>
__global__ void paged_decode_kernel(
    const T* __restrict__ q,               // [B, HQ, D]
    const T* __restrict__ k_pool,          // [P, HKV, BS, D]
    const T* __restrict__ v_pool,          // [P, HKV, BS, D]
    const int* __restrict__ block_table,   // [B, max_blocks]
    const int* __restrict__ seq_lens,      // [B]
    T* __restrict__ o,                     // [B, HQ, D]
    const int hq, const int hkv, const int bs, const int max_blocks,
    const float scale) {
    const int bh = static_cast<int>(blockIdx.x);   // b*HQ + h
    const int b = bh / hq;
    const int h = bh % hq;
    const int kvh = h / (hq / hkv);
    const int tid = static_cast<int>(threadIdx.x);
    const int n = seq_lens[b];                     // 每请求独立长度（引擎口径）

    __shared__ T q_s[D];
    __shared__ float acc[D];
    __shared__ float sc[BC];
    __shared__ float p[BC];
    // 两次背靠背 block_reduce 用独立 scratch（ch21 kernel_sim 评审教训）
    __shared__ float red_max[THREADS];
    __shared__ float red_sum[THREADS];

    q_s[tid] = q[static_cast<size_t>(bh) * D + tid];
    acc[tid] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    __syncthreads();

    const int* table_row = block_table + static_cast<size_t>(b) * max_blocks;
    for (int c0 = 0; c0 < n; c0 += BC) {
        const int nv = pkg_min(BC, n - c0);

        // ---- score 相：thread j 一行（表查找 → 池行）----
        float val = -INFINITY;
        if (tid < nv) {
            const int tok = c0 + tid;
            const int pblk = table_row[tok / bs];
            const int ot = tok % bs;
            const T* krow = k_pool + pool_row_offset(hkv, kvh, bs, pblk, ot);
            float dot = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                dot += IO<T>::to(q_s[dd]) * IO<T>::to(krow[dd]);
            }
            val = dot * scale;
            sc[tid] = val;
        }
        __syncthreads();

        // ---- block max + alpha ----
        if (tid >= nv) val = -INFINITY;
        const float m_tile = block_reduce<WARP, true>(val, red_max);
        const float m_new = fmaxf(m, m_tile);
        const float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);

        // ---- p + block sum ----
        const float pv = (tid < nv) ? expf(sc[tid] - m_new) : 0.0f;
        if (tid < nv) p[tid] = pv;
        const float p_sum = block_reduce<WARP, false>(pv, red_sum);
        l = l * alpha + p_sum;
        acc[tid] *= alpha;
        __syncthreads();                           // p 写完再被 PV 相读

        // ---- PV 相：thread d 拥有维度 d（逐 token 表查找）----
        for (int c = 0; c < nv; ++c) {
            const int tok = c0 + c;
            const int pblk = table_row[tok / bs];
            const int ot = tok % bs;
            const T* vrow = v_pool + pool_row_offset(hkv, kvh, bs, pblk, ot);
            acc[tid] += p[c] * IO<T>::to(vrow[tid]);
        }
        m = m_new;
        __syncthreads();                           // 下一 chunk 复用 sc/p
    }

    const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    o[static_cast<size_t>(bh) * D + tid] = IO<T>::from(acc[tid] * inv_l);
}

// ============================================================================
// 核 2：contig decode（控制组：同核去块表化——除地址计算外逐行相同）
// ============================================================================

template <typename T, int WARP>
__global__ void contig_decode_kernel(
    const T* __restrict__ q,               // [B, HQ, D]
    const T* __restrict__ k_cache,         // [B, HKV, Nmax, D]
    const T* __restrict__ v_cache,
    const int* __restrict__ seq_lens,      // [B]
    T* __restrict__ o,                     // [B, HQ, D]
    const int hq, const int hkv, const int n_max, const float scale) {
    const int bh = static_cast<int>(blockIdx.x);
    const int b = bh / hq;
    const int h = bh % hq;
    const int kvh = h / (hq / hkv);
    const int tid = static_cast<int>(threadIdx.x);
    const int n = seq_lens[b];

    __shared__ T q_s[D];
    __shared__ float acc[D];
    __shared__ float sc[BC];
    __shared__ float p[BC];
    __shared__ float red_max[THREADS];
    __shared__ float red_sum[THREADS];

    q_s[tid] = q[static_cast<size_t>(bh) * D + tid];
    acc[tid] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    __syncthreads();

    const size_t kv_base = (static_cast<size_t>(b) * hkv + kvh) * n_max;
    for (int c0 = 0; c0 < n; c0 += BC) {
        const int nv = pkg_min(BC, n - c0);

        float val = -INFINITY;
        if (tid < nv) {
            const T* krow = k_cache + (kv_base + c0 + tid) * D;
            float dot = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                dot += IO<T>::to(q_s[dd]) * IO<T>::to(krow[dd]);
            }
            val = dot * scale;
            sc[tid] = val;
        }
        __syncthreads();

        if (tid >= nv) val = -INFINITY;
        const float m_tile = block_reduce<WARP, true>(val, red_max);
        const float m_new = fmaxf(m, m_tile);
        const float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);

        const float pv = (tid < nv) ? expf(sc[tid] - m_new) : 0.0f;
        if (tid < nv) p[tid] = pv;
        const float p_sum = block_reduce<WARP, false>(pv, red_sum);
        l = l * alpha + p_sum;
        acc[tid] *= alpha;
        __syncthreads();

        for (int c = 0; c < nv; ++c) {
            const float vv = IO<T>::to(v_cache[(kv_base + c0 + c) * D + tid]);
            acc[tid] += p[c] * vv;
        }
        m = m_new;
        __syncthreads();
    }

    const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    o[static_cast<size_t>(bh) * D + tid] = IO<T>::from(acc[tid] * inv_l);
}

// ============================================================================
// 核 3：paged prefill（ch21 prefill 结构 + K/V 行块表寻址；可选对照项）
// ============================================================================

template <typename T>
__global__ void paged_prefill_kernel(
    const T* __restrict__ q,               // [B, HQ, S, D]
    const T* __restrict__ k_pool,          // [P, HKV, BS, D]
    const T* __restrict__ v_pool,
    const int* __restrict__ block_table,   // [B, max_blocks]
    T* __restrict__ o,                     // [B, HQ, S, D]
    const int hq, const int hkv, const int s, const int bs,
    const int max_blocks, const float scale) {
    const int bh = static_cast<int>(blockIdx.y);
    const int b = bh / hq;
    const int h = bh % hq;
    const int kvh = h / (hq / hkv);
    const int tid = static_cast<int>(threadIdx.x);
    const int q0 = static_cast<int>(blockIdx.x) * BR;
    if (q0 >= s) return;

    const int q_end = pkg_min(q0 + BR, s);
    const int nr = q_end - q0;
    const int k_tiles = (q_end + BC - 1) / BC;      // causal early-termination

    __shared__ T q_s[BR][D];                        // 4KB(fp16)/8KB(fp32)
    __shared__ float s_s[BR][BC];                   // 8KB
    // 无独立 p 缓冲：PV 相从 s_s/m_s 重算 exp（M-43 修复模式，省 8KB）
    __shared__ float acc[BR][D];                    // 8KB
    __shared__ float m_s[BR];
    __shared__ float l_s[BR];
    __shared__ float alpha_s[BR];

    const T* q_ptr = q + (static_cast<size_t>(bh) * s + q0) * D;
    for (int i = tid; i < nr * D; i += THREADS) {
        q_s[i / D][i % D] = q_ptr[i];
    }
    for (int r = 0; r < BR; ++r) {
        if (tid == r) { m_s[r] = -INFINITY; l_s[r] = 0.0f; }
        for (int d0 = tid; d0 < D; d0 += THREADS) acc[r][d0] = 0.0f;
    }
    __syncthreads();

    const int* table_row = block_table + static_cast<size_t>(b) * max_blocks;
    for (int t = 0; t < k_tiles; ++t) {
        const int c0 = t * BC;
        const int cmax = pkg_min(c0 + BC, s);
        const int ncols = cmax - c0;

        // ---- score 相：thread j 算 key (c0+j)（表查找 → 池行）----
        if (tid < ncols) {
            const int tok = c0 + tid;
            const int pblk = table_row[tok / bs];
            const int ot = tok % bs;
            const T* krow = k_pool + pool_row_offset(hkv, kvh, bs, pblk, ot);
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

        // ---- per-row online softmax（ch21 同式；-inf 守卫）----
        if (tid < BR && tid < nr) {
            const int r = tid;
            const int qpos = q0 + r;
            alpha_s[r] = 1.0f;
            float m_new = -INFINITY;
            for (int c = 0; c < ncols; ++c) {
                if (c0 + c <= qpos) m_new = fmaxf(m_new, s_s[r][c]);
            }
            if (m_new > -INFINITY) {
                const float a = (m_s[r] == -INFINITY) ? 0.0f : expf(m_s[r] - m_new);
                float l_new = 0.0f;
                for (int c = 0; c < ncols; ++c) {
                    if (c0 + c <= qpos) l_new += expf(s_s[r][c] - m_new);
                }
                alpha_s[r] = a;
                l_s[r] = l_s[r] * a + l_new;
                m_s[r] = m_new;
            }
        }
        __syncthreads();

        // ---- 重归一 + PV 相（逐 token 表查找）----
        for (int r = 0; r < nr; ++r) acc[r][tid] *= alpha_s[r];
        for (int c = 0; c < ncols; ++c) {
            const int tok = c0 + c;
            const int pblk = table_row[tok / bs];
            const int ot = tok % bs;
            const T* vrow = v_pool + pool_row_offset(hkv, kvh, bs, pblk, ot);
            const float vv = IO<T>::to(vrow[tid]);
            for (int r = 0; r < nr; ++r) {
                if (tok <= q0 + r) acc[r][tid] += expf(s_s[r][c] - m_s[r]) * vv;
            }
        }
        __syncthreads();
    }

    T* o_ptr = o + (static_cast<size_t>(bh) * s + q0) * D;
    for (int r = 0; r < nr; ++r) {
        const float inv_l = (l_s[r] > 0.0f) ? 1.0f / l_s[r] : 0.0f;
        o_ptr[r * D + tid] = IO<T>::from(acc[r][tid] * inv_l);
    }
}

// ============================================================================
// 核 4：contig prefill（控制组 = ch21 gqa_prefill_kernel 原样）
// ============================================================================

template <typename T>
__global__ void contig_prefill_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,               // [B, HKV, S, D]
    const T* __restrict__ v,
    T* __restrict__ o,
    const int hq, const int hkv, const int s, const float scale) {
    const int bh = static_cast<int>(blockIdx.y);
    const int b = bh / hq;
    const int h = bh % hq;
    const int kvh = h / (hq / hkv);
    const int tid = static_cast<int>(threadIdx.x);
    const int q0 = static_cast<int>(blockIdx.x) * BR;
    if (q0 >= s) return;

    const int q_end = pkg_min(q0 + BR, s);
    const int nr = q_end - q0;
    const int k_tiles = (q_end + BC - 1) / BC;

    __shared__ T q_s[BR][D];
    __shared__ float s_s[BR][BC];
    __shared__ float acc[BR][D];
    __shared__ float m_s[BR];
    __shared__ float l_s[BR];
    __shared__ float alpha_s[BR];

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
        const int cmax = pkg_min(c0 + BC, s);
        const int ncols = cmax - c0;

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

        if (tid < BR && tid < nr) {
            const int r = tid;
            const int qpos = q0 + r;
            alpha_s[r] = 1.0f;
            float m_new = -INFINITY;
            for (int c = 0; c < ncols; ++c) {
                if (c0 + c <= qpos) m_new = fmaxf(m_new, s_s[r][c]);
            }
            if (m_new > -INFINITY) {
                const float a = (m_s[r] == -INFINITY) ? 0.0f : expf(m_s[r] - m_new);
                float l_new = 0.0f;
                for (int c = 0; c < ncols; ++c) {
                    if (c0 + c <= qpos) l_new += expf(s_s[r][c] - m_new);
                }
                alpha_s[r] = a;
                l_s[r] = l_s[r] * a + l_new;
                m_s[r] = m_new;
            }
        }
        __syncthreads();

        for (int r = 0; r < nr; ++r) acc[r][tid] *= alpha_s[r];
        for (int c = 0; c < ncols; ++c) {
            const int j = c0 + c;
            const float vv = IO<T>::to(v[(kv_base + j) * D + tid]);
            for (int r = 0; r < nr; ++r) {
                if (j <= q0 + r) acc[r][tid] += expf(s_s[r][c] - m_s[r]) * vv;
            }
        }
        __syncthreads();
    }

    T* o_ptr = o + (static_cast<size_t>(bh) * s + q0) * D;
    for (int r = 0; r < nr; ++r) {
        const float inv_l = (l_s[r] > 0.0f) ? 1.0f / l_s[r] : 0.0f;
        o_ptr[r * D + tid] = IO<T>::from(acc[r][tid] * inv_l);
    }
}

// ============================================================================
// 核 5：gather-only（池 → 连续布局；K+V 一遍；归因分层用）
//    线程 i 处理一个 (kv_sel, b, kvh, tok, d) 元素；total = 2·B·HKV·Nmax·D
// ============================================================================

template <typename T>
__global__ void gather_kv_kernel(
    const T* __restrict__ k_pool,          // [P, HKV, BS, D]
    const T* __restrict__ v_pool,
    const int* __restrict__ block_table,   // [B, max_blocks]
    const int* __restrict__ seq_lens,      // [B]
    T* __restrict__ k_out,                 // [B, HKV, Nmax, D]
    T* __restrict__ v_out,
    const int bsz, const int hkv, const int bs, const int max_blocks,
    const int n_max, const long long total) {
    const long long i = static_cast<long long>(blockIdx.x) * THREADS
                        + static_cast<int>(threadIdx.x);
    if (i >= total) return;
    const int d = static_cast<int>(i % D);
    long long r = i / D;
    const int tok = static_cast<int>(r % n_max);
    r /= n_max;
    const int kvh = static_cast<int>(r % hkv);
    r /= hkv;
    const int b = static_cast<int>(r % bsz);
    const int kv_sel = static_cast<int>(r / bsz);
    const size_t oidx = (((static_cast<size_t>(b) * hkv + kvh) * n_max) + tok) * D + d;
    if (tok >= seq_lens[b]) {
        (kv_sel == 0 ? k_out : v_out)[oidx] = IO<T>::from(0.0f);  // 无效区写 0
        return;
    }
    const int pblk = block_table[static_cast<size_t>(b) * max_blocks + tok / bs];
    const int ot = tok % bs;
    const T* pool = (kv_sel == 0) ? k_pool : v_pool;
    const size_t sidx = ((static_cast<size_t>(pblk) * hkv + kvh) * bs + ot) * D + d;
    (kv_sel == 0 ? k_out : v_out)[oidx] = pool[sidx];
}

}  // namespace pkg

#endif  // PAGED_KERNELS_CUH
