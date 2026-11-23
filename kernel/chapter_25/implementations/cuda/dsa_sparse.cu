// ============================================================================
// dsa_sparse.cu —— 第 25 期 DSA naive 三段核（Lightning Indexer + 阈值 top-k +
// gather Attention）。骨架沿用第 24 期 MLA 吸收核（split-K flash-decoding +
// 无冲突 acc 布局），在两处改结构：
//   (a) split 核从"连续 token 区间"改为"索引列表区间"——token = indices[i]
//       （gather 模式）；indices==nullptr 时 token=i（identity 模式 = dense
//       对照，同一份代码两条路径，直接支撑 §9 "给定索引集 vs 子集 dense"）；
//   (b) 新增索引器 score 核与阈值 top-k 核组（count + 前缀 + 确定性 compact）。
//
// 输入（fp16 连续 CUDA device；w_used fp32）：
//   q_i   [64,128]  索引 Q（torch 侧 q_a @ W_QI 切分）
//   w_used[64]      索引头权重（已乘 n_heads^-0.5）
//   k_i   [N,128]   索引 K 缓存（torch 侧 c_kv @ W_KL；官方为独立 K cache）
//   q_abs [H,512]   吸收后 latent query；q_rope [H,64]（已旋转）
//   c_kv  [N,512]   latent cache；k_rope [N,64]
//   scores[N] fp32  索引器输出 / topk 输入
// 输出：
//   scores [N] fp32；indices [min(k,N)] int32（升序集合，见下"写序声明"）；
//   s [H,512] fp16（加权 latent，ep25 同口径）
//
// top-k 算法（任务书指定 naive 阈值法）：
//   fp32→uint32 单调键（正数翻转符号位 / 负数全翻）→ 宿主 32 步二分，
//   每步一个 count kernel（块内归约 + 每 block 一次 atomicAdd——整数加法
//   交换律 → count 与执行序无关 → 阈值确定 → 结果确定）；
//   compact 三 kernel：C1 每块 gt/eq 计数 → C2 单块前缀 + 配额 quota=k−c_gt
//   → C3 每块两段写（>T 全写；==T 按前缀+配额截断）。写序声明：输出 =
//   ">T 段（索引升序）+ ==T 段（索引升序）"两段拼接——非全局索引升序
//   （开发期 compare 的 ascending 断言据此修正为集合相等 + 分段升序）；
//   与 CPU torch.topk（value 降序）只是"顺序"不同，集合相等，注意力对集合
//   序不敏感（求和顺序差异在容差内，compare_correctness 以集合相等为准）。
//
// 确定性：除 count 的交换律整数 atomicAdd 外无 atomics；三次连跑位级一致由
// compare_correctness 验证。
// ============================================================================

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int D_LAT = 512;             // kv_lora_rank
constexpr int D_ROPE = 64;             // qk_rope_head_dim
constexpr int D_IDX = 128;             // index_head_dim
constexpr int H_IDX = 64;              // index_n_heads
constexpr int THREADS = 256;           // 8 warps
constexpr int WARPS = 8;
constexpr int LANES = 32;
constexpr unsigned int FULL_MASK = 0xffffffffu;
constexpr int DIMS_PER_LANE = D_LAT / 32;          // 16：sparse 核 acc 布局（ep25 同款）
constexpr int PAIRS_PER_LANE = DIMS_PER_LANE / 2;  // 8
constexpr int ROPE_LANES = D_ROPE / 8;             // 8
constexpr int IDX_DIMS_PER_LANE = D_IDX / LANES;   // 4：索引器 lane L 持 dims [4L,4L+4)
constexpr int MAX_SPLITS = 64;
constexpr int CHUNK_TARGET = 256;
constexpr uint32_t KEY_SIGN_FLIP_POS = 0x80000000u;  // 正数：翻转符号位
constexpr uint32_t KEY_SIGN_FLIP_NEG = 0xFFFFFFFFu;  // 负数：全部翻转

__device__ __forceinline__ uint32_t float_to_key(float v) {
    const uint32_t bits = *reinterpret_cast<const uint32_t*>(&v);
    return (bits & 0x80000000u) ? (bits ^ KEY_SIGN_FLIP_NEG) : (bits ^ KEY_SIGN_FLIP_POS);
}

__device__ __forceinline__ void unpack2(const unsigned int v, float* out) {
    const __half2 h2 = *reinterpret_cast<const __half2*>(&v);
    const float2 f = __half22float2(h2);
    out[0] = f.x;
    out[1] = f.y;
}

__device__ __forceinline__ void unpack8(const uint4& v, float* out) {
    const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h2[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(FULL_MASK, v, off);
    }
    return v;
}

// ----------------------------------------------------------------------------
// Kernel I：索引器 score。grid = ceil(N/256)，block 256。
// 每 warp 串行处理 t = block_start + warp，步长 8：
//   score_t = Σ_h w_h · ReLU(scale · (Σ_d q^I_h[d]·k^I_t[d]))
// 维度所有权：lane L 持 k^I/q^I dims (2L, 2L+1, 2L+64, 2L+65)——即两个 uint
// kv[L]、kv[L+32]（全局加载完全 coalesced），q shared 读 uint L、L+32
// （bank = L、L+32 → 32 lane 零冲突，ep25 布局教训的 q 侧应用）。
// 开发期仿真以 [4L,4L+4) 分组验证算法（算术等价、fp32 求和序差在容差内），
// 最终核采用零冲突配对布局。逐头 5 次 shfl 归约。
// ----------------------------------------------------------------------------
__global__ void dsa_indexer_score_kernel(
    const __half* __restrict__ q_i,      // [64,128]
    const float* __restrict__ w_used,    // [64]
    const __half* __restrict__ k_i,      // [N,128]
    float* __restrict__ scores,          // [N]
    const int n,
    const float scale) {
    __shared__ __half q_shared[H_IDX][D_IDX];          // 16KB
    __shared__ float w_shared[H_IDX];
    const int block_start = blockIdx.x * THREADS;
    const int block_end = min(block_start + THREADS, n);  // 块尾界——初版漏写导致每块
    const int tid = static_cast<int>(threadIdx.x);        // 越界扫到 N（N=65536 实测
    // 协作装载 q^I（8192 个 fp16 = 4096 uint）与 w            128× 重复计算，34.7ms）
    const unsigned int* qv = reinterpret_cast<const unsigned int*>(q_i);
#pragma unroll 4
    for (int u = tid; u < H_IDX * D_IDX / 2; u += THREADS) {
        const int h = u / (D_IDX / 2);
        const int c = (u % (D_IDX / 2)) * 2;
        unsigned int v = qv[u];
        __half2 h2 = *reinterpret_cast<const __half2*>(&v);
        q_shared[h][c] = h2.x;
        q_shared[h][c + 1] = h2.y;
    }
    if (tid < H_IDX) {
        w_shared[tid] = w_used[tid];
    }
    __syncthreads();

    const int lane = tid & 31;
    int t = block_start + (tid >> 5);
    for (; t < block_end; t += WARPS) {
        // k^I 行：lane L 取 uint[L]（dims 2L,2L+1）+ uint[L+32]（dims 2L+64,2L+65）
        const unsigned int* kv = reinterpret_cast<const unsigned int*>(k_i + static_cast<size_t>(t) * D_IDX);
        float kreg[IDX_DIMS_PER_LANE];
        unpack2(kv[lane], kreg);
        unpack2(kv[lane + LANES], kreg + 2);
        float score = 0.0f;
#pragma unroll 8
        for (int h = 0; h < H_IDX; ++h) {
            const unsigned int* qh = reinterpret_cast<const unsigned int*>(q_shared[h]);
            float qreg[IDX_DIMS_PER_LANE];
            unpack2(qh[lane], qreg);              // dims 2L,2L+1（零 bank conflict）
            unpack2(qh[lane + LANES], qreg + 2);  // dims 2L+64,2L+65
            float partial = 0.0f;
#pragma unroll
            for (int i = 0; i < IDX_DIMS_PER_LANE; ++i) {
                partial += qreg[i] * kreg[i];
            }
            partial = warp_reduce_sum(partial);
            score += w_shared[h] * fmaxf(partial * scale, 0.0f);  // ReLU(scale·dot)
        }
        scores[t] = score;
    }
}

// ----------------------------------------------------------------------------
// Kernel T：count(key >= thr_key)。grid = ceil(N/256)，block 256，每线程 1 个
// 位置；块内 shared 归约后每块一次 atomicAdd（整数加法交换律 → 结果确定）。
// ----------------------------------------------------------------------------
__global__ void dsa_count_kernel(
    const float* __restrict__ scores,    // [N]
    unsigned int* __restrict__ counter,  // [1]（宿主每步先清零）
    const int n,
    const uint32_t thr_key) {
    __shared__ unsigned int block_count;
    const int p = blockIdx.x * THREADS + static_cast<int>(threadIdx.x);
    if (threadIdx.x == 0) {
        block_count = 0;
    }
    __syncthreads();
    if (p < n) {
        const uint32_t key = float_to_key(scores[p]);
        if (key >= thr_key) {
            atomicAdd(&block_count, 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && block_count > 0) {
        atomicAdd(counter, block_count);
    }
}

// ----------------------------------------------------------------------------
// Kernel C1：每块 gt（key > T）/ eq（key == T）计数。grid = nblocks。
// ----------------------------------------------------------------------------
__global__ void dsa_block_counts_kernel(
    const float* __restrict__ scores,
    const int n,
    const uint32_t thr_key,
    int* __restrict__ gt_counts,   // [nblocks]
    int* __restrict__ eq_counts) { // [nblocks]
    __shared__ int s_gt[THREADS];
    __shared__ int s_eq[THREADS];
    const int p = blockIdx.x * THREADS + static_cast<int>(threadIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    int gt = 0;
    int eq = 0;
    if (p < n) {
        const uint32_t key = float_to_key(scores[p]);
        gt = (key > thr_key) ? 1 : 0;
        eq = (key == thr_key) ? 1 : 0;
    }
    s_gt[tid] = gt;
    s_eq[tid] = eq;
    __syncthreads();
    if (tid == 0) {
        int a = 0;
        int e = 0;
        for (int i = 0; i < THREADS; ++i) {
            a += s_gt[i];
            e += s_eq[i];
        }
        gt_counts[blockIdx.x] = a;
        eq_counts[blockIdx.x] = e;
    }
}

// ----------------------------------------------------------------------------
// Kernel C2：单块，块级前缀 + 配额。nblocks ≤ 1024，thread 0 串行前缀。
// ----------------------------------------------------------------------------
__global__ void dsa_prefix_kernel(
    const int* __restrict__ gt_counts,   // [nblocks]
    const int* __restrict__ eq_counts,   // [nblocks]
    const int nblocks,
    const int k,
    int* __restrict__ gt_prefix,         // [nblocks]（exclusive）
    int* __restrict__ eq_prefix,         // [nblocks]（exclusive）
    int* __restrict__ scalars) {         // [0]=c_gt [1]=quota
    if (threadIdx.x == 0) {
        int a = 0;
        int e = 0;
        for (int b = 0; b < nblocks; ++b) {
            gt_prefix[b] = a;
            a += gt_counts[b];
            eq_prefix[b] = e;
            e += eq_counts[b];
        }
        scalars[0] = a;                    // c_gt = count(> T) ≤ k−1（二分不变量）
        scalars[1] = k - a;                // quota = k − c_gt ≤ count(== T)
    }
}

// ----------------------------------------------------------------------------
// Kernel C3：确定性 compact。每块覆盖连续 256 个位置（每线程 1 个位置，
// 线程序 = 位置升序）；gt 全写，eq 按"块前缀 + 全局配额"截断——无 atomics。
// ----------------------------------------------------------------------------
__global__ void dsa_compact_kernel(
    const float* __restrict__ scores,
    const int n,
    const uint32_t thr_key,
    const int* __restrict__ gt_prefix,   // [nblocks] exclusive
    const int* __restrict__ eq_prefix,   // [nblocks] exclusive
    const int* __restrict__ scalars,     // [0]=c_gt [1]=quota
    int* __restrict__ out_indices) {     // [k]
    __shared__ int s_rank[THREADS];      // 本块内本线程之前（位置更小）的 eligible 数
    __shared__ int s_rank_eq[THREADS];
    const int p = blockIdx.x * THREADS + static_cast<int>(threadIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int gt = (p < n) ? (float_to_key(scores[p]) > thr_key ? 1 : 0) : 0;
    const int eq = (p < n) ? (float_to_key(scores[p]) == thr_key ? 1 : 0) : 0;
    s_rank[tid] = gt;
    s_rank_eq[tid] = eq;
    __syncthreads();
    if (tid == 0) {  // 块内 exclusive scan（256 元素，单线程串行，确定性）
        int a = 0;
        int e = 0;
        for (int i = 0; i < THREADS; ++i) {
            const int g = s_rank[i];
            s_rank[i] = a;
            a += g;
            const int q = s_rank_eq[i];
            s_rank_eq[i] = e;
            e += q;
        }
    }
    __syncthreads();
    const int c_gt = scalars[0];
    const int quota = scalars[1];
    if (p < n && gt) {
        out_indices[gt_prefix[blockIdx.x] + s_rank[tid]] = p;       // >T 全写
    }
    if (p < n && eq) {
        // 本块允许写的 eq 数 = clamp(quota − eq_prefix[b], 0, 本块 eq 计数)；
        // 配额耗尽的块不写——全局（位置升序）截断，确定性、无 atomics。
        const int allowed = quota - eq_prefix[blockIdx.x];
        if (allowed > 0 && s_rank_eq[tid] < allowed) {
            out_indices[c_gt + eq_prefix[blockIdx.x] + s_rank_eq[tid]] = p;
        }
    }
}

// ----------------------------------------------------------------------------
// Kernel S：gather split attention（双模式）。ep25 mla_split_kernel 骨架，
// 唯一结构改动：token = indices ? indices[i] : i（identity 模式 = dense 对照）。
// acc 布局沿用 ep25 无冲突方案（lane L 持 dims (2L+64k, 2L+64k+1)）。
// ----------------------------------------------------------------------------
__global__ void dsa_gather_split_kernel(
    const __half* __restrict__ q_abs,    // [H,512]
    const __half* __restrict__ q_rope,   // [H,64]
    const __half* __restrict__ c_kv,     // [N,512]
    const __half* __restrict__ k_rope,   // [N,64]
    const int* __restrict__ indices,     // [k]（nullptr = identity）
    float* __restrict__ psum,            // [H,S,512]
    float* __restrict__ pl,              // [H,S]
    float* __restrict__ pm,              // [H,S]
    const int k_list,
    const int splits,
    const int chunk,
    const float scale) {
    const int h = static_cast<int>(blockIdx.x);
    const int sp = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int i_start = sp * chunk;
    const int i_end = min(i_start + chunk, k_list);

    __shared__ float acc[WARPS][D_LAT];  // 16KB，无冲突布局（ep25 教训③）
    __shared__ float warp_m[WARPS];
    __shared__ float warp_l[WARPS];
    __shared__ float warp_w[WARPS];
    __shared__ float merge_scratch[2];

    float qreg[DIMS_PER_LANE];
    {
        const unsigned int* qv = reinterpret_cast<const unsigned int*>(
            q_abs + static_cast<size_t>(h) * D_LAT);
#pragma unroll
        for (int kk = 0; kk < PAIRS_PER_LANE; ++kk) {
            unpack2(qv[lane + 32 * kk], qreg + 2 * kk);
        }
    }
#pragma unroll
    for (int kk = 0; kk < PAIRS_PER_LANE; ++kk) {
        acc[warp][2 * lane + 64 * kk] = 0.0f;
        acc[warp][2 * lane + 64 * kk + 1] = 0.0f;
    }
    float qrope_reg[8];
    if (lane < ROPE_LANES) {
        const uint4* qr = reinterpret_cast<const uint4*>(
            q_rope + static_cast<size_t>(h) * D_ROPE);
        unpack8(qr[lane], qrope_reg);
    }

    float m = -INFINITY;
    float l = 0.0f;
    int i = i_start + warp;
    unsigned int cur[PAIRS_PER_LANE];
    unsigned int nxt[PAIRS_PER_LANE];
    uint4 rcur0 = make_uint4(0u, 0u, 0u, 0u), rnext0 = rcur0;
    auto load_row = [&](int tok) {
        const unsigned int* cv = reinterpret_cast<const unsigned int*>(
            c_kv + static_cast<size_t>(tok) * D_LAT);
#pragma unroll
        for (int kk = 0; kk < PAIRS_PER_LANE; ++kk) {
            cur[kk] = cv[lane + 32 * kk];
        }
        if (lane < ROPE_LANES) {
            rcur0 = reinterpret_cast<const uint4*>(k_rope + static_cast<size_t>(tok) * D_ROPE)[lane];
        }
    };
    if (i < i_end) {
        load_row(indices ? static_cast<int>(indices[i]) : i);
    }
    while (i < i_end) {
        const int inext = i + WARPS;
        if (inext < i_end) {  // 预取下一选中行（gather 模式为随机行）
            const int tok_next = indices ? static_cast<int>(indices[inext]) : inext;
            const unsigned int* cvn = reinterpret_cast<const unsigned int*>(
                c_kv + static_cast<size_t>(tok_next) * D_LAT);
#pragma unroll
            for (int kk = 0; kk < PAIRS_PER_LANE; ++kk) {
                nxt[kk] = cvn[lane + 32 * kk];
            }
            if (lane < ROPE_LANES) {
                rnext0 = reinterpret_cast<const uint4*>(k_rope + static_cast<size_t>(tok_next) * D_ROPE)[lane];
            }
        }
        float creg[DIMS_PER_LANE];
#pragma unroll
        for (int kk = 0; kk < PAIRS_PER_LANE; ++kk) {
            unpack2(cur[kk], creg + 2 * kk);
        }
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < DIMS_PER_LANE; ++j) {
            dot += qreg[j] * creg[j];
        }
        dot = warp_reduce_sum(dot);
        float rope_dot = 0.0f;
        if (lane < ROPE_LANES) {
            float kreg[8];
            unpack8(rcur0, kreg);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                rope_dot += qrope_reg[j] * kreg[j];
            }
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                rope_dot += __shfl_xor_sync(0x000000ffu, rope_dot, off);
            }
        }
        rope_dot = __shfl_sync(FULL_MASK, rope_dot, 0);
        const float score = (dot + rope_dot) * scale;
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);   // m=-inf → 0
        const float p = expf(score - m_new);
        l = l * alpha + p;
        if (m_new > m) {
#pragma unroll
            for (int kk = 0; kk < PAIRS_PER_LANE; ++kk) {
                acc[warp][2 * lane + 64 * kk] *= alpha;
                acc[warp][2 * lane + 64 * kk + 1] *= alpha;
            }
        }
        if (p > 0.0f) {
#pragma unroll
            for (int kk = 0; kk < PAIRS_PER_LANE; ++kk) {
                acc[warp][2 * lane + 64 * kk] += p * creg[2 * kk];
                acc[warp][2 * lane + 64 * kk + 1] += p * creg[2 * kk + 1];
            }
        }
        m = m_new;
#pragma unroll
        for (int kk = 0; kk < PAIRS_PER_LANE; ++kk) {
            cur[kk] = nxt[kk];
        }
        rcur0 = rnext0;
        i = inext;
    }

    warp_m[warp] = m;
    warp_l[warp] = l;
    __syncthreads();
    if (tid == 0) {
        float m_global = -INFINITY;
        for (int wi = 0; wi < WARPS; ++wi) {
            m_global = fmaxf(m_global, warp_m[wi]);
        }
        float l_total = 0.0f;
        for (int wi = 0; wi < WARPS; ++wi) {
            const float w = (warp_m[wi] == -INFINITY) ? 0.0f : expf(warp_m[wi] - m_global);
            warp_w[wi] = w;
            warp_l[wi] *= w;
            l_total += warp_l[wi];
        }
        merge_scratch[0] = l_total;
    }
    __syncthreads();
    const float l_total = merge_scratch[0];

    float* dst = psum + (static_cast<size_t>(h) * splits + sp) * D_LAT;
    for (int d0 = tid * 2; d0 < D_LAT; d0 += THREADS * 2) {
        float s0 = 0.0f;
        float s1 = 0.0f;
#pragma unroll
        for (int wi = 0; wi < WARPS; ++wi) {
            s0 += warp_w[wi] * acc[wi][d0];
            s1 += warp_w[wi] * acc[wi][d0 + 1];
        }
        dst[d0] = s0;
        dst[d0 + 1] = s1;
    }
    if (tid == 0) {
        pl[static_cast<size_t>(h) * splits + sp] = l_total;
        float m_split = -INFINITY;
        for (int wi = 0; wi < WARPS; ++wi) {
            m_split = fmaxf(m_split, warp_m[wi]);
        }
        pm[static_cast<size_t>(h) * splits + sp] = m_split;
    }
}

// ----------------------------------------------------------------------------
// Kernel M：跨 split 合并（ep25 merge 同款：pm 局部 max 重归一，空 split 守卫）。
// ----------------------------------------------------------------------------
__global__ void dsa_gather_merge_kernel(
    const float* __restrict__ psum,   // [H,S,512]
    const float* __restrict__ pl,     // [H,S]
    const float* __restrict__ pm,     // [H,S]
    __half* __restrict__ s_out,       // [H,512]
    const int splits) {
    const int h = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    float m_global = -INFINITY;
    for (int j = 0; j < splits; ++j) {
        m_global = fmaxf(m_global, pm[static_cast<size_t>(h) * splits + j]);
    }
    float l_total = 0.0f;
    for (int j = 0; j < splits; ++j) {
        const float m_j = pm[static_cast<size_t>(h) * splits + j];
        const float w = (m_j == -INFINITY) ? 0.0f : expf(m_j - m_global);
        l_total += w * pl[static_cast<size_t>(h) * splits + j];
    }
    const float inv = (l_total > 0.0f) ? 1.0f / l_total : 0.0f;
    for (int d0 = tid * 2; d0 < D_LAT; d0 += THREADS * 2) {
        float s0 = 0.0f;
        float s1 = 0.0f;
        for (int j = 0; j < splits; ++j) {
            const float m_j = pm[static_cast<size_t>(h) * splits + j];
            const float w = (m_j == -INFINITY) ? 0.0f : expf(m_j - m_global);
            const size_t base = (static_cast<size_t>(h) * splits + j) * D_LAT;
            s0 += w * psum[base + d0];
            s1 += w * psum[base + d0 + 1];
        }
        s_out[static_cast<size_t>(h) * D_LAT + d0] = __float2half(s0 * inv);
        s_out[static_cast<size_t>(h) * D_LAT + d0 + 1] = __float2half(s1 * inv);
    }
}

}  // namespace

// ----------------------------------------------------------------------------
// torch 扩展入口
// ----------------------------------------------------------------------------

at::Tensor dsa_indexer_scores(
    const at::Tensor& q_i,
    const at::Tensor& w_used,
    const at::Tensor& k_i,
    double scale) {
    TORCH_CHECK(q_i.scalar_type() == at::kHalf && k_i.scalar_type() == at::kHalf,
                "合同：q_i/k_i io dtype = float16，实际 ", q_i.scalar_type(), "/", k_i.scalar_type());
    TORCH_CHECK(w_used.scalar_type() == at::kFloat,
                "合同：w_used dtype = float32，实际 ", w_used.scalar_type());
    TORCH_CHECK(q_i.is_contiguous() && k_i.is_contiguous() && w_used.is_contiguous(),
                "合同：输入必须连续");
    TORCH_CHECK(q_i.dim() == 2 && q_i.size(0) == H_IDX && q_i.size(1) == D_IDX,
                "q_i 必须是 [64,128]，实际 ", q_i.sizes());
    TORCH_CHECK(w_used.numel() == H_IDX, "w_used 必须是 [64]，实际 ", w_used.sizes());
    TORCH_CHECK(k_i.dim() == 2 && k_i.size(1) == D_IDX,
                "k_i 必须是 [N,128]，实际 ", k_i.sizes());
    const int n = static_cast<int>(k_i.size(0));
    TORCH_CHECK(n > 0, "合同：N ≥ 1");
    auto scores = at::empty({n}, q_i.options().dtype(at::kFloat));
    const at::cuda::CUDAGuard device_guard(q_i.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (n + THREADS - 1) / THREADS;
    dsa_indexer_score_kernel<<<blocks, THREADS, 0, stream>>>(
        reinterpret_cast<const __half*>(q_i.data_ptr<at::Half>()),
        w_used.data_ptr<float>(),
        reinterpret_cast<const __half*>(k_i.data_ptr<at::Half>()),
        scores.data_ptr<float>(),
        n, static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scores;
}

at::Tensor dsa_topk(const at::Tensor& scores, int64_t k_req) {
    TORCH_CHECK(scores.scalar_type() == at::kFloat && scores.dim() == 1 && scores.is_contiguous(),
                "合同：scores 必须是连续 [N] float32，实际 dtype=", scores.scalar_type());
    const int n = static_cast<int>(scores.numel());
    TORCH_CHECK(n > 0, "合同：N ≥ 1");
    TORCH_CHECK(k_req >= 0, "k 必须 ≥ 0");
    const int k = static_cast<int>(std::min<int64_t>(k_req, n));
    auto indices = at::empty({k}, scores.options().dtype(at::kInt));
    if (k == 0) {
        return indices;  // 空集：合法输入，返回空索引（行为记录于合同/日志）
    }
    const at::cuda::CUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int nblocks = (n + THREADS - 1) / THREADS;
    auto counter = at::zeros({1}, scores.options().dtype(at::kInt));
    auto gt_counts = at::empty({nblocks}, scores.options().dtype(at::kInt));
    auto eq_counts = at::empty({nblocks}, scores.options().dtype(at::kInt));
    auto gt_prefix = at::empty({nblocks}, scores.options().dtype(at::kInt));
    auto eq_prefix = at::empty({nblocks}, scores.options().dtype(at::kInt));
    auto scalars = at::zeros({2}, scores.options().dtype(at::kInt));
    const float* s_ptr = scores.data_ptr<float>();

    // 32 步二分（不变量：count(key ≥ lo) ≥ k）
    uint32_t lo = 0u;
    uint32_t hi = 0xFFFFFFFFu;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2 + 1;
        cudaMemsetAsync(counter.data_ptr(), 0, sizeof(int), stream);
        dsa_count_kernel<<<nblocks, THREADS, 0, stream>>>(
            s_ptr, reinterpret_cast<unsigned int*>(counter.data_ptr()), n, mid);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        int count = 0;
        C10_CUDA_CHECK(cudaMemcpyAsync(&count, counter.data_ptr(), sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (count >= k) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    const uint32_t thr_key = lo;
    dsa_block_counts_kernel<<<nblocks, THREADS, 0, stream>>>(
        s_ptr, n, thr_key,
        gt_counts.data_ptr<int>(), eq_counts.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dsa_prefix_kernel<<<1, 1, 0, stream>>>(
        gt_counts.data_ptr<int>(), eq_counts.data_ptr<int>(), nblocks, k,
        gt_prefix.data_ptr<int>(), eq_prefix.data_ptr<int>(), scalars.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dsa_compact_kernel<<<nblocks, THREADS, 0, stream>>>(
        s_ptr, n, thr_key,
        gt_prefix.data_ptr<int>(), eq_prefix.data_ptr<int>(), scalars.data_ptr<int>(),
        indices.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return indices;
}

at::Tensor dsa_sparse_attention(
    const at::Tensor& q_abs,
    const at::Tensor& q_rope,
    const at::Tensor& c_kv,
    const at::Tensor& k_rope,
    const at::Tensor& indices,   // int32 [k]；identity 模式传空张量
    bool identity,
    double scale,
    int64_t splits_req) {
    TORCH_CHECK(q_abs.scalar_type() == at::kHalf && q_rope.scalar_type() == at::kHalf &&
                c_kv.scalar_type() == at::kHalf && k_rope.scalar_type() == at::kHalf,
                "合同：本期 CUDA 核 io dtype = float16，实际 ", q_abs.scalar_type(), "/",
                q_rope.scalar_type(), "/", c_kv.scalar_type(), "/", k_rope.scalar_type());
    TORCH_CHECK(q_abs.device().type() == c10::DeviceType::CUDA &&
                c_kv.device().type() == c10::DeviceType::CUDA,
                "本扩展是 CUDA 实现，不接受 ", q_abs.device(), " Tensor");
    TORCH_CHECK(q_abs.is_contiguous() && q_rope.is_contiguous() &&
                c_kv.is_contiguous() && k_rope.is_contiguous(),
                "合同：本期 Kernel 只接受连续输入（非连续布局是第 06 期主题）");
    TORCH_CHECK(q_abs.dim() == 2 && q_abs.size(1) == D_LAT,
                "q_abs 必须是 [H,512]，实际 ", q_abs.sizes());
    TORCH_CHECK(q_rope.dim() == 2 && q_rope.size(1) == D_ROPE,
                "q_rope 必须是 [H,64]，实际 ", q_rope.sizes());
    TORCH_CHECK(c_kv.dim() == 2 && c_kv.size(1) == D_LAT,
                "c_kv 必须是 [N,512]，实际 ", c_kv.sizes());
    TORCH_CHECK(k_rope.dim() == 2 && k_rope.size(0) == c_kv.size(0) && k_rope.size(1) == D_ROPE,
                "k_rope 必须是 [N,64]");
    TORCH_CHECK(indices.scalar_type() == at::kInt && indices.is_contiguous(),
                "indices 必须是连续 int32（identity 模式传空张量）");
    const int heads = static_cast<int>(q_abs.size(0));
    TORCH_CHECK(q_rope.size(0) == heads && c_kv.size(0) > 0,
                "H 不一致或 N=0（合同：N ≥ 1）");
    // 注：indices 的取值范围 [0,N) 是调用方合同（不做逐元素检查——那会引入
    // 每次 D2H 同步，破坏事件计时；越界行为在 smoke/合同文档中声明为未定义）。
    const int k_list = identity ? static_cast<int>(c_kv.size(0))
                                : static_cast<int>(indices.numel());
    auto s_out = at::empty({heads, D_LAT}, q_abs.options());
    if (k_list == 0) {
        s_out.zero_();  // 空集：s=0（§9 空集行为，与 CPU 参考一致）
        return s_out;
    }
    int splits;
    if (splits_req > 0) {
        splits = static_cast<int>(std::min<int64_t>(splits_req, MAX_SPLITS));
    } else {
        splits = (k_list + CHUNK_TARGET - 1) / CHUNK_TARGET;
        splits = std::max(1, std::min(splits, MAX_SPLITS));
    }
    int chunk = (k_list + splits - 1) / splits;
    chunk = ((chunk + WARPS - 1) / WARPS) * WARPS;
    chunk = std::max(chunk, WARPS);

    auto psum = at::empty({heads, splits, D_LAT}, q_abs.options().dtype(at::kFloat));
    auto pl = at::empty({heads, splits}, q_abs.options().dtype(at::kFloat));
    auto pm = at::empty({heads, splits}, q_abs.options().dtype(at::kFloat));

    const at::cuda::CUDAGuard device_guard(q_abs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(splits));
    dsa_gather_split_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __half*>(q_abs.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(q_rope.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(c_kv.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_rope.data_ptr<at::Half>()),
        identity ? nullptr : indices.data_ptr<int>(),
        psum.data_ptr<float>(), pl.data_ptr<float>(), pm.data_ptr<float>(),
        k_list, splits, chunk, static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dsa_gather_merge_kernel<<<dim3(static_cast<unsigned>(heads)), THREADS, 0, stream>>>(
        psum.data_ptr<float>(), pl.data_ptr<float>(), pm.data_ptr<float>(),
        reinterpret_cast<__half*>(s_out.data_ptr<at::Half>()), splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return s_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("dsa_indexer_scores", &dsa_indexer_scores,
               py::arg("q_i"), py::arg("w_used"), py::arg("k_i"), py::arg("scale"),
               "第 25 期 Lightning Indexer score 核（warp/头分工，fp16 io fp32 累加）");
    module.def("dsa_topk", &dsa_topk,
               py::arg("scores"), py::arg("k"),
               "阈值法 top-k（32 步二分 count + 确定性 compact，共 ~35 launch）");
    module.def("dsa_sparse_attention", &dsa_sparse_attention,
               py::arg("q_abs"), py::arg("q_rope"), py::arg("c_kv"), py::arg("k_rope"),
               py::arg("indices"), py::arg("identity"), py::arg("scale"), py::arg("splits") = 0,
               "gather split attention（identity=true 为 dense 对照）");
}
