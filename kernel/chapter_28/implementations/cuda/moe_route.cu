// 第 28 期 CUDA：naive 路由四核 + 分桶 GEMM——selftest / 30 格分段计时矩阵 / 负载不均注入
//
// 核分解（语义由 implementations/cpu/kernel_sim.py 位级仿真锁定 9/9）：
//   K1 gate_scores    每 (token, expert) 一线程：x[N,d]·router[E,d] → logits[N,E]
//   K2 softmax_topk   每 token 一 warp：warp max/sum + k 轮 (val,idx) 字典序
//                     argmax（tie-break = 更低专家索引，与 stable argsort 等价）
//                     + 归一化（Qwen3 norm_topk_prob=true 口径）
//   K3 histogram      smem 直方图（atomicAdd 整数和 = 顺序无关确定）+ E 上
//                     exclusive scan（单 block）
//   K4 scatter_fill   确定性散位（无原子）：rank = pair 字典序前驱中同专家计数；
//                     桶内顺序 = token 升序；输出 pair_pos 反查表
//   G  分桶 GEMM      逐非空桶 32×32 tiled GEMM（ch18 设计，smem 8KB）
//   + gather_pairs / silu_mul / weighted_combine（slot 序确定性合并，无原子）
//
// 用法：
//   moe_route --selftest    小 shape 全链 device 对拍 host naive 参考（默认）
//   moe_route --bench       30 格分段计时矩阵（N×E×k，同 CPU 列口径）+ dense 等价
//   moe_route --imbalance   均匀 vs 80/20-Zipf 注入（E=64,k=6,N∈{256,2048}）
//
// 编译：nvcc -O3 -std=c++17 -arch=sm_89 moe_route.cu -o moe_route

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// 确定性 PRNG（xorshift32；负载不均注入与权重生成用，跨平台可复现）
// ---------------------------------------------------------------------------
static unsigned g_rng = 20260329u;
static unsigned xrnd() {
    unsigned x = g_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_rng = x;
    return x;
}
static double urand() { return (double)(xrnd() % 1000000) / 1000000.0; }
static float frand_scale(float s) { return s * (float)(urand() * 2.0 - 1.0); }

// ---------------------------------------------------------------------------
// K1: gate scores —— 每 (token, expert) 一线程点积
// ---------------------------------------------------------------------------
__global__ void gate_scores_kernel(const float* __restrict__ x,
                                   const float* __restrict__ router,
                                   float* __restrict__ logits,
                                   int n_tokens, int d, int n_experts) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;  // pair = t * E + e
    int total = n_tokens * n_experts;
    if (pair >= total) return;
    int t = pair / n_experts;
    int e = pair % n_experts;
    const float* xt = x + (size_t)t * d;
    const float* re = router + (size_t)e * d;
    float acc = 0.0f;
    for (int i = 0; i < d; ++i) acc += xt[i] * re[i];
    logits[pair] = acc;
}

// ---------------------------------------------------------------------------
// K2: top-k（按原始 logits 降序选择——softmax 单调故与概率序等价；tie-break =
//      更低专家索引，(score, idx) 字典序，与 stable argsort 等价）
//      + 选中分数的 softmax 权重（expf 保精度）+ 归一化（Qwen3 口径）
//     每 token 一 warp（blockDim.x == 32）
//     设计注记（KI-29-03）：早期版本对每轮候选用 __expf 重算概率再比较，
//     设备/主机 exp 舍入差会翻转近分专家的选择（selftest max_abs 6.2e-02 的
//     根因）——改为对 logits 本身排序后，选择只依赖 (score, idx) 字典序，
//     与 host 语义严格一致；权重误差 ~1e-6 级。
// ---------------------------------------------------------------------------
__global__ void softmax_topk_kernel(const float* __restrict__ logits,
                                    int* __restrict__ topk_idx,
                                    float* __restrict__ topk_w,
                                    int n_tokens, int n_experts, int k) {
    int t = blockIdx.x;
    int lane = threadIdx.x;
    unsigned full = 0xffffffffu;
    const float* lg = logits + (size_t)t * n_experts;

    float local_max = -INFINITY;
    for (int e = lane; e < n_experts; e += 32)
        local_max = fmaxf(local_max, lg[e]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(full, local_max, off));
    float m = __shfl_sync(full, local_max, 0);

    float local_sum = 0.0f;
    for (int e = lane; e < n_experts; e += 32)
        local_sum += expf(lg[e] - m);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor_sync(full, local_sum, off);
    float denom_inv = 1.0f / __shfl_sync(full, local_sum, 0);

    int* sel = topk_idx + (size_t)t * k;
    float* w = topk_w + (size_t)t * k;
    for (int round = 0; round < k; ++round) {
        float best_v = -INFINITY;
        int best_e = n_experts;
        for (int e = lane; e < n_experts; e += 32) {
            bool taken = false;
            for (int r = 0; r < round; ++r) taken |= (sel[r] == e);
            if (taken) continue;
            float v = lg[e];  // 原始 logits 比较（单调等价于概率序）
            if (v > best_v || (v == best_v && e < best_e)) { best_v = v; best_e = e; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(full, best_v, off);
            int   oe = __shfl_xor_sync(full, best_e, off);
            if (ov > best_v || (ov == best_v && oe < best_e)) { best_v = ov; best_e = oe; }
        }
        best_e = __shfl_sync(full, best_e, 0);
        if (lane == 0) {
            sel[round] = best_e;
            w[round] = expf(lg[best_e] - m) * denom_inv;
        }
        __syncwarp();
    }
    if (lane == 0) {
        float s = 0.0f;
        for (int r = 0; r < k; ++r) s += w[r];
        for (int r = 0; r < k; ++r) w[r] /= s;
    }
}

// ---------------------------------------------------------------------------
// K3: 直方图 + exclusive scan
// ---------------------------------------------------------------------------
__global__ void histogram_kernel(const int* __restrict__ topk_idx, int* __restrict__ counts,
                                 int total_pairs) {
    __shared__ int smem_hist[1024];
    for (int e = threadIdx.x; e < 1024; e += blockDim.x) smem_hist[e] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_pairs;
         i += gridDim.x * blockDim.x)
        atomicAdd(&smem_hist[topk_idx[i]], 1);
    __syncthreads();
    for (int e = threadIdx.x; e < 1024; e += blockDim.x)
        atomicAdd(&counts[e], smem_hist[e]);
}

__global__ void exclusive_scan_kernel(const int* __restrict__ counts, int* __restrict__ offsets,
                                      int n_experts) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    if (tid < n_experts) temp[tid] = counts[tid];
    __syncthreads();
    for (int off = 1; off < n_experts; off <<= 1) {
        int v = (tid >= off && tid < n_experts) ? temp[tid - off] : 0;
        __syncthreads();
        if (tid < n_experts) temp[tid] += v;
        __syncthreads();
    }
    if (tid < n_experts) offsets[tid] = (tid == 0) ? 0 : temp[tid - 1];
}

// ---------------------------------------------------------------------------
// K4: 确定性散位（无原子）——rank = pair 字典序前驱中同专家计数
//     （教学版 O(P²)；正式版换 per-slot token 前缀扫描或 stable radix sort，
//      语义与 kernel_sim 公式一致）
// ---------------------------------------------------------------------------
__global__ void scatter_fill_kernel(const int* __restrict__ topk_idx,
                                    const int* __restrict__ offsets,
                                    int* __restrict__ pair_token,
                                    int* __restrict__ pair_expert,
                                    int* __restrict__ pair_slot,
                                    int* __restrict__ pair_pos,
                                    int n_tokens, int k) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * k;
    if (pair >= total) return;
    int t = pair / k;
    int s = pair % k;
    int e = topk_idx[pair];
    int rank = 0;
    for (int p = 0; p < pair; ++p)
        if (topk_idx[p] == e) ++rank;
    int pos = offsets[e] + rank;
    pair_token[pos] = t;
    pair_expert[pos] = e;
    pair_slot[pos] = s;
    pair_pos[pair] = pos;
}

// ---------------------------------------------------------------------------
// gather / silu·mul / 加权合并（slot 序确定性，无原子）
// ---------------------------------------------------------------------------
__global__ void gather_pairs_kernel(const float* __restrict__ x,
                                    const int* __restrict__ pair_token,
                                    float* __restrict__ pair_x,
                                    int total_pairs, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs * d) return;
    int p = idx / d;
    int i = idx % d;
    pair_x[(size_t)p * d + i] = x[(size_t)pair_token[p] * d + i];
}

__global__ void silu_mul_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                float* __restrict__ act, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float g = gate[i];
    act[i] = (g / (1.0f + __expf(-g))) * up[i];
}

// 每 (token, d-elem) 一线程：slot 序加权求和（经 pair_pos 反查排序后位置）
__global__ void weighted_combine_kernel(const float* __restrict__ down_out,
                                        const float* __restrict__ topk_w,
                                        const int* __restrict__ pair_pos,
                                        float* __restrict__ y,
                                        int n_tokens, int k, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * d;
    if (i >= total) return;
    int t = i / d;
    int col = i % d;
    float acc = 0.0f;
    for (int s = 0; s < k; ++s) {
        int p = pair_pos[(size_t)t * k + s];
        acc += topk_w[(size_t)t * k + s] * down_out[(size_t)p * d + col];
    }
    y[i] = acc;
}

// ---------------------------------------------------------------------------
// G: 分桶 GEMM——32×32 tile（ch18 单层 tiling，smem 8KB）
// ---------------------------------------------------------------------------
#define TILE 32
__global__ void tiled_gemm_kernel(const float* __restrict__ A,   // [M, K]
                                  const float* __restrict__ B,   // [N, K] 行主权重（转置视角）
                                  float* __restrict__ C,         // [M, N]
                                  int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    int n_tiles = (K + TILE - 1) / TILE;
    for (int tk = 0; tk < n_tiles; ++tk) {
        int a_col = tk * TILE + threadIdx.x;
        int b_row = tk * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[(size_t)row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && b_row < K) ? B[(size_t)col * K + b_row] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int p = 0; p < TILE; ++p) acc += As[threadIdx.y][p] * Bs[p][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

static void gemm_nt(const float* d_A, const float* d_W, float* d_C,
                    int M, int rows, int K, cudaStream_t stream) {
    if (M <= 0 || rows <= 0) return;
    dim3 block(TILE, TILE);
    dim3 grid((rows + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    tiled_gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_W, d_C, M, rows, K);
}

// ---------------------------------------------------------------------------
// host 参考（真值）：朴素逐 token 全量专家 + slot 序加权合并
// ---------------------------------------------------------------------------
static void host_reference(const std::vector<float>& x, const std::vector<float>& Wg,
                           const std::vector<float>& Wu, const std::vector<float>& Wd,
                           const std::vector<float>& router, int N, int d, int d_ff,
                           int E, int k, std::vector<float>& y) {
    y.assign((size_t)N * d, 0.0f);
    for (int t = 0; t < N; ++t) {
        std::vector<float> lg(E), pr(E);
        for (int e = 0; e < E; ++e) {
            float acc = 0.0f;
            for (int i = 0; i < d; ++i) acc += x[(size_t)t * d + i] * router[(size_t)e * d + i];
            lg[e] = acc;
        }
        float m = *std::max_element(lg.begin(), lg.end()), s = 0.0f;
        for (int e = 0; e < E; ++e) { pr[e] = std::exp(lg[e] - m); s += pr[e]; }
        // 选择 = 原始 logits 降序（softmax 单调等价），同分低索引胜出——
        // 与 device K2 语义严格一致（KI-29-03：概率序会引入 exp 舍入翻转）
        std::vector<int> sel;
        std::vector<bool> taken(E, false);
        for (int r = 0; r < k; ++r) {
            int best = -1;
            for (int e = 0; e < E; ++e)
                if (!taken[e] && (best < 0 || lg[e] > lg[best])) best = e;
            taken[best] = true;
            sel.push_back(best);
        }
        std::vector<float> w(k);
        float ws = 0.0f;
        for (int r = 0; r < k; ++r) { w[r] = pr[sel[r]] / s; ws += w[r]; }
        for (int r = 0; r < k; ++r) w[r] /= ws;
        for (int r = 0; r < k; ++r) {
            int e = sel[r];
            for (int f = 0; f < d_ff; ++f) {
                float g = 0.0f, u = 0.0f;
                for (int i = 0; i < d; ++i) {
                    g += x[(size_t)t * d + i] * Wg[((size_t)e * d_ff + f) * d + i];
                    u += x[(size_t)t * d + i] * Wu[((size_t)e * d_ff + f) * d + i];
                }
                float act = (g / (1.0f + std::exp(-g))) * u;
                for (int i = 0; i < d; ++i)
                    y[(size_t)t * d + i] += w[r] * act * Wd[((size_t)e * d + i) * d_ff + f];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 设备缓冲集合（一 cell 的全链缓冲）
// ---------------------------------------------------------------------------
struct DeviceBuf {
    float *d_x = nullptr, *d_router = nullptr, *d_Wg = nullptr, *d_Wu = nullptr, *d_Wd = nullptr;
    float *d_logits = nullptr, *d_w = nullptr;
    int *d_idx = nullptr, *d_counts = nullptr, *d_offsets = nullptr;
    int *d_pair_token = nullptr, *d_pair_expert = nullptr, *d_pair_slot = nullptr, *d_pair_pos = nullptr;
    float *d_pair_x = nullptr, *d_gate = nullptr, *d_up = nullptr, *d_act = nullptr, *d_down = nullptr;
    float *d_y = nullptr;
    // dense 等价容量
    float *d_dWg = nullptr, *d_dWu = nullptr, *d_dWd = nullptr, *d_dm1 = nullptr, *d_dm2 = nullptr, *d_dm3 = nullptr;
    int N = 0, E = 0, k = 0, d = 0, d_ff = 0;

    void alloc(int n_, int e_, int k_, int d_, int dff_) {
        N = n_; E = e_; k = k_; d = d_; d_ff = dff_;
        size_t pairs = (size_t)N * k;
        CUDA_CHECK(cudaMalloc(&d_x, (size_t)N * d * 4));
        CUDA_CHECK(cudaMalloc(&d_router, (size_t)E * d * 4));
        CUDA_CHECK(cudaMalloc(&d_Wg, (size_t)E * d_ff * d * 4));
        CUDA_CHECK(cudaMalloc(&d_Wu, (size_t)E * d_ff * d * 4));
        CUDA_CHECK(cudaMalloc(&d_Wd, (size_t)E * d * d_ff * 4));
        CUDA_CHECK(cudaMalloc(&d_logits, (size_t)N * E * 4));
        CUDA_CHECK(cudaMalloc(&d_w, pairs * 4));
        CUDA_CHECK(cudaMalloc(&d_idx, pairs * 4));
        CUDA_CHECK(cudaMalloc(&d_counts, 1024 * 4));
        CUDA_CHECK(cudaMalloc(&d_offsets, 1024 * 4));
        CUDA_CHECK(cudaMalloc(&d_pair_token, pairs * 4));
        CUDA_CHECK(cudaMalloc(&d_pair_expert, pairs * 4));
        CUDA_CHECK(cudaMalloc(&d_pair_slot, pairs * 4));
        CUDA_CHECK(cudaMalloc(&d_pair_pos, pairs * 4));
        CUDA_CHECK(cudaMalloc(&d_pair_x, pairs * d * 4));
        CUDA_CHECK(cudaMalloc(&d_gate, pairs * d_ff * 4));
        CUDA_CHECK(cudaMalloc(&d_up, pairs * d_ff * 4));
        CUDA_CHECK(cudaMalloc(&d_act, pairs * d_ff * 4));
        CUDA_CHECK(cudaMalloc(&d_down, pairs * d * 4));
        CUDA_CHECK(cudaMalloc(&d_y, (size_t)N * d * 4));
        size_t dff_t = (size_t)k * d_ff;
        CUDA_CHECK(cudaMalloc(&d_dWg, dff_t * d * 4));
        CUDA_CHECK(cudaMalloc(&d_dWu, dff_t * d * 4));
        CUDA_CHECK(cudaMalloc(&d_dWd, d * dff_t * 4));
        CUDA_CHECK(cudaMalloc(&d_dm1, (size_t)N * dff_t * 4));
        CUDA_CHECK(cudaMalloc(&d_dm2, (size_t)N * dff_t * 4));
        CUDA_CHECK(cudaMalloc(&d_dm3, (size_t)N * d * 4));
    }
    void free_all() {
        cudaFree(d_x); cudaFree(d_router); cudaFree(d_Wg); cudaFree(d_Wu); cudaFree(d_Wd);
        cudaFree(d_logits); cudaFree(d_w); cudaFree(d_idx); cudaFree(d_counts); cudaFree(d_offsets);
        cudaFree(d_pair_token); cudaFree(d_pair_expert); cudaFree(d_pair_slot); cudaFree(d_pair_pos);
        cudaFree(d_pair_x); cudaFree(d_gate); cudaFree(d_up); cudaFree(d_act); cudaFree(d_down);
        cudaFree(d_y); cudaFree(d_dWg); cudaFree(d_dWu); cudaFree(d_dWd);
        cudaFree(d_dm1); cudaFree(d_dm2); cudaFree(d_dm3);
    }
};

// 分段事件集合（8 段，与 CPU 列 profiled 同名）
struct SegEvents {
    cudaEvent_t ev[8][2];
    void create() { for (int i = 0; i < 8; ++i) { CUDA_CHECK(cudaEventCreate(&ev[i][0])); CUDA_CHECK(cudaEventCreate(&ev[i][1])); } }
    void destroy() { for (int i = 0; i < 8; ++i) { cudaEventDestroy(ev[i][0]); cudaEventDestroy(ev[i][1]); } }
};

// 单遍分段执行：桶界用调用方传入的 offsets（路由确定且分布不变）
static void run_expert_segments(DeviceBuf& B, const std::vector<int>& h_offsets,
                                cudaStream_t stream, SegEvents& ev, bool timing) {
    int threads = 256;
    int P = B.N * B.k;
    auto rec = [&](int i, bool begin) { if (timing) CUDA_CHECK(cudaEventRecord(ev.ev[i][begin ? 0 : 1], stream)); };

    rec(0, true);
    gate_scores_kernel<<<(B.N * B.E + threads - 1) / threads, threads, 0, stream>>>(
        B.d_x, B.d_router, B.d_logits, B.N, B.d, B.E);
    rec(0, false);

    rec(1, true);
    softmax_topk_kernel<<<B.N, 32, 0, stream>>>(B.d_logits, B.d_idx, B.d_w, B.N, B.E, B.k);
    rec(1, false);

    rec(2, true);
    CUDA_CHECK(cudaMemsetAsync(B.d_counts, 0, 1024 * 4, stream));
    histogram_kernel<<<(P + threads - 1) / threads, threads, 0, stream>>>(B.d_idx, B.d_counts, P);
    exclusive_scan_kernel<<<1, 1024, 1024 * 4, stream>>>(B.d_counts, B.d_offsets, B.E);
    scatter_fill_kernel<<<(P + threads - 1) / threads, threads, 0, stream>>>(
        B.d_idx, B.d_offsets, B.d_pair_token, B.d_pair_expert, B.d_pair_slot, B.d_pair_pos, B.N, B.k);
    rec(2, false);

    rec(3, true);
    gather_pairs_kernel<<<((size_t)P * B.d + threads - 1) / threads, threads, 0, stream>>>(
        B.d_x, B.d_pair_token, B.d_pair_x, P, B.d);
    rec(3, false);

    // 桶循环（gate/up）
    rec(4, true);
    for (int e = 0; e < B.E; ++e) {
        int s = h_offsets[e];       // exclusive scan：桶 e 起点 = offsets[e]
        int t = (e == B.E - 1) ? (B.N * B.k) : h_offsets[e + 1];
        int M = t - s;              // 桶 e 深度（修复：勿混用 CPU inclusive-cumsum 的桶界模式）
        if (M <= 0) continue;
        gemm_nt(B.d_pair_x + (size_t)s * B.d, B.d_Wg + (size_t)e * B.d_ff * B.d,
                B.d_gate + (size_t)s * B.d_ff, M, B.d_ff, B.d, stream);
        gemm_nt(B.d_pair_x + (size_t)s * B.d, B.d_Wu + (size_t)e * B.d_ff * B.d,
                B.d_up + (size_t)s * B.d_ff, M, B.d_ff, B.d, stream);
    }
    rec(4, false);

    rec(5, true);
    silu_mul_kernel<<<((unsigned long long)P * B.d_ff + threads - 1) / threads, threads, 0, stream>>>(
        B.d_gate, B.d_up, B.d_act, (long long)P * B.d_ff);
    rec(5, false);

    rec(6, true);
    for (int e = 0; e < B.E; ++e) {
        int s = h_offsets[e];       // exclusive scan：桶 e 起点 = offsets[e]
        int t = (e == B.E - 1) ? (B.N * B.k) : h_offsets[e + 1];
        int M = t - s;              // 桶 e 深度（修复：勿混用 CPU inclusive-cumsum 的桶界模式）
        if (M <= 0) continue;
        gemm_nt(B.d_act + (size_t)s * B.d_ff, B.d_Wd + (size_t)e * B.d * B.d_ff,
                B.d_down + (size_t)s * B.d, M, B.d, B.d_ff, stream);
    }
    rec(6, false);

    rec(7, true);
    weighted_combine_kernel<<<((size_t)B.N * B.d + threads - 1) / threads, threads, 0, stream>>>(
        B.d_down, B.d_w, B.d_pair_pos, B.d_y, B.N, B.k, B.d);
    rec(7, false);
}

static double elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return (double)ms;
}

static double p50_of(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// 无计时整链（bench 预热/桶界预热用）
static void launch_moe_chain(DeviceBuf& B, cudaStream_t stream) {
    std::vector<int> dummy_offsets(1024, 0);
    SegEvents dummy;
    run_expert_segments(B, dummy_offsets, stream, dummy, false);
}

// ---------------------------------------------------------------------------
// selftest：K1 数值检查 + 同一 host logits 之上的 K2..combine 全链 device
// 对拍 host naive 参考（选择语义严格一致的构造性对拍，见 KI-29-07）
// ---------------------------------------------------------------------------
static int run_selftest() {
    int N = 8, d = 64, d_ff = 128, E = 16, k = 6;
    printf("[ch29] selftest N=%d d=%d d_ff=%d E=%d k=%d（K1 数值检查 + 同 logits 下 K2..G 全链 device vs host naive）\n",
           N, d, d_ff, E, k);
    DeviceBuf B;
    B.alloc(N, E, k, d, d_ff);

    std::vector<float> x((size_t)N * d), router((size_t)E * d), Wg((size_t)E * d_ff * d),
        Wu((size_t)E * d_ff * d), Wd((size_t)E * d * d_ff);
    g_rng = 20260329u;
    for (auto& v : x) v = frand_scale(1.0f);
    for (auto& v : router) v = frand_scale(0.05f);
    for (auto& v : Wg) v = frand_scale(1.0f / std::sqrt((float)d));
    for (auto& v : Wu) v = frand_scale(1.0f / std::sqrt((float)d));
    for (auto& v : Wd) v = frand_scale(1.0f / std::sqrt((float)d_ff));

    CUDA_CHECK(cudaMemcpy(B.d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B.d_router, router.data(), router.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B.d_Wg, Wg.data(), Wg.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B.d_Wu, Wu.data(), Wu.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B.d_Wd, Wd.data(), Wd.size() * 4, cudaMemcpyHostToDevice));

    // ---- K1 数值检查：device dot vs host dot（fp32 累加序差异容忍）----
    gate_scores_kernel<<<(N * E + 255) / 256, 256>>>(B.d_x, B.d_router, B.d_logits, N, d, E);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> lg_host((size_t)N * E);
    for (int t = 0; t < N; ++t)
        for (int e = 0; e < E; ++e) {
            float acc = 0.0f;
            for (int i = 0; i < d; ++i)
                acc += x[(size_t)t * d + i] * router[(size_t)e * d + i];
            lg_host[(size_t)t * E + e] = acc;
        }
    std::vector<float> lg_dev((size_t)N * E);
    CUDA_CHECK(cudaMemcpy(lg_dev.data(), B.d_logits, lg_dev.size() * 4, cudaMemcpyDeviceToHost));
    double k1_max = 0.0;
    for (size_t i = 0; i < lg_host.size(); ++i)
        k1_max = std::max(k1_max, (double)std::fabs(lg_dev[i] - lg_host[i]));
    printf("[ch29] K1 gate 数值检查：device vs host logits max_abs=%.3e %s\n",
           k1_max, (k1_max < 1e-3 ? "PASS" : "FAIL"));

    // ---- 上传 host logits（选择语义按构造一致），跑 K2..combine 全链 ----
    CUDA_CHECK(cudaMemcpy(B.d_logits, lg_host.data(), lg_host.size() * 4, cudaMemcpyHostToDevice));
    int threads = 256;
    int P = N * k;
    softmax_topk_kernel<<<N, 32>>>(B.d_logits, B.d_idx, B.d_w, N, E, k);
    CUDA_CHECK(cudaMemset(B.d_counts, 0, 1024 * 4));
    histogram_kernel<<<(P + threads - 1) / threads, threads>>>(B.d_idx, B.d_counts, P);
    exclusive_scan_kernel<<<1, 1024, 1024 * 4>>>(B.d_counts, B.d_offsets, E);
    scatter_fill_kernel<<<(P + threads - 1) / threads, threads>>>(
        B.d_idx, B.d_offsets, B.d_pair_token, B.d_pair_expert, B.d_pair_slot, B.d_pair_pos, N, k);
    gather_pairs_kernel<<<((size_t)P * d + threads - 1) / threads, threads>>>(
        B.d_x, B.d_pair_token, B.d_pair_x, P, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 分段诊断：device 中间产物 vs host 复算（同 logits 同规则）----
    std::vector<int> h_idx(P);
    {
        CUDA_CHECK(cudaMemcpy(h_idx.data(), B.d_idx, P * 4, cudaMemcpyDeviceToHost));
        std::vector<float> h_dw(P);
        CUDA_CHECK(cudaMemcpy(h_dw.data(), B.d_w, P * 4, cudaMemcpyDeviceToHost));
        std::vector<int> h_pp(P);
        CUDA_CHECK(cudaMemcpy(h_pp.data(), B.d_pair_pos, P * 4, cudaMemcpyDeviceToHost));
        std::vector<int> h_pt(P);
        CUDA_CHECK(cudaMemcpy(h_pt.data(), B.d_pair_token, P * 4, cudaMemcpyDeviceToHost));
        // host 复算
        std::vector<int> sel_ref(P);
        std::vector<float> w_ref(P);
        std::vector<int> pos_ref(P);
        for (int t = 0; t < N; ++t) {
            const float* row = &lg_host[(size_t)t * E];
            std::vector<int> taken(E, 0);
            for (int r = 0; r < k; ++r) {
                int best = -1;
                for (int e = 0; e < E; ++e)
                    if (!taken[e] && (best < 0 || row[e] > row[best])) best = e;
                taken[best] = 1;
                sel_ref[(size_t)t * k + r] = best;
            }
            float mx = row[0];
            for (int e = 1; e < E; ++e) mx = std::max(mx, row[e]);
            std::vector<float> pr(E);
            float ssum = 0.0f;
            for (int e = 0; e < E; ++e) { pr[e] = std::exp(row[e] - mx); ssum += pr[e]; }
            float ws = 0.0f;
            for (int r = 0; r < k; ++r) { w_ref[(size_t)t * k + r] = pr[sel_ref[(size_t)t * k + r]] / ssum; ws += w_ref[(size_t)t * k + r]; }
            for (int r = 0; r < k; ++r) w_ref[(size_t)t * k + r] /= ws;
        }
        // host 分桶（stable：桶=专家升序，桶内 pair 索引升序）
        std::vector<int> order(P);
        for (int p = 0; p < P; ++p) order[p] = p;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            int ea = sel_ref[a], eb = sel_ref[b];
            if (ea != eb) return ea < eb;
            return a < b;
        });
        for (int r = 0; r < P; ++r) {
            int p = order[r];
            pos_ref[p] = r;
        }
        int idx_diff = 0; double w_max = 0.0; int pos_diff = 0; int tok_diff = 0;
        for (int p = 0; p < P; ++p) {
            if (h_idx[p] != sel_ref[p]) ++idx_diff;
            w_max = std::max(w_max, (double)std::fabs(h_dw[p] - w_ref[p]));
            if (h_pp[p] != pos_ref[p]) ++pos_diff;
        }
        for (int r = 0; r < P; ++r)
            if (h_pt[r] != sel_ref[order[r]]) ++tok_diff;
        printf("[ch29] 分段自洽：idx_diff=%d w_max=%.3e pos_diff=%d\n", idx_diff, w_max, pos_diff);
    }

    std::vector<int> h_offsets(1024), h_counts(1024);
    CUDA_CHECK(cudaMemcpy(h_offsets.data(), B.d_offsets, 1024 * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), B.d_counts, 1024 * 4, cudaMemcpyDeviceToHost));
    int nonempty = 0, depth_max = 0;
    for (int e = 0; e < E; ++e) {
        if (h_counts[e] > 0) ++nonempty;
        depth_max = std::max(depth_max, h_counts[e]);
    }
    for (int e = 0; e < E; ++e) {
        int s = h_offsets[e];       // exclusive scan：桶 e 起点 = offsets[e]
        int M = h_counts[e];        // 桶 e 深度 = counts[e]（修复：勿混用 CPU inclusive-cumsum 桶界模式）
        if (M <= 0) continue;
        gemm_nt(B.d_pair_x + (size_t)s * d, B.d_Wg + (size_t)e * d_ff * d,
                B.d_gate + (size_t)s * d_ff, M, d_ff, d, 0);
        gemm_nt(B.d_pair_x + (size_t)s * d, B.d_Wu + (size_t)e * d_ff * d,
                B.d_up + (size_t)s * d_ff, M, d_ff, d, 0);
    }
    silu_mul_kernel<<<((unsigned long long)P * d_ff + threads - 1) / threads, threads>>>(
        B.d_gate, B.d_up, B.d_act, (long long)P * d_ff);
    for (int e = 0; e < E; ++e) {
        int s = h_offsets[e];       // exclusive scan：桶 e 起点 = offsets[e]
        int M = h_counts[e];        // 桶 e 深度 = counts[e]（修复：勿混用 CPU inclusive-cumsum 桶界模式）
        if (M <= 0) continue;
        gemm_nt(B.d_act + (size_t)s * d_ff, B.d_Wd + (size_t)e * d * d_ff,
                B.d_down + (size_t)s * d, M, d, d_ff, 0);
    }
    weighted_combine_kernel<<<((size_t)N * d + threads - 1) / threads, threads>>>(
        B.d_down, B.d_w, B.d_pair_pos, B.d_y, N, k, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    // （分段诊断已定位桶界语义 bug 并修复：exclusive scan 桶界 = offsets[e],
    //   深度 = counts[e]；教训见 EXPERIMENT_LOG §CUDA 与 KI-29-08。）

    std::vector<float> y_dev((size_t)N * d);
    CUDA_CHECK(cudaMemcpy(y_dev.data(), B.d_y, y_dev.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<float> y_ref;
    host_reference(x, Wg, Wu, Wd, router, N, d, d_ff, E, k, y_ref);
    double max_abs = 0.0;
    for (size_t i = 0; i < y_dev.size(); ++i)
        max_abs = std::max(max_abs, (double)std::fabs(y_dev[i] - y_ref[i]));
    printf("[ch29] 非空桶=%d/%d depth_max=%d  grouped(device) vs naive(host) max_abs=%.3e %s\n",
           nonempty, E, depth_max, max_abs, (max_abs < 2e-3 ? "PASS" : "FAIL"));
    B.free_all();
    int rc = (max_abs < 2e-3 && k1_max < 1e-3) ? 0 : 1;
    return rc;
}

// ---------------------------------------------------------------------------
// bench：30 格分段计时矩阵 + dense 等价容量（tokens×experts×topk，同 CPU 列口径）
// ---------------------------------------------------------------------------
static void gen_weights_host(int E, int d, int d_ff, std::vector<float>& Wg,
                             std::vector<float>& Wu, std::vector<float>& Wd,
                             std::vector<float>& router) {
    Wg.resize((size_t)E * d_ff * d); Wu.resize((size_t)E * d_ff * d);
    Wd.resize((size_t)E * d * d_ff); router.resize((size_t)E * d);
    for (auto& v : Wg) v = frand_scale(1.0f / std::sqrt((float)d));
    for (auto& v : Wu) v = frand_scale(1.0f / std::sqrt((float)d));
    for (auto& v : Wd) v = frand_scale(1.0f / std::sqrt((float)d_ff));
    for (auto& v : router) v = frand_scale(0.05f);
}

static int run_bench() {
    const int d = 2048, d_ff = 1408;
    const int NS[] = {1, 8, 64, 256, 2048};
    const int ES[] = {8, 64};
    const int KS[] = {2, 6, 8};
    const int WARMUP = 3, REPEATS = 10;
    FILE* out = fopen("results/benchmark-results-cuda.json", "w");
    fprintf(out, "{\n  \"schema_version\": \"kernel-chapter29.v1\",\n"
                 "  \"backend\": \"cuda\",\n  \"device\": \"RTX 4090\",\n"
                 "  \"timing\": \"cudaEvent per-segment, sync-at-end, warmup=3 repeats=10, p50\",\n"
                 "  \"entries\": [\n");
    bool first = true;

    for (int ei = 0; ei < 2; ++ei) {
        int E = ES[ei];
        std::vector<float> Wg, Wu, Wd, router;
        gen_weights_host(E, d, d_ff, Wg, Wu, Wd, router);
        for (int ni = 0; ni < 5; ++ni) {
            int N = NS[ni];
            for (int ki = 0; ki < 3; ++ki) {
                int k = KS[ki];
                DeviceBuf B;
                B.alloc(N, E, k, d, d_ff);
                CUDA_CHECK(cudaMemcpy(B.d_router, router.data(), router.size() * 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(B.d_Wg, Wg.data(), Wg.size() * 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(B.d_Wu, Wu.data(), Wu.size() * 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(B.d_Wd, Wd.data(), Wd.size() * 4, cudaMemcpyHostToDevice));
                std::vector<float> x((size_t)N * d);
                for (auto& v : x) v = frand_scale(1.0f);
                CUDA_CHECK(cudaMemcpy(B.d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));

                // dense 等价容量权重（intermediate = k×d_ff）
                size_t dff_t = (size_t)k * d_ff;
                std::vector<float> dWg(dff_t * d), dWu(dff_t * d), dWd(d * dff_t);
                for (auto& v : dWg) v = frand_scale(1.0f / std::sqrt((float)d));
                for (auto& v : dWu) v = frand_scale(1.0f / std::sqrt((float)d));
                for (auto& v : dWd) v = frand_scale(1.0f / std::sqrt((float)dff_t));
                CUDA_CHECK(cudaMemcpy(B.d_dWg, dWg.data(), dWg.size() * 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(B.d_dWu, dWu.data(), dWu.size() * 4, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(B.d_dWd, dWd.data(), dWd.size() * 4, cudaMemcpyHostToDevice));

                // 桶界预热（分布不变，供分段循环用）
                launch_moe_chain(B, 0);
                std::vector<int> h_offsets(1024), h_counts(1024);
                CUDA_CHECK(cudaMemcpy(h_offsets.data(), B.d_offsets, 1024 * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_counts.data(), B.d_counts, 1024 * 4, cudaMemcpyDeviceToHost));
                int touched = 0, depth_max = 0;
                for (int e = 0; e < E; ++e) {
                    int c = h_counts[e];
                    if (c > 0) ++touched;
                    depth_max = std::max(depth_max, c);
                }

                SegEvents ev;
                ev.create();
                std::vector<std::vector<double>> seg(8);
                std::vector<double> e2e;
                cudaEvent_t ev_start, ev_end;
                CUDA_CHECK(cudaEventCreate(&ev_start));
                CUDA_CHECK(cudaEventCreate(&ev_end));
                for (int r = 0; r < WARMUP + REPEATS; ++r) {
                    bool timing = (r >= WARMUP);
                    CUDA_CHECK(cudaEventRecord(ev_start, 0));
                    run_expert_segments(B, h_offsets, 0, ev, timing);
                    CUDA_CHECK(cudaEventRecord(ev_end, 0));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    if (timing) {
                        for (int s = 0; s < 8; ++s) seg[s].push_back(elapsed_ms(ev.ev[s][0], ev.ev[s][1]));
                        e2e.push_back(elapsed_ms(ev_start, ev_end));
                    }
                }

                // dense 等价容量计时（gate/up 两次 [N,k×d_ff] GEMM + silu·mul + down）
                std::vector<double> dense_times;
                int threads = 256;
                for (int r = 0; r < WARMUP + REPEATS; ++r) {
                    bool timing = (r >= WARMUP);
                    CUDA_CHECK(cudaEventRecord(ev_start, 0));
                    gemm_nt(B.d_x, B.d_dWg, B.d_dm1, N, (int)dff_t, d, 0);
                    gemm_nt(B.d_x, B.d_dWu, B.d_dm2, N, (int)dff_t, d, 0);
                    silu_mul_kernel<<<((unsigned long long)N * (long long)dff_t + threads - 1) / threads, threads, 0, 0>>>(
                        B.d_dm1, B.d_dm2, B.d_dm1, (long long)N * (long long)dff_t);
                    gemm_nt(B.d_dm1, B.d_dWd, B.d_dm3, N, d, (int)dff_t, 0);
                    CUDA_CHECK(cudaEventRecord(ev_end, 0));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    if (timing) dense_times.push_back(elapsed_ms(ev_start, ev_end));
                }

                double e2e_p50 = p50_of(e2e), dense_p50 = p50_of(dense_times);
                double routing = 0.0, expert = 0.0;
                for (int s : {0, 1, 2, 3, 7}) routing += p50_of(seg[s]);
                for (int s : {4, 5, 6}) expert += p50_of(seg[s]);
                if (!first) fprintf(out, ",\n");
                first = false;
                fprintf(out,
                    "    {\"tokens\": %d, \"experts\": %d, \"topk\": %d, \"d_model\": %d, \"d_ff\": %d, "
                    "\"e2e_p50_ms\": %.4f, \"routing_p50_ms\": %.4f, \"expert_p50_ms\": %.4f, "
                    "\"routing_share\": %.4f, \"expert_share\": %.4f, "
                    "\"segments_p50_ms\": {\"gate_mm\": %.4f, \"topk_norm\": %.4f, \"sort_scatter\": %.4f, "
                    "\"gather_x\": %.4f, \"expert_gate_up\": %.4f, \"silu_mul\": %.4f, \"expert_down\": %.4f, "
                    "\"scatter_combine\": %.4f}, "
                    "\"dense_equiv_p50_ms\": %.4f, \"moe_over_dense\": %.4f, "
                    "\"experts_touched\": %d, \"touch_ratio\": %.4f, \"depth_max\": %d, "
                    "\"degenerate_topk_equals_E\": %s}",
                    N, E, k, d, d_ff, e2e_p50, routing, expert,
                    routing / e2e_p50, expert / e2e_p50,
                    p50_of(seg[0]), p50_of(seg[1]), p50_of(seg[2]), p50_of(seg[3]),
                    p50_of(seg[4]), p50_of(seg[5]), p50_of(seg[6]), p50_of(seg[7]),
                    dense_p50, e2e_p50 / dense_p50, touched, (double)touched / E, depth_max,
                    (k >= E) ? "true" : "false");
                printf("[bench] N=%5d E=%2d k=%d | e2e p50=%9.4fms | route=%8.4fms (%5.1f%%) | "
                       "expert=%9.4fms | dense=%9.4fms | moe/dense=%7.2fx | touch=%d/%d depth_max=%d\n",
                       N, E, k, e2e_p50, routing, 100.0 * routing / e2e_p50, expert,
                       dense_p50, e2e_p50 / dense_p50, touched, E, depth_max);
                ev.destroy();
                cudaEventDestroy(ev_start);
                cudaEventDestroy(ev_end);
                B.free_all();
            }
        }
    }
    fprintf(out, "\n  ]\n}\n");
    fclose(out);
    printf("[bench] results/benchmark-results-cuda.json 已写入\n");
    return 0;
}

// ---------------------------------------------------------------------------
// imbalance：均匀 vs 80/20-Zipf 选择注入（E=64, k=6, N∈{256, 2048}）
// ---------------------------------------------------------------------------
static void gen_selections(int N, int E, int k, bool zipf, std::vector<int>& idx) {
    idx.assign((size_t)N * k, 0);
    int H = std::max(1, (int)std::round(0.2 * E));
    std::vector<double> cdf(H);
    double acc = 0.0;
    for (int i = 0; i < H; ++i) { acc += 1.0 / (i + 1); cdf[i] = acc; }
    for (auto& c : cdf) c /= acc;
    for (int t = 0; t < N; ++t) {
        int top1;
        if (zipf && urand() < 0.8) {
            double u = urand();
            top1 = H - 1;
            for (int i = 0; i < H; ++i) if (u < cdf[i]) { top1 = i; break; }
        } else {
            top1 = (int)(urand() * E) % E;
        }
        idx[(size_t)t * k] = top1;
        // 其余 k-1：均匀不放回
        std::vector<int> pool;
        for (int e = 0; e < E; ++e) if (e != top1) pool.push_back(e);
        for (int j = (int)pool.size() - 1; j > 0; --j) {
            int jj = (int)(urand() * (j + 1)) % (j + 1);
            std::swap(pool[j], pool[jj]);
        }
        for (int s = 1; s < k; ++s) idx[(size_t)t * k + s] = pool[s - 1];
    }
}

static int run_imbalance() {
    const int d = 2048, d_ff = 1408, E = 64, k = 6;
    const int NS[] = {256, 2048};
    const int WARMUP = 3, REPEATS = 10;
    FILE* out = fopen("results/load-imbalance-cuda.json", "w");
    fprintf(out, "{\n  \"schema_version\": \"kernel-chapter29.v1\",\n  \"backend\": \"cuda\",\n"
                 "  \"construction\": \"top-1: 0.8×Zipf(13 热专家, α=1) + 0.2×uniform(E=64)；其余 k-1 全池均匀；xorshift32 seed=20260329+N\",\n"
                 "  \"note\": \"选择注入（跳过 K1/K2）：桶界由一遍预热链取 offsets；事件对链=e2e_chain（gather→gate_up→silu→down→combine），expert=e2e_chain−gather−combine 近似前段\",\n"
                 "  \"entries\": [\n");
    bool first = true;

    std::vector<float> Wg, Wu, Wd, router;
    gen_weights_host(E, d, d_ff, Wg, Wu, Wd, router);
    for (int ni = 0; ni < 2; ++ni) {
        int N = NS[ni];
        DeviceBuf B;
        B.alloc(N, E, k, d, d_ff);
        CUDA_CHECK(cudaMemcpy(B.d_router, router.data(), router.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B.d_Wg, Wg.data(), Wg.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B.d_Wu, Wu.data(), Wu.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B.d_Wd, Wd.data(), Wd.size() * 4, cudaMemcpyHostToDevice));
        std::vector<float> x((size_t)N * d);
        for (auto& v : x) v = frand_scale(1.0f);
        CUDA_CHECK(cudaMemcpy(B.d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));

        for (int di = 0; di < 2; ++di) {
            bool zipf = (di == 1);
            g_rng = 20260329u + (unsigned)N;  // 两分布同 base seed：同 token 流
            std::vector<int> idx;
            gen_selections(N, E, k, zipf, idx);
            CUDA_CHECK(cudaMemcpy(B.d_idx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
            {
                std::vector<float> w_uniform((size_t)N * k, 1.0f / k);  // combine 读 d_w：注入常权重
                CUDA_CHECK(cudaMemcpy(B.d_w, w_uniform.data(), w_uniform.size() * 4, cudaMemcpyHostToDevice));
            }
            // 预热拿桶界 + 元数据
            int threads = 256, P = N * k;
            CUDA_CHECK(cudaMemset(B.d_counts, 0, 1024 * 4));
            histogram_kernel<<<(P + threads - 1) / threads, threads>>>(B.d_idx, B.d_counts, P);
            exclusive_scan_kernel<<<1, 1024, 1024 * 4>>>(B.d_counts, B.d_offsets, E);
            scatter_fill_kernel<<<(P + threads - 1) / threads, threads>>>(
                B.d_idx, B.d_offsets, B.d_pair_token, B.d_pair_expert, B.d_pair_slot, B.d_pair_pos, N, k);
            gather_pairs_kernel<<<((size_t)P * d + threads - 1) / threads, threads>>>(
                B.d_x, B.d_pair_token, B.d_pair_x, P, d);
            std::vector<int> h_offsets(1024), h_counts(1024);
            CUDA_CHECK(cudaMemcpy(h_offsets.data(), B.d_offsets, 1024 * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_counts.data(), B.d_counts, 1024 * 4, cudaMemcpyDeviceToHost));
            int depth_max = 0, nonempty = 0;
            std::vector<int> cs(h_counts.begin(), h_counts.begin() + E);
            std::sort(cs.begin(), cs.end(), std::greater<int>());
            for (int e = 0; e < E; ++e) {
                depth_max = std::max(depth_max, h_counts[e]);
                if (h_counts[e] > 0) ++nonempty;
            }
            int depth_p90 = cs[std::min(E - 1, std::max(0, (int)std::round(E * 0.90) - 1))];

            std::vector<double> e2e, expert;
            cudaEvent_t ev_s, ev_e, ex_s, ex_e;
            CUDA_CHECK(cudaEventCreate(&ev_s)); CUDA_CHECK(cudaEventCreate(&ev_e));
            CUDA_CHECK(cudaEventCreate(&ex_s)); CUDA_CHECK(cudaEventCreate(&ex_e));
            for (int r = 0; r < WARMUP + REPEATS; ++r) {
                bool timing = (r >= WARMUP);
                CUDA_CHECK(cudaEventRecord(ev_s, 0));
                CUDA_CHECK(cudaEventRecord(ex_s, 0));
                // gather 起链（选择已注入，K1-K4 不在计时内）
                gather_pairs_kernel<<<((size_t)P * d + threads - 1) / threads, threads, 0, 0>>>(
                    B.d_x, B.d_pair_token, B.d_pair_x, P, d);
                CUDA_CHECK(cudaEventRecord(ex_s, 0));
                for (int e = 0; e < E; ++e) {
                    int s = h_offsets[e];   // exclusive scan：桶 e 起点 = offsets[e]
                    int M = h_counts[e];    // 桶 e 深度 = counts[e]
                    if (M <= 0) continue;
                    gemm_nt(B.d_pair_x + (size_t)s * d, B.d_Wg + (size_t)e * d_ff * d,
                            B.d_gate + (size_t)s * d_ff, M, d_ff, d, 0);
                    gemm_nt(B.d_pair_x + (size_t)s * d, B.d_Wu + (size_t)e * d_ff * d,
                            B.d_up + (size_t)s * d_ff, M, d_ff, d, 0);
                }
                silu_mul_kernel<<<((unsigned long long)P * d_ff + threads - 1) / threads, threads, 0, 0>>>(
                    B.d_gate, B.d_up, B.d_act, (long long)P * d_ff);
                CUDA_CHECK(cudaEventRecord(ex_e, 0));
                for (int e = 0; e < E; ++e) {
                    int s = h_offsets[e];   // exclusive scan：桶 e 起点 = offsets[e]
                    int M = h_counts[e];    // 桶 e 深度 = counts[e]
                    if (M <= 0) continue;
                    gemm_nt(B.d_act + (size_t)s * d_ff, B.d_Wd + (size_t)e * d * d_ff,
                            B.d_down + (size_t)s * d, M, d, d_ff, 0);
                }
                weighted_combine_kernel<<<((size_t)N * d + threads - 1) / threads, threads, 0, 0>>>(
                    B.d_down, B.d_w, B.d_pair_pos, B.d_y, N, k, d);
                CUDA_CHECK(cudaEventRecord(ev_e, 0));
                CUDA_CHECK(cudaDeviceSynchronize());
                if (timing) {
                    e2e.push_back(elapsed_ms(ev_s, ev_e));
                    expert.push_back(elapsed_ms(ex_s, ex_e));
                }
            }
            double e2e_p50 = p50_of(e2e), expert_p50 = p50_of(expert);
            if (!first) fprintf(out, ",\n");
            first = false;
            fprintf(out,
                "    {\"tokens\": %d, \"experts\": %d, \"topk\": %d, \"distribution\": \"%s\", "
                "\"chain_p50_ms\": %.4f, \"expert_p50_ms\": %.4f, "
                "\"depth_max\": %d, \"depth_p90\": %d, \"nonempty\": %d, "
                "\"warmup\": %d, \"repeats\": %d}",
                N, E, k, zipf ? "zipf8020" : "uniform", e2e_p50, expert_p50,
                depth_max, depth_p90, nonempty, WARMUP, REPEATS);
            printf("[imbalance] N=%5d %-9s | chain p50=%9.4fms | expert(gate_up+silu) p50=%9.4fms | "
                   "depth_max=%4d p90=%4d nonempty=%2d\n",
                   N, zipf ? "zipf8020" : "uniform", e2e_p50, expert_p50, depth_max, depth_p90, nonempty);
            cudaEventDestroy(ev_s); cudaEventDestroy(ev_e);
            cudaEventDestroy(ex_s); cudaEventDestroy(ex_e);
        }
        B.free_all();
    }
    fprintf(out, "\n  ]\n}\n");
    fclose(out);
    printf("[imbalance] results/load-imbalance-cuda.json 已写入\n");
    return 0;
}

static int run_gemmtest() {
    // 独立验证 tiled_gemm_kernel：C[M,N] = A[M,K]·B[N,K]^T，M=5,N=128,K=64
    int M = 5, N = 128, K = 64;
    std::vector<float> A((size_t)M * K), B((size_t)N * K), C((size_t)M * N, -999.0f);
    g_rng = 777u;
    for (auto& v : A) v = frand_scale(1.0f);
    for (auto& v : B) v = frand_scale(0.1f);
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, A.size() * 4));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * 4));
    CUDA_CHECK(cudaMalloc(&dC, C.size() * 4));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
    gemm_nt(dA, dB, dC, M, N, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
    double max_abs = 0.0;
    int bad = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float ref = 0.0f;
            for (int kk = 0; kk < K; ++kk) ref += A[(size_t)m * K + kk] * B[(size_t)n * K + kk];
            double d = std::fabs(C[(size_t)m * N + n] - ref);
            if (d > 1e-4) ++bad;
            max_abs = std::max(max_abs, d);
        }
    printf("[gemmtest] max_abs=%.3e bad_cells=%d %s\n", max_abs, bad, (bad == 0 ? "PASS" : "FAIL"));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return (bad == 0) ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "--selftest";
    if (mode == "--gemmtest") return run_gemmtest();
    if (mode == "--selftest") return run_selftest();
    if (mode == "--bench") return run_bench();
    if (mode == "--imbalance") return run_imbalance();
    fprintf(stderr, "usage: moe_route [--selftest|--bench|--imbalance]\n");
    return 2;
}
