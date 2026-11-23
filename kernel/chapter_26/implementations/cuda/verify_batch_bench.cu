// 第 26 期（MTP/投机解码）：verify 批量化收益的 GPU 测量（CUDA 列，待 4090 修复后回填）。
//
// 测量两件事（对应 EXPERIMENT_LOG 实验 5 的 GPU 列）：
//   A. Attention verify：批量核（一次 launch，q 行数 = k+1，第 qr 行 causal
//      上界 N+qr+1）vs 逐 token（k+1 次 launch，每行单独、各自的 causal 上界）。
//      核体 = chapter_20 gqa_decode_kernel 的多行推广（布局与 launch 参数
//      同款：q [B*HQ, q_rows, D] / KV [B*HKV, N+q_rows, D]，GQA kv_group=
//      HQ/HKV，在线 softmax 同 ch21）。FP32 主线（与 CPU 列 dtype 口径一致）。
//   B. LM Head GEMM：M=k+1 一次 GEMM vs M=1 GEMM ×(k+1)（cuBLAS FP32，
//      厂商库口径 = 上界；与 ch18 的 M 维 batch 效应结论对账）。
//
// 计时：每样本一对 cudaEvent、发射完末尾一次同步（ch21 benchmark.py 的
// Python 侧口径一致）；warmup=5、repeats=30；P50/min 入 JSON。
// 形状（Qwen3-0.6B）：D=128, HQ=16, HKV=8；N ∈ {512, 2048, 8192}；k ∈ {2,4,8,16}。
//
// 自检（--selftest）：批量核第 qr 行 == 单行核同参数输出（容差 1e-5，FP32
// 固定分块顺序理论上应位级同值——分块边界一致，在线 softmax 重标定路径一致）。
//
// 构建：bash build.sh（nvcc -O3 -arch=native）。本地无 nvcc——回填步骤见
// CUDA_PENDING.md；运行后把 results/verify-batch-results-cuda.json 拉回本地。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define D 128
#define HQ 16
#define HKV 8
#define THREADS 128
#define BC 128

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(err__), __FILE__, __LINE__);         \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t st__ = (call);                                       \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",                   \
                    (int)st__, __FILE__, __LINE__);                         \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

__device__ __forceinline__ float block_reduce_max(float v, float* scratch, int tid) {
    // 两阶段树形归约（ch21 decode 核同款；独立 scratch 避免背靠背竞争）。
    scratch[tid] = v;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    float result = scratch[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* scratch, int tid) {
    scratch[tid] = v;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float result = scratch[0];
    __syncthreads();
    return result;
}

// 核体（__device__，批量/单行两个 launcher 共用；smem 指针由 launcher 传入）。
__device__ void verify_row_body(
    const float* __restrict__ q_row,
    const float* __restrict__ k,   // [B*HKV, kv_len, D]
    const float* __restrict__ v,
    float* __restrict__ o_row,
    const int head_index, const int total_keys, const int kv_len,
    const float scale, const int kv_group,
    float* q_s, float* acc_s, float* sc, float* red_max, float* red_sum) {
    const int h = head_index % HQ;
    const int kvh = h / kv_group;
    const int tid = static_cast<int>(threadIdx.x);

    q_s[tid] = q_row[tid];
    acc_s[tid] = 0.0f;
    __syncthreads();
    float m = -INFINITY;
    float l = 0.0f;

    for (int c0 = 0; c0 < total_keys; c0 += BC) {
        const int nv = min(BC, total_keys - c0);
        float local_max = -INFINITY;
        for (int c = tid; c < nv; c += THREADS) {
            const float* kp = k + (static_cast<size_t>(kvh) * kv_len + (c0 + c)) * D;
            float dot = 0.0f;
#pragma unroll 8
            for (int d = 0; d < D; ++d) dot += q_s[d] * kp[d];
            sc[c] = dot * scale;
            local_max = fmaxf(local_max, sc[c]);
        }
        const float block_max = block_reduce_max(local_max, red_max, tid);
        float local_sum = 0.0f;
        for (int c = tid; c < nv; c += THREADS) {
            const float p = __expf(sc[c] - block_max);
            sc[c] = p;
            local_sum += p;
        }
        __syncthreads();  // sc 被后续 acc 相全量读取，先同步
        const float block_sum = block_reduce_sum(local_sum, red_sum, tid);
        const float new_m = fmaxf(m, block_max);
        const float alpha = __expf(m - new_m);
        const float factor = block_sum * __expf(block_max - new_m);
        for (int d = tid; d < D; d += THREADS) {
            float acc_d = acc_s[d] * alpha;
            const float* vp = v + (static_cast<size_t>(kvh) * kv_len + c0) * D;
            for (int c = 0; c < nv; ++c) acc_d += sc[c] * vp[static_cast<size_t>(c) * D + d];
            acc_s[d] = acc_d;
        }
        m = new_m;
        l = l * alpha + factor;
        __syncthreads();
    }
    o_row[tid] = acc_s[tid] / l;
}

// launcher A（逐 token 基线）：第 qr 行的一次 launch，grid = (1, B*HQ)，
// q/o 基指针 + 行号，内部按 blockIdx.y（b*HQ+h）推每头地址。
__global__ void gqa_verify_row_kernel(
    const float* __restrict__ q,   // [B*HQ, q_rows, D]
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,         // [B*HQ, q_rows, D]
    const int q_rows, const int qr,
    const int total_keys, const int kv_len, const float scale, const int kv_group) {
    __shared__ float q_s[D];
    __shared__ float acc[D];
    __shared__ float sc[BC];
    __shared__ float red_max[THREADS];
    __shared__ float red_sum[THREADS];
    const size_t offset = static_cast<size_t>(blockIdx.y) * q_rows * D + static_cast<size_t>(qr) * D;
    verify_row_body(q + offset, k, v, o + offset, static_cast<int>(blockIdx.y),
                    total_keys, kv_len, scale, kv_group, q_s, acc, sc, red_max, red_sum);
}

// launcher B（批量 verify）：一次 launch，grid = (q_rows, B*HQ)。
// blockIdx.x = query 行（0..q_rows-1），每行 causal 上界 N+qr+1 —— MTP verify
// 的真实形状：k+1 个候选行在同一个 launch 里并行验证。
__global__ void gqa_verify_batched_kernel(
    const float* __restrict__ q,   // [B*HQ, q_rows, D]
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,         // [B*HQ, q_rows, D]
    const int q_rows, const int n, const float scale, const int kv_group) {
    __shared__ float q_s[D];
    __shared__ float acc[D];
    __shared__ float sc[BC];
    __shared__ float red_max[THREADS];
    __shared__ float red_sum[THREADS];
    const int qr = static_cast<int>(blockIdx.x);
    verify_row_body(q + static_cast<size_t>(blockIdx.y) * q_rows * D + static_cast<size_t>(qr) * D,
                    k, v, o + static_cast<size_t>(blockIdx.y) * q_rows * D + static_cast<size_t>(qr) * D,
                    static_cast<int>(blockIdx.y), n + qr + 1, n + q_rows, scale, kv_group,
                    q_s, acc, sc, red_max, red_sum);
}

struct Timing {
    std::vector<float> samples_ms;
    float p50_ms = 0.0f;
    float min_ms = 0.0f;
};

// 逐样本事件对 + 末尾一次同步（ch21 口径）。
Timing time_pairs(const std::function<void(cudaStream_t)>& launch, cudaStream_t stream,
                  int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<cudaEvent_t> starts(repeats), stops(repeats);
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaEventCreate(&starts[i]));
        CUDA_CHECK(cudaEventCreate(&stops[i]));
    }
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaEventRecord(starts[i], stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stops[i], stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    Timing timing;
    timing.samples_ms.resize(repeats);
    for (int i = 0; i < repeats; ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&timing.samples_ms[i], starts[i], stops[i]));
        CUDA_CHECK(cudaEventDestroy(starts[i]));
        CUDA_CHECK(cudaEventDestroy(stops[i]));
    }
    std::vector<float> ordered = timing.samples_ms;
    std::sort(ordered.begin(), ordered.end());
    timing.p50_ms = ordered[ordered.size() / 2];
    timing.min_ms = ordered[0];
    return timing;
}

static void fill_random(std::vector<float>& host, unsigned seed) {
    srand(seed);
    for (auto& x : host) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
}

// --selftest：批量核第 qr 行输出 == 单行核同参数输出（max_abs <= 1e-5）。
static int selftest(cudaStream_t stream) {
    const int b = 1;
    const int kv_group = HQ / HKV;
    const float scale = 1.0f / sqrtf(static_cast<float>(D));
    const int n = 64;
    const int q_rows = 4;  // k = 3
    const int kv_len = n + q_rows;
    std::vector<float> h_q(static_cast<size_t>(b * HQ) * q_rows * D);
    std::vector<float> h_kv(static_cast<size_t>(b * HKV * 2) * kv_len * D);
    fill_random(h_q, 2027u);
    fill_random(h_kv, 7000u);
    float *q_d, *kv_d, *o_batched_d, *o_single_d;
    CUDA_CHECK(cudaMalloc(&q_d, h_q.size() * 4));
    CUDA_CHECK(cudaMalloc(&kv_d, h_kv.size() * 4));
    CUDA_CHECK(cudaMalloc(&o_batched_d, h_q.size() * 4));
    CUDA_CHECK(cudaMalloc(&o_single_d, h_q.size() * 4));
    CUDA_CHECK(cudaMemcpy(q_d, h_q.data(), h_q.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kv_d, h_kv.data(), h_kv.size() * 4, cudaMemcpyHostToDevice));

    const float* v_d = kv_d + static_cast<size_t>(b * HKV) * kv_len * D;
    gqa_verify_batched_kernel<<<dim3(q_rows, b * HQ), THREADS, 0, stream>>>(
        q_d, kv_d, v_d, o_batched_d, q_rows, n, scale, kv_group);
    // 单行核：逐行独立计算（同参数；grid.y 覆盖全部头）。
    for (int qr = 0; qr < q_rows; ++qr) {
        gqa_verify_row_kernel<<<dim3(1, b * HQ), THREADS, 0, stream>>>(
            q_d, kv_d, v_d, o_single_d, q_rows, qr,
            n + qr + 1, kv_len, scale, kv_group);
    }
    std::vector<float> out_a(h_q.size()), out_b(h_q.size());
    CUDA_CHECK(cudaMemcpy(out_a.data(), o_batched_d, out_a.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_b.data(), o_single_d, out_b.size() * 4, cudaMemcpyDeviceToHost));
    float max_abs = 0.0f;
    for (size_t i = 0; i < out_a.size(); ++i) max_abs = fmaxf(max_abs, fabsf(out_a[i] - out_b[i]));
    printf("[selftest] batched-vs-single row max_abs = %.3e -> %s\n", max_abs,
           max_abs <= 1e-5f ? "PASS" : "FAIL");
    CUDA_CHECK(cudaFree(q_d));
    CUDA_CHECK(cudaFree(kv_d));
    CUDA_CHECK(cudaFree(o_batched_d));
    CUDA_CHECK(cudaFree(o_single_d));
    return max_abs <= 1e-5f ? 0 : 1;
}

int main(int argc, char** argv) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (argc > 1 && strcmp(argv[1], "--selftest") == 0) {
        return selftest(stream);
    }

    const int b = 1;
    const int kv_group = HQ / HKV;
    const float scale = 1.0f / sqrtf(static_cast<float>(D));
    const std::vector<int> n_sweep = {512, 2048, 8192};
    const std::vector<int> k_sweep = {2, 4, 8, 16};
    const int warmup = 5;
    const int repeats = 30;

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    // LM Head 权重（hidden=1024 × V=151936 FP32，约 595MB，一次分配常驻）。
    const int v_dim = 151936;
    const int hidden = 1024;
    std::vector<float> h_w(static_cast<size_t>(hidden) * v_dim);
    fill_random(h_w, 12345u);
    float *w_d = nullptr, *x_d = nullptr, *logits_d = nullptr;
    CUDA_CHECK(cudaMalloc(&w_d, h_w.size() * 4));
    CUDA_CHECK(cudaMemcpyAsync(w_d, h_w.data(), h_w.size() * 4, cudaMemcpyHostToDevice, stream));
    std::vector<float>().swap(h_w);  // 释放 host 侧
    const int max_m = 17;            // k=16 → M=k+1=17
    CUDA_CHECK(cudaMalloc(&x_d, static_cast<size_t>(max_m) * hidden * 4));
    CUDA_CHECK(cudaMalloc(&logits_d, static_cast<size_t>(max_m) * v_dim * 4));
    {
        std::vector<float> h_x(static_cast<size_t>(max_m) * hidden);
        fill_random(h_x, 54321u);
        CUDA_CHECK(cudaMemcpyAsync(x_d, h_x.data(), h_x.size() * 4, cudaMemcpyHostToDevice, stream));
    }
    const float alpha = 1.0f, beta = 0.0f;
    // C_rm[M,V] = x_rm[M,hidden] × W_rm[hidden,V] 的 cuBLAS 列主序写法：
    // C_cm[V,M] = W_cm[V,hidden] × x_cm[hidden,M]，全 OP_N（见文件头注释推导）。
    auto lmhead_batched = [&](cudaStream_t s, int m) {
        (void)s;
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, v_dim, m, hidden,
                                 &alpha, w_d, v_dim, x_d, hidden, &beta, logits_d, v_dim));
    };

    printf("{\n  \"schema\": \"kernel-chapter27.verify-batch-cuda.v1\",\n");
    printf("  \"episode\": 27, \"backend\": \"cuda\", \"dtype\": \"float32\",\n");
    printf("  \"attention_shape\": {\"D\": %d, \"HQ\": %d, \"HKV\": %d}, \"warmup\": %d, \"repeats\": %d,\n",
           D, HQ, HKV, warmup, repeats);
    printf("  \"rows\": [\n");
    bool first_row = true;
    for (int n : n_sweep) {
        for (int k : k_sweep) {
            const int q_rows = k + 1;
            const int kv_len = n + q_rows;
            // K 与 V 共用一块连续分配（前半 K、后半 V），计数必须 ×2——
            // 本 bug 曾在首轮回填时触发 illegal memory access（selftest 的
            // h_kv 手工 ×2 故未暴露）：修复于 2026-11-22 CUDA 回填会话。
            const size_t kv_count = static_cast<size_t>(b * HKV) * 2 * kv_len * D;
            std::vector<float> h_q(static_cast<size_t>(b * HQ) * q_rows * D);
            std::vector<float> h_kv(kv_count);
            fill_random(h_q, 2027u + static_cast<unsigned>(n + k));
            fill_random(h_kv, 7000u + static_cast<unsigned>(n + k));
            float *q_d, *kv_d, *o_d;
            CUDA_CHECK(cudaMalloc(&q_d, h_q.size() * 4));
            CUDA_CHECK(cudaMalloc(&kv_d, h_kv.size() * 4));
            CUDA_CHECK(cudaMalloc(&o_d, h_q.size() * 4));
            CUDA_CHECK(cudaMemcpyAsync(q_d, h_q.data(), h_q.size() * 4, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(kv_d, h_kv.data(), h_kv.size() * 4, cudaMemcpyHostToDevice, stream));
            const float* v_d = kv_d + static_cast<size_t>(b * HKV) * kv_len * D;

            // A1: 批量 verify——一次 launch，grid = (q_rows, B*HQ)。
            Timing batch_t = time_pairs(
                [&](cudaStream_t s) {
                    gqa_verify_batched_kernel<<<dim3(q_rows, b * HQ), THREADS, 0, s>>>(
                        q_d, kv_d, v_d, o_d, q_rows, n, scale, kv_group);
                },
                stream, warmup, repeats);

            // A2: 逐 token 基线——q_rows 次单行 launch（投机解码不做 verify
            // 批量化时的等价成本结构：k+1 次 decode 步），每行 grid.y 覆盖
            // 全部头、q/o 按行内基指针偏移（与批量核同数据）。
            Timing seq_t = time_pairs(
                [&](cudaStream_t s) {
                    for (int qr = 0; qr < q_rows; ++qr) {
                        gqa_verify_row_kernel<<<dim3(1, b * HQ), THREADS, 0, s>>>(
                            q_d, kv_d, v_d, o_d, q_rows, qr,
                            n + qr + 1, kv_len, scale, kv_group);
                    }
                },
                stream, warmup, repeats);

            // B: LM Head GEMM：M=k+1 一次 vs M=1 ×(k+1)。
            Timing gemm_batch_t = time_pairs(
                [&](cudaStream_t s) {
                    (void)s;
                    lmhead_batched(s, q_rows);
                },
                stream, warmup, repeats);
            Timing gemm_seq_t = time_pairs(
                [&](cudaStream_t s) {
                    (void)s;
                    for (int m = 0; m < q_rows; ++m) {
                        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, v_dim, 1, hidden,
                                    &alpha, w_d, v_dim, x_d + static_cast<size_t>(m) * hidden,
                                    hidden, &beta, logits_d + static_cast<size_t>(m) * v_dim, v_dim);
                    }
                },
                stream, warmup, repeats);

            if (!first_row) printf(",\n");
            first_row = false;
            printf(
                "    {\"n\": %d, \"k\": %d, \"q_rows\": %d, "
                "\"attn_batched_p50_ms\": %.4f, \"attn_batched_min_ms\": %.4f, "
                "\"attn_sequential_p50_ms\": %.4f, \"attn_batching_speedup\": %.4f, "
                "\"lmhead_batched_p50_ms\": %.4f, \"lmhead_sequential_p50_ms\": %.4f, "
                "\"lmhead_batching_speedup\": %.4f}",
                n, k, q_rows, batch_t.p50_ms, batch_t.min_ms, seq_t.p50_ms,
                seq_t.p50_ms > 0 ? seq_t.p50_ms / batch_t.p50_ms : 0.0,
                gemm_batch_t.p50_ms, gemm_seq_t.p50_ms,
                gemm_seq_t.p50_ms > 0 ? gemm_seq_t.p50_ms / gemm_batch_t.p50_ms : 0.0);
            fflush(stdout);

            CUDA_CHECK(cudaFree(q_d));
            CUDA_CHECK(cudaFree(kv_d));
            CUDA_CHECK(cudaFree(o_d));
        }
        printf("\n");
    }
    printf("  ]\n}\n");
    CUDA_CHECK(cudaFree(w_d));
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(logits_d));
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
