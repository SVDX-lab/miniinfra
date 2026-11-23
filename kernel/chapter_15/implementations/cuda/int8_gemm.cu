// ============================================================================
// 第 15 期：量化算子与反量化融合——per-channel INT8 weight-only GEMM
// ============================================================================
//
// 数学语义：C[M,N] = A_fp16[M,K] · dequant(W_int8[K,N], scale[N])
//   scale[n] = max_k|W[k,n]| / 127（per-channel symmetric，见 CPU 参考）
//   "改变权重的表示（FP16 → INT8 + scale），数学语义不变"（大纲第 15 期思想）；
//   "省访存"（读 1 字节/权重）与"反量化插在哪里"（Kernel 内融合 vs 预物化）
//   是两个独立的收益来源——本二进制把两条路径都做成同结构 tiled kernel。
//
// Kernel 组织（结构参照第 17 期 naive GEMM：单层 tiling + 片上缓冲 + 寄存器分块）：
//   1. gemm_fp16_tiled_kernel    —— A(fp16) × W(fp16)：物化反量化后的 FP16 GEMM
//                                   （分离路径的稳态；物化本身是一次性成本，
//                                    由 dequant_materialize_kernel 单独计时）；
//   2. gemm_int8_fused_kernel    —— A(fp16) × W(int8)：B-tile 协作加载时
//                                   load int8 → 转 float → × scale[n] → 写入
//                                   shared memory 的 float tile，之后与 FP16
//                                   版逐指令相同——唯一差异 = 权重字节数与
//                                   tile 加载端的反量化算术（融合路径）；
//   3. dequant_materialize_kernel —— 物化反量化（int8×scale → fp16），一次性
//                                   权重加载成本（计入"加载期"账本，不计入
//                                   稳态 decode 延迟——这正是两条路径的
//                                   账本口径差异，EXPERIMENT_LOG 分开记录）；
//   4. decodeblock 模式           —— Qwen3-0.6B 单 Transformer Block 的 6 个
//                                   投影 GEMM 背靠背链（Decode 单步端到端代理）。
//
// 误差语义注记（误差分层实验的实现基础）：
//   - 融合路径反量化在 FP32 算术（(float)q × scale，权重不落回 FP16 存储）；
//   - 物化路径反量化产物四舍五入回 FP16 存储（Ŵ16 = half(q×scale)）；
//   两者的差 = "FP16 权重存储舍入"一项，与 CPU 侧 La1/La2 分层一一对应。
//
// MUSA 迁移：与第 17 期同一模式——全部 API 差异集中在文件头 USE_MUSA shim，
// musa/int8_gemm.mu 只是 2 行入口包装（#define USE_MUSA 1 + include 本文件，
// M-43 教训：入口 .mu 必须自带 define，不能依赖外部 -D）。
// 资源预算：64x64x8 配置静态 smem = (64×9 + 8×65)×4 = 4384B ≪ 28,672B
// （MUSA 静态上限，M-43）；最大 block 1024 线程（64x64x8_reg2x2）。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#if defined(USE_MUSA)
  #include <musa_runtime.h>
  #if defined(__has_include)
    #if __has_include(<musa_fp16.h>)
      #include <musa_fp16.h>
      #define HAVE_MUSA_FP16_H 1
    #endif
  #endif
  // ---- GPU API 兼容 shim：CUDA 命名 → MUSA 命名（机械替换单点化，ch18 同款）----
  #define gpuError_t          musaError_t
  #define gpuSuccess          musaSuccess
  #define gpuGetErrorString   musaGetErrorString
  #define gpuGetLastError     musaGetLastError
  #define gpuDeviceProp_t     musaDeviceProp
  #define gpuGetDeviceProperties musaGetDeviceProperties
  #define gpuMalloc           musaMalloc
  #define gpuFree             musaFree
  #define gpuMemcpy           musaMemcpy
  #define gpuMemcpyHostToHost musaMemcpyHostToHost
  #define gpuMemcpyHostToDevice musaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost musaMemcpyDeviceToHost
  #define gpuEvent_t          musaEvent_t
  #define gpuEventCreate      musaEventCreate
  #define gpuEventDestroy     musaEventDestroy
  #define gpuEventRecord      musaEventRecord
  #define gpuEventSynchronize musaEventSynchronize
  #define gpuEventElapsedTime musaEventElapsedTime
  #define gpuDeviceSynchronize musaDeviceSynchronize
  #define gpuFuncAttributes   musaFuncAttributes
  #define gpuFuncGetAttributes musaFuncGetAttributes
#else
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  #define gpuError_t          cudaError_t
  #define gpuSuccess          cudaSuccess
  #define gpuGetErrorString   cudaGetErrorString
  #define gpuGetLastError     cudaGetLastError
  #define gpuDeviceProp_t     cudaDeviceProp
  #define gpuGetDeviceProperties cudaGetDeviceProperties
  #define gpuMalloc           cudaMalloc
  #define gpuFree             cudaFree
  #define gpuMemcpy           cudaMemcpy
  #define gpuMemcpyHostToHost cudaMemcpyHostToHost
  #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
  #define gpuEvent_t          cudaEvent_t
  #define gpuEventCreate      cudaEventCreate
  #define gpuEventDestroy     cudaEventDestroy
  #define gpuEventRecord      cudaEventRecord
  #define gpuEventSynchronize cudaEventSynchronize
  #define gpuEventElapsedTime cudaEventElapsedTime
  #define gpuDeviceSynchronize cudaDeviceSynchronize
  #define gpuFuncAttributes   cudaFuncAttributes
  #define gpuFuncGetAttributes cudaFuncGetAttributes
#endif

// FP16 类型（ch18 同款：musa_fp16.h 可用则用 __half，否则位级 RNE fallback）
#if !defined(USE_MUSA) || defined(HAVE_MUSA_FP16_H)
  #define KERNEL_HALF_TYPE __half
#endif
#if defined(USE_MUSA) && !defined(HAVE_MUSA_FP16_H)
  #define KERNEL_HALF_TYPE gpu_half_bits
#endif

// ---------------------------------------------------------------------------
// 自实现 binary16（IEEE 754 RNE）——MUSA 平台边界 fallback（ch18 原样）
// ---------------------------------------------------------------------------
#if defined(USE_MUSA) && !defined(HAVE_MUSA_FP16_H)
struct gpu_half_bits {
    unsigned short bits;
};

__host__ __device__ inline unsigned short float_to_half_rne(float value) {
    unsigned int bits;
    memcpy(&bits, &value, sizeof(bits));
    unsigned int sign = (bits >> 16) & 0x8000u;
    unsigned int exp32 = (bits >> 23) & 0xFFu;
    unsigned int mant = bits & 0x7FFFFFu;
    unsigned short out;
    if (exp32 == 0xFF) {
        out = (unsigned short)(mant ? 0xFE00u : 0x7C00u);
    } else if (exp32 == 0 && mant == 0) {
        out = 0;
    } else {
        int e = (int)exp32 - 127;
        if (e > 15) {
            out = 0x7C00u;
        } else if (e >= -14) {
            unsigned int hm = mant >> 13;
            unsigned int rem = mant & 0x1FFFu;
            unsigned int h = ((unsigned int)(e + 15) << 10) | hm;
            if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) h += 1;
            out = (unsigned short)h;
        } else if (e >= -25) {
            unsigned int full = 0x800000u | mant;
            int shift = -e - 1;
            if (shift > 24) {
                out = 0;
            } else {
                unsigned int s = full >> shift;
                unsigned int rem = full & ((1u << shift) - 1u);
                unsigned int half_point = 1u << (shift - 1);
                if (rem > half_point || (rem == half_point && (s & 1u))) s += 1;
                out = (unsigned short)s;
            }
        } else {
            out = 0;
        }
    }
    return (unsigned short)(sign | out);
}
#endif

// ---------------------------------------------------------------------------
// 统一转换入口（half 主线；int8→float 为 C++ 内建转换）
// ---------------------------------------------------------------------------
template <typename T>
__device__ __host__ inline T to_type(float v);
template <typename T>
__device__ __host__ inline float to_float_val(T v);

template <>
__device__ __host__ inline float to_type<float>(float v) { return v; }
template <>
__device__ __host__ inline float to_float_val<float>(float v) { return v; }

#if defined(USE_MUSA)
#if defined(HAVE_MUSA_FP16_H)
template <>
__device__ __host__ inline __half to_type<__half>(float v) { return __float2half(v); }
template <>
__device__ __host__ inline float to_float_val<__half>(__half v) { return __half2float(v); }
#else
template <>
__device__ __host__ inline gpu_half_bits to_type<gpu_half_bits>(float v) {
    gpu_half_bits h;
    h.bits = float_to_half_rne(v);
    return h;
}
template <>
__device__ __host__ inline float to_float_val<gpu_half_bits>(gpu_half_bits v) {
    unsigned short h = v.bits;
    unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            unsigned int m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; e -= 1; }
            m &= 0x3FFu;
            bits = sign | ((unsigned int)(127 + e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}
#endif
#else
template <>
__device__ __host__ inline __half to_type<__half>(float v) { return __float2half(v); }
template <>
__device__ __host__ inline float to_float_val<__half>(__half v) { return __half2float(v); }
#endif

// ---------------------------------------------------------------------------
// 错误检查（ch18 原样）
// ---------------------------------------------------------------------------
#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        gpuError_t err_ = (call);                                              \
        if (err_ != gpuSuccess) {                                              \
            fprintf(stderr, "[GPU-ERROR] %s 失败 (%s:%d): %s\n", #call,        \
                    __FILE__, __LINE__, gpuGetErrorString(err_));              \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

static void check_launch(const char* where) {
    gpuError_t err = gpuGetLastError();
    if (err != gpuSuccess) {
        fprintf(stderr, "[GPU-ERROR] launch %s: %s\n", where, gpuGetErrorString(err));
        exit(2);
    }
}

// ---------------------------------------------------------------------------
// Kernel 1：FP16 GEMM（物化路径稳态）——ch18 gemm_tiled_kernel 的 ep16 版
// （无 epilogue；A/W 输入 half，FP32 累加，C 输出 half）
// ---------------------------------------------------------------------------
template <int TILE_M, int TILE_N, int TILE_K, int REG_M, int REG_N>
__global__ void gemm_fp16_tiled_kernel(const KERNEL_HALF_TYPE* __restrict__ A,
                                       const KERNEL_HALF_TYPE* __restrict__ W,
                                       KERNEL_HALF_TYPE* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int nthreads = blockDim.x * blockDim.y;
    const int tid = ty * blockDim.x + tx;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float acc[REG_M][REG_N];
#pragma unroll
    for (int m = 0; m < REG_M; ++m)
#pragma unroll
        for (int n = 0; n < REG_N; ++n) acc[m][n] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        for (int idx = tid; idx < TILE_M * TILE_K; idx += nthreads) {
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int gm = m0 + r;
            int gk = k0 + c;
            As[r][c] = (gm < M && gk < K) ? to_float_val(A[(size_t)gm * K + gk]) : 0.0f;
        }
        for (int idx = tid; idx < TILE_K * TILE_N; idx += nthreads) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gk = k0 + r;
            int gn = n0 + c;
            Bs[r][c] = (gk < K && gn < N) ? to_float_val(W[(size_t)gk * N + gn]) : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_reg[REG_M];
            float b_reg[REG_N];
#pragma unroll
            for (int m = 0; m < REG_M; ++m) a_reg[m] = As[ty * REG_M + m][kk];
#pragma unroll
            for (int n = 0; n < REG_N; ++n) b_reg[n] = Bs[kk][tx * REG_N + n];
#pragma unroll
            for (int m = 0; m < REG_M; ++m)
#pragma unroll
                for (int n = 0; n < REG_N; ++n) acc[m][n] += a_reg[m] * b_reg[n];
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < REG_M; ++m) {
        int gm = m0 + ty * REG_M + m;
#pragma unroll
        for (int n = 0; n < REG_N; ++n) {
            int gn = n0 + tx * REG_N + n;
            if (gm < M && gn < N) C[(size_t)gm * N + gn] = to_type<KERNEL_HALF_TYPE>(acc[m][n]);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2：INT8 weight-only 融合反量化 GEMM（融合路径，本期主实现）
//
// 与 Kernel 1 唯一差异在 B-tile 协作加载：
//   Bs[r][c] = (float)W_int8[gk*N+gn] * scale[gn]   ← 反量化融合在加载端
// 之后每条 FMA 指令与 FP16 版逐指令相同。所以：
//   - 时间差 = 权重字节减半（访存） + 加载端转换算术（int8→float + 乘法）
//     —— "省访存"与"反量化算术开销"在这个结构里被放在一起，归因靠
//       decode 档（权重主导）与 prefill 档（激活/输出占比上升）的对照；
//   - 与"物化路径"的语义差 = 融合路径的 dequant 产物不落回 FP16 存储
//     （FP32 精度直送累加），物化路径多一次 FP16 存储舍入。
// ---------------------------------------------------------------------------
template <int TILE_M, int TILE_N, int TILE_K, int REG_M, int REG_N>
__global__ void gemm_int8_fused_kernel(const KERNEL_HALF_TYPE* __restrict__ A,
                                       const int8_t* __restrict__ W,
                                       const float* __restrict__ scale,
                                       KERNEL_HALF_TYPE* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int nthreads = blockDim.x * blockDim.y;
    const int tid = ty * blockDim.x + tx;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float acc[REG_M][REG_N];
#pragma unroll
    for (int m = 0; m < REG_M; ++m)
#pragma unroll
        for (int n = 0; n < REG_N; ++n) acc[m][n] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        for (int idx = tid; idx < TILE_M * TILE_K; idx += nthreads) {
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int gm = m0 + r;
            int gk = k0 + c;
            As[r][c] = (gm < M && gk < K) ? to_float_val(A[(size_t)gm * K + gk]) : 0.0f;
        }
        for (int idx = tid; idx < TILE_K * TILE_N; idx += nthreads) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gk = k0 + r;
            int gn = n0 + c;
            Bs[r][c] = (gk < K && gn < N)
                           ? ((float)W[(size_t)gk * N + gn]) * scale[gn]
                           : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_reg[REG_M];
            float b_reg[REG_N];
#pragma unroll
            for (int m = 0; m < REG_M; ++m) a_reg[m] = As[ty * REG_M + m][kk];
#pragma unroll
            for (int n = 0; n < REG_N; ++n) b_reg[n] = Bs[kk][tx * REG_N + n];
#pragma unroll
            for (int m = 0; m < REG_M; ++m)
#pragma unroll
                for (int n = 0; n < REG_N; ++n) acc[m][n] += a_reg[m] * b_reg[n];
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < REG_M; ++m) {
        int gm = m0 + ty * REG_M + m;
#pragma unroll
        for (int n = 0; n < REG_N; ++n) {
            int gn = n0 + tx * REG_N + n;
            if (gm < M && gn < N) C[(size_t)gm * N + gn] = to_type<KERNEL_HALF_TYPE>(acc[m][n]);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3：物化反量化（一次性权重加载成本）：out[k*N+n] = half(q × scale[n])
// v1：1D 网格 + size_t 取模（首测 21 GB/s，S4000 上异常——诊断变体见 v2/v3）
// ---------------------------------------------------------------------------
__global__ void dequant_materialize_kernel(const int8_t* __restrict__ q,
                                           const float* __restrict__ scale,
                                           KERNEL_HALF_TYPE* __restrict__ out, size_t total,
                                           int N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int n = (int)(idx % (size_t)N);
        out[idx] = to_type<KERNEL_HALF_TYPE>((float)q[idx] * scale[n]);
    }
}

// v2：2D 网格（y=行 k，x=列段）——n 直接由 blockIdx.x 推出，无取模；
// 64 位算术只保留行基址。诊断目的：分离"取模"与"1 字节访存"。
__global__ void dequant_materialize_v2(const int8_t* __restrict__ q,
                                       const float* __restrict__ scale,
                                       KERNEL_HALF_TYPE* __restrict__ out, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        size_t idx = (size_t)blockIdx.y * N + n;
        out[idx] = to_type<KERNEL_HALF_TYPE>((float)q[idx] * scale[n]);
    }
}

// v3：32 位线性索引 + 32 位取模——与 v1 只差"64 位取模 vs 32 位取模"。
__global__ void dequant_materialize_v3(const int8_t* __restrict__ q,
                                       const float* __restrict__ scale,
                                       KERNEL_HALF_TYPE* __restrict__ out, unsigned int total,
                                       unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        unsigned int n = idx % N;
        out[idx] = to_type<KERNEL_HALF_TYPE>((float)q[idx] * scale[n]);
    }
}

// ---------------------------------------------------------------------------
// 随机填充 kernel（bench 模式数据；确定性 hash，ch18 同款 LCG）
// ---------------------------------------------------------------------------
__host__ __device__ inline unsigned int hash_u32(size_t i, unsigned long long seed) {
    unsigned int x = (unsigned int)(i * 2654435761ULL) ^ (unsigned int)(seed & 0xFFFFFFFFu) ^
                     (unsigned int)((seed >> 32) & 0xFFFFu);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

__global__ void fill_random_float(float* out, size_t count, unsigned long long seed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = ((float)hash_u32(idx, seed) / 2147483648.0f) - 1.0f;
    }
}

__global__ void fill_random_half(KERNEL_HALF_TYPE* out, size_t count, unsigned long long seed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = to_type<KERNEL_HALF_TYPE>(((float)hash_u32(idx, seed) / 2147483648.0f) - 1.0f);
    }
}

__global__ void fill_random_int8(int8_t* out, size_t count, unsigned long long seed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = (int8_t)((int)(hash_u32(idx, seed) % 255u) - 127);
    }
}

// scale 填充：正值 [0.004, 0.012)——量级贴近 1/127 量级的真实 per-channel scale
__global__ void fill_scale(float* out, size_t count, unsigned long long seed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = 0.004f + ((float)(hash_u32(idx, seed) >> 8) / 8388608.0f) * 0.004f;
    }
}

// ---------------------------------------------------------------------------
// 配置注册表（5 配置 × 2 kernel = 10 个 GEMM 实例；smem 最大 4384B ≪ M-43 上限）
// ---------------------------------------------------------------------------
struct Config {
    const char* name;
    int tm, tn, tk;
    int rm, rn;
};

static const std::vector<Config>& config_table() {
    static const std::vector<Config> table = {
        {"16x16x16_reg2x2", 16, 16, 16, 2, 2},
        {"32x32x8_reg2x2", 32, 32, 8, 2, 2},
        {"32x32x8_reg4x4", 32, 32, 8, 4, 4},
        {"64x64x8_reg2x2", 64, 64, 8, 2, 2},
        {"64x64x8_reg4x4", 64, 64, 8, 4, 4},
    };
    return table;
}

static const Config* find_config(const std::string& tile, const std::string& reg) {
    std::string want = tile + "_reg" + reg;
    for (const auto& c : config_table())
        if (want == c.name) return &c;
    return nullptr;
}

static size_t smem_bytes_of(const Config& c) {
    return (size_t)(c.tm * (c.tk + 1) + c.tk * (c.tn + 1)) * sizeof(float);
}

static dim3 grid_of(const Config& c, int m, int n) {
    return dim3((unsigned)((n + c.tn - 1) / c.tn), (unsigned)((m + c.tm - 1) / c.tm));
}
static dim3 block_of(const Config& c) {
    return dim3((unsigned)(c.tn / c.rn), (unsigned)(c.tm / c.rm));
}

// ---------------------------------------------------------------------------
// 分发（模板实例化表）
// ---------------------------------------------------------------------------
template <int TM, int TN, int TK, int RM, int RN>
static bool launch_pair(const Config& cfg, int which, dim3 grid, dim3 block,
                        const KERNEL_HALF_TYPE* dA, const KERNEL_HALF_TYPE* dWf,
                        const int8_t* dWq, const float* dScale, KERNEL_HALF_TYPE* dC, int M, int N,
                        int K) {
    if (cfg.tm != TM || cfg.tn != TN || cfg.tk != TK || cfg.rm != RM || cfg.rn != RN) return false;
    if (which == 0) {
        gemm_fp16_tiled_kernel<TM, TN, TK, RM, RN>
            <<<grid, block>>>(dA, dWf, dC, M, N, K);
        return true;
    }
    if (which == 1) {
        gemm_int8_fused_kernel<TM, TN, TK, RM, RN>
            <<<grid, block>>>(dA, dWq, dScale, dC, M, N, K);
        return true;
    }
    return false;
}

static void launch_gemm(const Config& cfg, int which, const KERNEL_HALF_TYPE* dA,
                        const KERNEL_HALF_TYPE* dWf, const int8_t* dWq, const float* dScale,
                        KERNEL_HALF_TYPE* dC, int M, int N, int K) {
    dim3 grid = grid_of(cfg, M, N);
    dim3 block = block_of(cfg);
    bool ok = false;
#define L(TM, TN, TK, RM, RN) ok = launch_pair<TM, TN, TK, RM, RN>(cfg, which, grid, block, dA, dWf, dWq, dScale, dC, M, N, K) || ok;
    L(16, 16, 16, 2, 2);
    L(32, 32, 8, 2, 2);
    L(32, 32, 8, 4, 4);
    L(64, 64, 8, 2, 2);
    L(64, 64, 8, 4, 4);
#undef L
    if (!ok) {
        fprintf(stderr, "[GPU-ERROR] 未实例化的配置 %s\n", cfg.name);
        exit(2);
    }
    check_launch("gemm launch");
}

// ---------------------------------------------------------------------------
// 计时（ch18 同款：warmup 后逐样本事件对，末尾一次同步）
// ---------------------------------------------------------------------------
struct TimingStats {
    int warmup = 0;
    int repeats = 0;
    float p50_ms = 0, p95_ms = 0, mean_ms = 0, min_ms = 0;
    std::vector<float> samples_ms;
};

template <typename LaunchFn>
static TimingStats time_launches(int warmup, int repeats, LaunchFn launch_once) {
    for (int i = 0; i < warmup; ++i) launch_once();
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<gpuEvent_t> starts(repeats), ends(repeats);
    for (int r = 0; r < repeats; ++r) {
        GPU_CHECK(gpuEventCreate(&starts[r]));
        GPU_CHECK(gpuEventCreate(&ends[r]));
    }
    for (int r = 0; r < repeats; ++r) {
        GPU_CHECK(gpuEventRecord(starts[r], 0));
        launch_once();
        GPU_CHECK(gpuEventRecord(ends[r], 0));
    }
    GPU_CHECK(gpuDeviceSynchronize());

    TimingStats st;
    st.warmup = warmup;
    st.repeats = repeats;
    st.samples_ms.resize(repeats);
    for (int r = 0; r < repeats; ++r) {
        float ms = 0.f;
        GPU_CHECK(gpuEventElapsedTime(&ms, starts[r], ends[r]));
        st.samples_ms[r] = ms;
        GPU_CHECK(gpuEventDestroy(starts[r]));
        GPU_CHECK(gpuEventDestroy(ends[r]));
    }
    std::vector<float> sorted = st.samples_ms;
    std::sort(sorted.begin(), sorted.end());
    st.min_ms = sorted.front();
    st.p50_ms = sorted[repeats / 2];
    int p95_idx = std::min(repeats - 1, (int)std::lround(repeats * 0.95) - 1);
    p95_idx = std::max(p95_idx, 0);
    st.p95_ms = sorted[p95_idx];
    double sum = 0;
    for (float v : st.samples_ms) sum += v;
    st.mean_ms = (float)(sum / repeats);
    return st;
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------
static std::string fnum(float v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.6f", v);
    return buf;
}

static std::string fnum2(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f", v);
    return buf;
}

// 名义字节（per-path；bias 无；与 benchmark.py/roofline 口径单源）：
//   fp16 路径：A(MK×2) + W(KN×2) + C(MN×2)
//   int8 融合路径：A(MK×2) + W(KN×1) + scale(N×4) + C(MN×2)
//   dequant 物化 kernel：读 W(KN×1) + scale(N×4)，写 Ŵ(KN×2)
static void bytes_breakdown(int which, int m, int k, int n, double& a_b, double& w_b,
                            double& s_b, double& c_b, double& total) {
    a_b = (double)m * k * 2.0;
    c_b = (double)m * n * 2.0;
    if (which == 0) {
        w_b = (double)k * n * 2.0;
        s_b = 0.0;
    } else if (which == 1) {
        w_b = (double)k * n * 1.0;
        s_b = (double)n * 4.0;
    } else {
        w_b = (double)k * n * 1.0;
        s_b = (double)n * 4.0;
        a_b = 0.0;
        c_b = (double)k * n * 2.0;  // 写 Ŵ16
    }
    total = a_b + w_b + s_b + c_b;
}

static void append_bytes_json(std::string& line, int which, int m, int k, int n) {
    double a_b, w_b, s_b, c_b, total;
    bytes_breakdown(which, m, k, n, a_b, w_b, s_b, c_b, total);
    char tmp[512];
    snprintf(tmp, sizeof(tmp),
             ",\"bytes\":{\"a\":%.0f,\"w\":%.0f,\"scale\":%.0f,\"c\":%.0f,\"total_nominal\":%.0f,"
             "\"weight_mem_fp16\":%.0f,\"weight_mem_int8\":%.0f}",
             a_b, w_b, s_b, c_b, total, (double)k * n * 2.0, (double)k * n * 1.0 + (double)n * 4.0);
    line += tmp;
}

// ---------------------------------------------------------------------------
// bench 模式：单（shape × kernel × 配置）延迟
// ---------------------------------------------------------------------------
static void mode_bench(const std::map<std::string, std::string>& args) {
    int m = std::stoi(args.at("--m"));
    int k = std::stoi(args.at("--k"));
    int n = std::stoi(args.at("--n"));
    std::string kernel_kind = args.count("--kernel") ? args.at("--kernel") : "fp16";
    std::string tile = args.count("--tile") ? args.at("--tile") : "16x16x16";
    std::string reg = args.count("--reg") ? args.at("--reg") : "2x2";
    int warmup = args.count("--warmup") ? std::stoi(args.at("--warmup")) : 20;
    int repeats = args.count("--repeats") ? std::stoi(args.at("--repeats")) : 100;
    unsigned long long seed = args.count("--seed") ? std::stoull(args.at("--seed")) : 20261121ULL;

    const Config* cfg = find_config(tile, reg);
    if (!cfg) {
        fprintf(stderr, "[ERROR] 未知配置 tile=%s reg=%s\n", tile.c_str(), reg.c_str());
        exit(2);
    }
    int which;
    if (kernel_kind == "fp16") which = 0;
    else if (kernel_kind == "int8_fused") which = 1;
    else if (kernel_kind == "dequant") which = 2;
    else if (kernel_kind == "dequant2") which = 3;
    else if (kernel_kind == "dequant3") which = 4;
    else {
        fprintf(stderr, "[ERROR] 未知 kernel %s（fp16|int8_fused|dequant|dequant2|dequant3）\n",
                kernel_kind.c_str());
        exit(2);
    }

    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, 0));

    size_t count_a = (size_t)m * k, count_w = (size_t)k * n;
    KERNEL_HALF_TYPE* dA = nullptr;
    KERNEL_HALF_TYPE* dWf = nullptr;
    int8_t* dWq = nullptr;
    float* dScale = nullptr;
    KERNEL_HALF_TYPE* dC = nullptr;

    if (which <= 1) {
        GPU_CHECK(gpuMalloc((void**)&dA, count_a * sizeof(KERNEL_HALF_TYPE)));
        GPU_CHECK(gpuMalloc((void**)&dC, (size_t)m * n * sizeof(KERNEL_HALF_TYPE)));
        {
            dim3 fb(256);
            fill_random_half<<<dim3((unsigned)((count_a + 255) / 256)), fb>>>(dA, count_a, seed);
            check_launch("fill A");
        }
    }
    if (which == 0) {
        GPU_CHECK(gpuMalloc((void**)&dWf, count_w * sizeof(KERNEL_HALF_TYPE)));
        dim3 fb(256);
        fill_random_half<<<dim3((unsigned)((count_w + 255) / 256)), fb>>>(dWf, count_w, seed + 1);
        check_launch("fill W fp16");
    } else if (which == 1) {
        GPU_CHECK(gpuMalloc((void**)&dWq, count_w));
        GPU_CHECK(gpuMalloc((void**)&dScale, (size_t)n * sizeof(float)));
        dim3 fb(256);
        fill_random_int8<<<dim3((unsigned)((count_w + 255) / 256)), fb>>>(dWq, count_w, seed + 1);
        check_launch("fill W int8");
        fill_scale<<<dim3((unsigned)((n + 255) / 256)), fb>>>(dScale, n, seed + 2);
        check_launch("fill scale");
    } else {
        // dequant 物化（v1/v2/v3 共用同一填充）
        GPU_CHECK(gpuMalloc((void**)&dWq, count_w));
        GPU_CHECK(gpuMalloc((void**)&dScale, (size_t)n * sizeof(float)));
        GPU_CHECK(gpuMalloc((void**)&dWf, count_w * sizeof(KERNEL_HALF_TYPE)));
        dim3 fb(256);
        fill_random_int8<<<dim3((unsigned)((count_w + 255) / 256)), fb>>>(dWq, count_w, seed + 1);
        check_launch("fill W int8");
        fill_scale<<<dim3((unsigned)((n + 255) / 256)), fb>>>(dScale, n, seed + 2);
        check_launch("fill scale");
    }
    GPU_CHECK(gpuDeviceSynchronize());

    TimingStats st;
    if (which <= 1) {
        st = time_launches(warmup, repeats, [&] {
            launch_gemm(*cfg, which, dA, dWf, dWq, dScale, dC, m, n, k);
        });
    } else if (which == 2) {
        size_t total = count_w;
        dim3 fb(256);
        dim3 fgrid((unsigned)((total + 255) / 256));
        st = time_launches(warmup, repeats, [&] {
            dequant_materialize_kernel<<<fgrid, fb>>>(dWq, dScale, dWf, total, n);
            check_launch("dequant launch");
        });
    } else if (which == 3) {
        dim3 fb(256);
        dim3 fgrid((unsigned)((n + 255) / 256), (unsigned)k);
        st = time_launches(warmup, repeats, [&] {
            dequant_materialize_v2<<<fgrid, fb>>>(dWq, dScale, dWf, n);
            check_launch("dequant v2 launch");
        });
    } else {
        dim3 fb(256);
        dim3 fgrid((unsigned)((count_w + 255) / 256));
        st = time_launches(warmup, repeats, [&] {
            dequant_materialize_v3<<<fgrid, fb>>>(dWq, dScale, dWf, (unsigned int)count_w,
                                                  (unsigned int)n);
            check_launch("dequant v3 launch");
        });
    }

    double flops = (which == 2) ? 0.0 : 2.0 * m * n * k;
    double a_b, w_b, s_b, c_b, total_b;
    bytes_breakdown(which, m, k, n, a_b, w_b, s_b, c_b, total_b);

    std::string line;
    line += "{\"mode\":\"bench\"";
    line += ",\"backend_binary\":\"int8_gemm_bench\"";
    line += ",\"kernel\":\"" + kernel_kind + "\"";
    line += ",\"tile\":\"" + tile + "\"";
    line += ",\"reg\":\"" + reg + "\"";
    char tmp[512];
    snprintf(tmp, sizeof(tmp), ",\"m\":%d,\"k\":%d,\"n\":%d,\"grid\":[%u,%u],\"block\":[%u,%u]",
             m, k, n, grid_of(*cfg, m, n).x, grid_of(*cfg, m, n).y, block_of(*cfg).x,
             block_of(*cfg).y);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"warps_per_block\":%u,\"smem_bytes_static\":%zu",
             (block_of(*cfg).x * block_of(*cfg).y + 31) / 32, smem_bytes_of(*cfg));
    line += tmp;
    append_bytes_json(line, which, m, k, n);
    snprintf(tmp, sizeof(tmp), ",\"flops\":%.0f,\"warmup\":%d,\"repeats\":%d", flops, st.warmup,
             st.repeats);
    line += tmp;
    line += ",\"p50_ms\":" + fnum(st.p50_ms);
    line += ",\"p95_ms\":" + fnum(st.p95_ms);
    line += ",\"mean_ms\":" + fnum(st.mean_ms);
    line += ",\"min_ms\":" + fnum(st.min_ms);
    line += ",\"samples_ms\":[";
    for (size_t i = 0; i < st.samples_ms.size(); ++i) {
        line += fnum(st.samples_ms[i]);
        if (i + 1 < st.samples_ms.size()) line += ",";
    }
    line += "]";
    if (which <= 1) {
        snprintf(tmp, sizeof(tmp), ",\"tflops_p50\":%.4f", flops / (st.p50_ms / 1000.0) / 1e12);
        line += tmp;
    }
    snprintf(tmp, sizeof(tmp), ",\"effective_gb_s_p50\":%.2f",
             total_b / (st.p50_ms / 1000.0) / 1e9);
    line += tmp;
    line += ",\"device_name\":\"" + std::string(prop.name) + "\"";
    snprintf(tmp, sizeof(tmp), ",\"sm_count\":%d,\"warp_size\":%d", prop.multiProcessorCount,
             prop.warpSize);
    line += tmp;
    line += "}";
    printf("%s\n", line.c_str());

    if (dA) GPU_CHECK(gpuFree(dA));
    if (dWf) GPU_CHECK(gpuFree(dWf));
    if (dWq) GPU_CHECK(gpuFree(dWq));
    if (dScale) GPU_CHECK(gpuFree(dScale));
    if (dC) GPU_CHECK(gpuFree(dC));
}

// ---------------------------------------------------------------------------
// decodeblock 模式：Qwen3-0.6B 单 Transformer Block 6 投影 GEMM 链
//（hidden=1024, heads=16×head_dim=128 → q=2048, kv=8×128 → kv=1024,
//  intermediate=3072；qcat=4096；gate+up=6144。Decode 单步端到端代理。）
// ---------------------------------------------------------------------------
struct BlockShape {
    int k, n;
    const char* name;
};
static const std::vector<BlockShape>& block_shapes() {
    static const std::vector<BlockShape> v = {
        {1024, 2048, "q_proj"},
        {1024, 1024, "k_proj"},
        {1024, 1024, "v_proj"},
        {4096, 1024, "o_proj"},   // 输入 = qkv concat [M, 4096]
        {1024, 6144, "gate_up"},
        {3072, 1024, "down_proj"},
    };
    return v;
}

static void mode_decodeblock(const std::map<std::string, std::string>& args) {
    int m = std::stoi(args.at("--m"));
    std::string kernel_kind = args.count("--kernel") ? args.at("--kernel") : "fp16";
    std::string tile = args.count("--tile") ? args.at("--tile") : "16x16x16";
    std::string reg = args.count("--reg") ? args.at("--reg") : "2x2";
    int warmup = args.count("--warmup") ? std::stoi(args.at("--warmup")) : 20;
    int repeats = args.count("--repeats") ? std::stoi(args.at("--repeats")) : 100;
    unsigned long long seed = args.count("--seed") ? std::stoull(args.at("--seed")) : 20261121ULL;

    const Config* cfg = find_config(tile, reg);
    if (!cfg) {
        fprintf(stderr, "[ERROR] 未知配置 tile=%s reg=%s\n", tile.c_str(), reg.c_str());
        exit(2);
    }
    int which = (kernel_kind == "int8_fused") ? 1 : 0;
    if (kernel_kind != "fp16" && kernel_kind != "int8_fused") {
        fprintf(stderr, "[ERROR] decodeblock 仅支持 fp16|int8_fused\n");
        exit(2);
    }

    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, 0));

    dim3 fb(256);
    const auto& shapes = block_shapes();
    std::vector<KERNEL_HALF_TYPE*> dAs(shapes.size());
    std::vector<void*> dWs(shapes.size());
    std::vector<float*> dScales(shapes.size(), nullptr);
    std::vector<KERNEL_HALF_TYPE*> dCs(shapes.size());
    double w_bytes_total = 0, act_bytes_total = 0, c_bytes_total = 0, flops_total = 0;
    double w_elems_total = 0, scale_bytes_total = 0;  // 两种表示的权重显存账本共用

    for (size_t i = 0; i < shapes.size(); ++i) {
        int k = shapes[i].k, n = shapes[i].n;
        size_t count_a = (size_t)m * k, count_w = (size_t)k * n;
        GPU_CHECK(gpuMalloc((void**)&dAs[i], count_a * sizeof(KERNEL_HALF_TYPE)));
        GPU_CHECK(gpuMalloc((void**)&dCs[i], (size_t)m * n * sizeof(KERNEL_HALF_TYPE)));
        if (which == 0) {
            GPU_CHECK(gpuMalloc((void**)&dWs[i], count_w * sizeof(KERNEL_HALF_TYPE)));
            fill_random_half<<<dim3((unsigned)((count_w + 255) / 256)), fb>>>(
                (KERNEL_HALF_TYPE*)dWs[i], count_w, seed + 1 + i * 10);
            check_launch("fill W fp16");
            w_bytes_total += (double)count_w * 2.0;
            w_elems_total += (double)count_w;
        } else {
            GPU_CHECK(gpuMalloc((void**)&dWs[i], count_w));
            GPU_CHECK(gpuMalloc((void**)&dScales[i], (size_t)n * sizeof(float)));
            fill_random_int8<<<dim3((unsigned)((count_w + 255) / 256)), fb>>>(
                (int8_t*)dWs[i], count_w, seed + 1 + i * 10);
            check_launch("fill W int8");
            fill_scale<<<dim3((unsigned)((n + 255) / 256)), fb>>>(dScales[i], n, seed + 2 + i * 10);
            check_launch("fill scale");
            w_bytes_total += (double)count_w * 1.0 + (double)n * 4.0;
            w_elems_total += (double)count_w;
            scale_bytes_total += (double)n * 4.0;
        }
        fill_random_half<<<dim3((unsigned)((count_a + 255) / 256)), fb>>>(dAs[i], count_a,
                                                                          seed + 100 + i * 10);
        check_launch("fill A");
        act_bytes_total += (double)count_a * 2.0;
        c_bytes_total += (double)m * n * 2.0;
        flops_total += 2.0 * m * n * k;
    }
    GPU_CHECK(gpuDeviceSynchronize());

    TimingStats st = time_launches(warmup, repeats, [&] {
        for (size_t i = 0; i < shapes.size(); ++i) {
            launch_gemm(*cfg, which, dAs[i], (const KERNEL_HALF_TYPE*)dWs[i],
                        (const int8_t*)dWs[i], dScales[i], dCs[i], m, shapes[i].n, shapes[i].k);
        }
    });

    double total_bytes = w_bytes_total + act_bytes_total + c_bytes_total;
    std::string line;
    line += "{\"mode\":\"decodeblock\"";
    line += ",\"backend_binary\":\"int8_gemm_bench\"";
    line += ",\"kernel\":\"" + kernel_kind + "\"";
    line += ",\"tile\":\"" + tile + "\"";
    line += ",\"reg\":\"" + reg + "\"";
    char tmp[512];
    snprintf(tmp, sizeof(tmp), ",\"m\":%d,\"launch_count\":%zu,\"block_shapes\":[", m,
             shapes.size());
    line += tmp;
    for (size_t i = 0; i < shapes.size(); ++i) {
        snprintf(tmp, sizeof(tmp), "%s{\"name\":\"%s\",\"k\":%d,\"n\":%d}", i ? "," : "",
                 shapes[i].name, shapes[i].k, shapes[i].n);
        line += tmp;
    }
    line += "]";
    snprintf(tmp, sizeof(tmp),
             ",\"bytes\":{\"w\":%.0f,\"activations\":%.0f,\"outputs\":%.0f,\"total_nominal\":%.0f,"
             "\"weight_mem_fp16\":%.0f,\"weight_mem_int8\":%.0f}",
             w_bytes_total, act_bytes_total, c_bytes_total, total_bytes,
             w_elems_total * 2.0, w_elems_total * 1.0 + scale_bytes_total);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"flops\":%.0f,\"warmup\":%d,\"repeats\":%d", flops_total, warmup,
             repeats);
    line += tmp;
    line += ",\"p50_ms\":" + fnum(st.p50_ms);
    line += ",\"p95_ms\":" + fnum(st.p95_ms);
    line += ",\"mean_ms\":" + fnum(st.mean_ms);
    line += ",\"min_ms\":" + fnum(st.min_ms);
    line += ",\"samples_ms\":[";
    for (size_t i = 0; i < st.samples_ms.size(); ++i) {
        line += fnum(st.samples_ms[i]);
        if (i + 1 < st.samples_ms.size()) line += ",";
    }
    line += "]";
    snprintf(tmp, sizeof(tmp), ",\"effective_gb_s_p50\":%.2f",
             total_bytes / (st.p50_ms / 1000.0) / 1e9);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"tflops_p50\":%.4f", flops_total / (st.p50_ms / 1000.0) / 1e12);
    line += tmp;
    line += ",\"device_name\":\"" + std::string(prop.name) + "\"";
    snprintf(tmp, sizeof(tmp), ",\"sm_count\":%d,\"warp_size\":%d", prop.multiProcessorCount,
             prop.warpSize);
    line += tmp;
    line += "}";
    printf("%s\n", line.c_str());

    for (size_t i = 0; i < shapes.size(); ++i) {
        GPU_CHECK(gpuFree(dAs[i]));
        GPU_CHECK(gpuFree(dWs[i]));
        if (dScales[i]) GPU_CHECK(gpuFree(dScales[i]));
        GPU_CHECK(gpuFree(dCs[i]));
    }
}

// ---------------------------------------------------------------------------
// 文件 I/O helpers（run / materialize 模式，ch18 同款 raw bytes）
// ---------------------------------------------------------------------------
static std::vector<char> read_file_bytes(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] 无法打开 %s\n", path.c_str());
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> data((size_t)size);
    if (size > 0 && fread(data.data(), 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "[ERROR] 读取 %s 失败\n", path.c_str());
        exit(2);
    }
    fclose(f);
    return data;
}

static void write_file_bytes(const std::string& path, const void* data, size_t size) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[ERROR] 无法写入 %s\n", path.c_str());
        exit(2);
    }
    fwrite(data, 1, size, f);
    fclose(f);
}

// run 模式：单次执行（正确性验证），文件输入输出
//   fp16:      --in-a(fp16 raw) --in-w(fp16 raw) → --out-c(fp16 raw)
//   int8_fused: --in-a(fp16 raw) --in-w(int8 raw) --in-scale(fp32 raw) → --out-c
static void mode_run(const std::map<std::string, std::string>& args) {
    int m = std::stoi(args.at("--m"));
    int k = std::stoi(args.at("--k"));
    int n = std::stoi(args.at("--n"));
    std::string kernel_kind = args.count("--kernel") ? args.at("--kernel") : "fp16";
    std::string tile = args.count("--tile") ? args.at("--tile") : "16x16x16";
    std::string reg = args.count("--reg") ? args.at("--reg") : "2x2";

    const Config* cfg = find_config(tile, reg);
    if (!cfg) {
        fprintf(stderr, "[ERROR] 未知配置 tile=%s reg=%s\n", tile.c_str(), reg.c_str());
        exit(2);
    }
    int which;
    if (kernel_kind == "fp16") which = 0;
    else if (kernel_kind == "int8_fused") which = 1;
    else {
        fprintf(stderr, "[ERROR] run 模式 kernel 仅支持 fp16|int8_fused\n");
        exit(2);
    }

    size_t count_a = (size_t)m * k, count_w = (size_t)k * n;
    std::vector<char> a_data = read_file_bytes(args.at("--in-a"));
    std::vector<char> w_data = read_file_bytes(args.at("--in-w"));
    std::vector<char> scale_data;
    if (which == 1) scale_data = read_file_bytes(args.at("--in-scale"));
    if (a_data.size() != count_a * 2 || w_data.size() != count_w * (which == 0 ? 2 : 1)) {
        fprintf(stderr, "[ERROR] 输入尺寸不符：A=%zu（期望 %zu） W=%zu（期望 %zu）\n",
                a_data.size(), count_a * 2, w_data.size(), count_w * (which == 0 ? 2 : 1));
        exit(2);
    }
    if (which == 1 && scale_data.size() != (size_t)n * 4) {
        fprintf(stderr, "[ERROR] scale 尺寸不符：%zu（期望 %zu）\n", scale_data.size(),
                (size_t)n * 4);
        exit(2);
    }

    KERNEL_HALF_TYPE* dA;
    void* dW;
    float* dScale = nullptr;
    KERNEL_HALF_TYPE* dC;
    GPU_CHECK(gpuMalloc((void**)&dA, count_a * 2));
    GPU_CHECK(gpuMalloc(&dW, w_data.size()));
    GPU_CHECK(gpuMalloc((void**)&dC, (size_t)m * n * 2));
    if (which == 1) {
        GPU_CHECK(gpuMalloc((void**)&dScale, (size_t)n * 4));
        GPU_CHECK(gpuMemcpy(dScale, scale_data.data(), (size_t)n * 4, gpuMemcpyHostToDevice));
    }
    GPU_CHECK(gpuMemcpy(dA, a_data.data(), count_a * 2, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(dW, w_data.data(), w_data.size(), gpuMemcpyHostToDevice));

    launch_gemm(*cfg, which, dA, (const KERNEL_HALF_TYPE*)dW, (const int8_t*)dW, dScale, dC, m, n,
                k);
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<char> out((size_t)m * n * 2);
    GPU_CHECK(gpuMemcpy(out.data(), dC, out.size(), gpuMemcpyDeviceToHost));
    write_file_bytes(args.at("--out-c"), out.data(), out.size());

    GPU_CHECK(gpuFree(dA));
    GPU_CHECK(gpuFree(dW));
    GPU_CHECK(gpuFree(dC));
    if (dScale) GPU_CHECK(gpuFree(dScale));
}

// materialize 模式：int8 + scale → 物化 FP16 权重（文件进文件出）
static void mode_materialize(const std::map<std::string, std::string>& args) {
    int k = std::stoi(args.at("--k"));
    int n = std::stoi(args.at("--n"));
    std::vector<char> w_data = read_file_bytes(args.at("--in-w"));
    std::vector<char> scale_data = read_file_bytes(args.at("--in-scale"));
    size_t count_w = (size_t)k * n;
    if (w_data.size() != count_w || scale_data.size() != (size_t)n * 4) {
        fprintf(stderr, "[ERROR] 输入尺寸不符：W=%zu（期望 %zu） scale=%zu（期望 %zu）\n",
                w_data.size(), count_w, scale_data.size(), (size_t)n * 4);
        exit(2);
    }

    int8_t* dWq;
    float* dScale;
    KERNEL_HALF_TYPE* dWf;
    GPU_CHECK(gpuMalloc((void**)&dWq, count_w));
    GPU_CHECK(gpuMalloc((void**)&dScale, (size_t)n * 4));
    GPU_CHECK(gpuMalloc((void**)&dWf, count_w * 2));
    GPU_CHECK(gpuMemcpy(dWq, w_data.data(), count_w, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(dScale, scale_data.data(), (size_t)n * 4, gpuMemcpyHostToDevice));

    dim3 fb(256);
    dim3 fgrid((unsigned)((n + 255) / 256), (unsigned)k);
    dequant_materialize_v2<<<fgrid, fb>>>(dWq, dScale, dWf, n);
    check_launch("materialize launch (v2 2D grid)");
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<char> out(count_w * 2);
    GPU_CHECK(gpuMemcpy(out.data(), dWf, out.size(), gpuMemcpyDeviceToHost));
    write_file_bytes(args.at("--out-w"), out.data(), out.size());

    GPU_CHECK(gpuFree(dWq));
    GPU_CHECK(gpuFree(dScale));
    GPU_CHECK(gpuFree(dWf));
}

// ---------------------------------------------------------------------------
// selftest 模式：host 三重循环 vs 各配置（fp16 与 int8_fused 双 kernel）
// int8 的 host 参考用"同一份 q 的 dequant 权重"做 FP32 GEMM（隔离累加序差），
// 与真值 W 的 GEMM 差另报（量化误差，应与本层无关地偏大）。
// ---------------------------------------------------------------------------
static void mode_selftest(const std::map<std::string, std::string>& args) {
    (void)args;
    struct Shape {
        int m, k, n;
    };
    std::vector<Shape> shapes = {{32, 24, 40}, {17, 19, 13}, {4, 8, 4}, {1, 16, 24}};
    int passed = 0, failed = 0;

    for (const auto& s : shapes) {
        size_t count_a = (size_t)s.m * s.k, count_w = (size_t)s.k * s.n;
        std::vector<float> hA(count_a), hW(count_w), hScale(s.n);
        std::vector<int8_t> hQ(count_w);
        unsigned long long seed = 20261121ULL + s.m * 131 + s.k * 17 + s.n;
        for (size_t i = 0; i < count_a; ++i)
            hA[i] = ((float)hash_u32(i, seed) / 2147483648.0f) - 1.0f;
        for (size_t i = 0; i < count_w; ++i)
            hW[i] = ((float)hash_u32(i, seed + 1) / 2147483648.0f) - 1.0f;
        // host 侧 per-channel 量化（round-half-away；fixtures 链路用 numpy RNE——
        // 两者只在 .5 边界差 1 LSB，selftest 参考用同一份 q，不受影响）
        for (int j = 0; j < s.n; ++j) {
            float amax = 0.f;
            for (int i = 0; i < s.k; ++i) amax = std::max(amax, fabsf(hW[(size_t)i * s.n + j]));
            float sc = (amax > 0.f) ? amax / 127.0f : 1.0f;
            hScale[j] = sc;
            for (int i = 0; i < s.k; ++i) {
                int q = (int)lroundf(hW[(size_t)i * s.n + j] / sc);
                q = std::max(-127, std::min(127, q));
                hQ[(size_t)i * s.n + j] = (int8_t)q;
            }
        }
        // host 参考（真值口径对齐，ep18 教训的引申）：device 读的是 FP16 舍入后的
        // A/W，host 参考必须用【同一份 FP16 舍入值】的 float 形式——否则参考与
        // 被测差的是输入舍入项（~1e-3），不是实现误差（~1e-6）。量化也自
        // FP16 舍入权重计算（与 fixtures/服务化流程一致：checkpoint 为 FP16）。
        std::vector<float> hA16f(count_a), hW16f(count_w);
        for (size_t i = 0; i < count_a; ++i)
            hA16f[i] = to_float_val(to_type<KERNEL_HALF_TYPE>(hA[i]));
        for (size_t i = 0; i < count_w; ++i)
            hW16f[i] = to_float_val(to_type<KERNEL_HALF_TYPE>(hW[i]));
        for (int j = 0; j < s.n; ++j) {
            float amax = 0.f;
            for (int i = 0; i < s.k; ++i) amax = std::max(amax, fabsf(hW16f[(size_t)i * s.n + j]));
            float sc = (amax > 0.f) ? amax / 127.0f : 1.0f;
            hScale[j] = sc;
            for (int i = 0; i < s.k; ++i) {
                int q = (int)lroundf(hW16f[(size_t)i * s.n + j] / sc);
                q = std::max(-127, std::min(127, q));
                hQ[(size_t)i * s.n + j] = (int8_t)q;
            }
        }
        // host 参考：dequant 权重 FP32 三重循环（k 升序）
        std::vector<float> hRefF16((size_t)s.m * s.n, 0.0f);
        std::vector<float> hRefQ((size_t)s.m * s.n, 0.0f);
        std::vector<float> hWq(count_w);
        for (size_t i = 0; i < count_w; ++i) {
            int col = (int)(i % (size_t)s.n);
            hWq[i] = (float)hQ[i] * hScale[col];
        }
        for (int i = 0; i < s.m; ++i)
            for (int p = 0; p < s.k; ++p) {
                float a = hA16f[(size_t)i * s.k + p];
                for (int j = 0; j < s.n; ++j) {
                    hRefF16[(size_t)i * s.n + j] += a * hW16f[(size_t)p * s.n + j];
                    hRefQ[(size_t)i * s.n + j] += a * hWq[(size_t)p * s.n + j];
                }
            }

        KERNEL_HALF_TYPE *dA, *dWf, *dC;
        int8_t* dWq;
        float* dScale;
        GPU_CHECK(gpuMalloc((void**)&dA, count_a * 2));
        GPU_CHECK(gpuMalloc((void**)&dWf, count_w * 2));
        GPU_CHECK(gpuMalloc((void**)&dWq, count_w));
        GPU_CHECK(gpuMalloc((void**)&dScale, (size_t)s.n * 4));
        GPU_CHECK(gpuMalloc((void**)&dC, (size_t)s.m * s.n * 2));
        {
            std::vector<KERNEL_HALF_TYPE> tmpA(count_a), tmpW(count_w);
            for (size_t i = 0; i < count_a; ++i) tmpA[i] = to_type<KERNEL_HALF_TYPE>(hA[i]);
            for (size_t i = 0; i < count_w; ++i) tmpW[i] = to_type<KERNEL_HALF_TYPE>(hW[i]);
            GPU_CHECK(gpuMemcpy(dA, tmpA.data(), count_a * 2, gpuMemcpyHostToDevice));
            GPU_CHECK(gpuMemcpy(dWf, tmpW.data(), count_w * 2, gpuMemcpyHostToDevice));
            GPU_CHECK(gpuMemcpy(dWq, hQ.data(), count_w, gpuMemcpyHostToDevice));
            GPU_CHECK(gpuMemcpy(dScale, hScale.data(), (size_t)s.n * 4, gpuMemcpyHostToDevice));
        }

        for (int which = 0; which <= 1; ++which) {
            for (const auto& cfg : config_table()) {
                launch_gemm(cfg, which, dA, dWf, dWq, dScale, dC, s.m, s.n, s.k);
                GPU_CHECK(gpuDeviceSynchronize());
                std::vector<KERNEL_HALF_TYPE> out((size_t)s.m * s.n);
                GPU_CHECK(gpuMemcpy(out.data(), dC, out.size() * 2, gpuMemcpyDeviceToHost));
                const std::vector<float>& ref = (which == 0) ? hRefF16 : hRefQ;
                float max_abs = 0;
                for (size_t i = 0; i < out.size(); ++i) {
                    // 真值口径第二段：kernel 输出 C 是 FP16——参考也做同一 FP16
                    // 输出舍入，比较只剩累加序/FMA 差（否则差的是输出舍入项
                    // |C|~2 时半精度 ulp≈2e-3，掩盖实现误差）。
                    float ref_rounded = to_float_val(to_type<KERNEL_HALF_TYPE>(ref[i]));
                    max_abs = std::max(max_abs, fabsf(to_float_val(out[i]) - ref_rounded));
                }
                bool ok = max_abs < 1e-3f;
                printf("SELFTEST kernel=%s config=%s shape=%dx%dx%d max_abs=%.3e %s\n",
                       which == 0 ? "fp16" : "int8_fused", cfg.name, s.m, s.k, s.n, max_abs,
                       ok ? "PASS" : "FAIL");
                ok ? passed++ : failed++;
            }
        }
        // 量化误差 sanity（与实现误差分开）：int8 参考权重 vs 真 W 的 GEMM 差
        float q_err_max = 0;
        for (size_t i = 0; i < hRefQ.size(); ++i)
            q_err_max = std::max(q_err_max, fabsf(hRefQ[i] - hRefF16[i]));
        printf("SELFTEST quant_error_sanity shape=%dx%dx%d max_abs=%.3e（量化项，仅记录）\n", s.m,
               s.k, s.n, q_err_max);

        GPU_CHECK(gpuFree(dA));
        GPU_CHECK(gpuFree(dWf));
        GPU_CHECK(gpuFree(dWq));
        GPU_CHECK(gpuFree(dScale));
        GPU_CHECK(gpuFree(dC));
    }
    printf("{\"mode\":\"selftest\",\"passed\":%d,\"failed\":%d}\n", passed, failed);
    if (failed > 0) exit(3);
}

// ---------------------------------------------------------------------------
// attrs 模式：逐配置寄存器/共享内存占用
// ---------------------------------------------------------------------------
static void mode_attrs(const std::map<std::string, std::string>& args) {
    (void)args;
    std::string json = "{\"mode\":\"attrs\",\"configs\":[";
    bool first = true;
    for (const auto& cfg : config_table()) {
        dim3 block = block_of(cfg);
        for (int which = 0; which <= 1; ++which) {
            const void* func = nullptr;
#define GET_FUNC(TM, TN, TK, RM, RN)                                                     \
    if (cfg.tm == TM && cfg.tn == TN && cfg.tk == TK && cfg.rm == RM && cfg.rn == RN)    \
        func = (which == 0)                                                              \
                   ? (const void*)&gemm_fp16_tiled_kernel<TM, TN, TK, RM, RN>            \
                   : (const void*)&gemm_int8_fused_kernel<TM, TN, TK, RM, RN>;
            GET_FUNC(16, 16, 16, 2, 2);
            GET_FUNC(32, 32, 8, 2, 2);
            GET_FUNC(32, 32, 8, 4, 4);
            GET_FUNC(64, 64, 8, 2, 2);
            GET_FUNC(64, 64, 8, 4, 4);
#undef GET_FUNC
            if (!func) continue;
            char buf[512];
#if !defined(DISABLE_FUNC_ATTRS)
            gpuFuncAttributes attr;
            gpuError_t err = gpuFuncGetAttributes(&attr, func);
            if (err == gpuSuccess) {
                snprintf(buf, sizeof(buf),
                         "%s{\"kernel\":\"%s\",\"name\":\"%s\",\"block\":[%u,%u],\"threads\":%u,"
                         "\"num_regs\":%d,\"shared_bytes_static\":%zu,\"shared_bytes_attr\":%zu,"
                         "\"local_bytes\":%zu,\"max_threads_per_block\":%d}",
                         first ? "" : ",", which == 0 ? "fp16" : "int8_fused", cfg.name, block.x,
                         block.y, block.x * block.y, attr.numRegs, smem_bytes_of(cfg),
                         (size_t)attr.sharedSizeBytes, (size_t)attr.localSizeBytes,
                         attr.maxThreadsPerBlock);
            } else
#endif
            {
                snprintf(buf, sizeof(buf),
                         "%s{\"kernel\":\"%s\",\"name\":\"%s\",\"block\":[%u,%u],\"threads\":%u,"
                         "\"num_regs\":null,\"shared_bytes_static\":%zu,\"shared_bytes_attr\":null,"
                         "\"local_bytes\":null,\"max_threads_per_block\":null}",
                         first ? "" : ",", which == 0 ? "fp16" : "int8_fused", cfg.name, block.x,
                         block.y, block.x * block.y, smem_bytes_of(cfg));
            }
            json += buf;
            first = false;
        }
    }
    json += "]}";
    printf("%s\n", json.c_str());
}

// ---------------------------------------------------------------------------
// main：极简 key=value 参数解析
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "用法: int8_gemm_bench <selftest|attrs|run|materialize|bench|decodeblock> "
                "[--key value ...]\n");
        return 1;
    }
    std::string mode = argv[1];
    std::map<std::string, std::string> args;
    for (int i = 2; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];

    if (mode == "selftest")
        mode_selftest(args);
    else if (mode == "attrs")
        mode_attrs(args);
    else if (mode == "run")
        mode_run(args);
    else if (mode == "materialize")
        mode_materialize(args);
    else if (mode == "bench")
        mode_bench(args);
    else if (mode == "decodeblock")
        mode_decodeblock(args);
    else {
        fprintf(stderr, "[ERROR] 未知模式 %s\n", mode.c_str());
        return 1;
    }
    gpuError_t err = gpuGetLastError();
    if (err != gpuSuccess) {
        fprintf(stderr, "[GPU-ERROR] 尾部遗留错误: %s\n", gpuGetErrorString(err));
        return 2;
    }
    return 0;
}
