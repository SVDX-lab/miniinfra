// ============================================================================
// 第 18 期：Tensor Core / WMMA——计算单元升级（第 17 期同源，算法与 tiling 不变）
// ============================================================================
//
// 与第 17 期的关系（"只换计算单元"实验）：
//   - 本文件 = 第 17 期 naive_gemm.cu 的完整副本（naive/tiled/epilogue 全保留，
//     支持同会话同口径重跑对照）+ 新增 gemm_wmma_kernel；
//   - gemm_wmma_kernel：nvcuda::wmma fragment API（16×16×16），FP16 输入 +
//     FP32 累加（--acc fp32 默认）/ FP16 累加（--acc fp16，精度对照），
//     接第 05 期 dtype 结论；主循环仍为"smem tile 加载 → 片上 MMA"，
//     仅把逐条 FMA 的 kk 内层替换为 wmma::mma_sync；
//   - tile 映射：TILE_M/TILE_N 与第 17 期一致；TILE_K=8 的配置在 WMMA 版按
//     k-step=16 执行（=2 个 k-fragment，记录为 wm16 映射），TILE_K=16 天然对齐；
//     warp 布局 = (TILE_M/16)×(TILE_N/16)，每 warp 持 1 个 16×16 累加 fragment；
//   - fp32 累加器的输出经 smem 暂存换出（store_matrix_sync 的指针类型须与
//     累加器一致，half 输出需逐元素转换 + 越界 guard），fp16 累加器同路径统一。
//
// MUSA 列：MMA 接口为本期"待实机确认"项（大纲 §14 准入第 10 条）——
//   编译链按 __has_include 探测 <musa_wmma.h>/<mma.h>，未命中则 WMMA 编译
//   单元整体退出（-DDISABLE_WMMA 回退），探测过程与结论见 implementations/musa/
//   mma_probe/（这正是本期要回答的问题，缺失即平台边界证据）。
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
#include <type_traits>

#if defined(USE_MUSA)
  #include <musa_runtime.h>
  #if defined(__has_include)
    #if __has_include(<musa_fp16.h>)
      #include <musa_fp16.h>
      #define HAVE_MUSA_FP16_H 1
    #endif
  #endif
  // ---- GPU API 兼容 shim：CUDA 命名 → MUSA 命名（机械替换单点化）----
  #define gpuError_t          musaError_t
  #define gpuSuccess          musaSuccess
  #define gpuGetErrorString   musaGetErrorString
  #define gpuGetLastError     musaGetLastError
  // 注：MUSA SDK 只有 musaDeviceProp，无 CUDA 式 cudaDeviceProp_t 别名（实机核对，见台账）
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
  // ---- CUDA 分支：同样的 gpu* 统一命名（与 MUSA shim 对称）----
  // 注：CUDA 13 已删除 cudaDeviceProp_t 别名（与 MUSA 无 _t 对称），统一用裸类型名
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

// ---------------------------------------------------------------------------
// WMMA 头文件探测（第 18 期 MUSA "待实机确认"项的编译级证据）：
// - CUDA：<mma.h>（nvcuda::wmma，sm_70+，Ada 原生支持 16x16x16 fp16）；
// - MUSA：按 __has_include 探测 <musa_wmma.h> / <mma.h>（探测过程与 grep
//   证据存 implementations/musa/mma_probe/，未命中 = 平台边界）。
// 命名空间经 WMMA_NS 宏单点化（musa 侧若为 musa::wmma 只改一处）。
// ---------------------------------------------------------------------------
#if !defined(DISABLE_WMMA)
  #if !defined(USE_MUSA)
    #include <mma.h>
    #define HAVE_WMMA 1
    #if !defined(WMMA_NS)
      #define WMMA_NS nvcuda
    #endif
  #else
    #if defined(__has_include)
      #if __has_include(<musa_wmma.h>)
        #include <musa_wmma.h>
        #define HAVE_WMMA 1
        #define WMMA_NS_MUSA_WMMA_H 1
        #if !defined(WMMA_NS)
          #define WMMA_NS musa
        #endif
      #elif __has_include(<mma.h>)
        #include <mma.h>
        #define HAVE_WMMA 1
        #define WMMA_NS_MMA_H 1
        // <mma.h> 在 MUSA 上的命名空间未定（nvcuda 原样移植 or musa）：
        // 命令行 -DWMMA_NS=… 可覆盖默认猜测（build.sh 回退链尝试）
        #if !defined(WMMA_NS)
          #define WMMA_NS nvcuda
        #endif
      #endif
    #endif
  #endif
#endif

#if defined(HAVE_WMMA)
  namespace wmma_api = WMMA_NS::wmma;
#endif

// FP16 类型与转换：
// - CUDA：cuda_fp16.h 的 __half + __float2half/__half2float（RN）；
// - MUSA：默认走 musa_fp16.h 的 __half；若实机头文件不可用/命名不同，
//   编译加 -DMUSA_HALF_BITS=1 走自实现 IEEE RNE 位级转换（本地已用
//   numpy.float16 全域对拍验证，见 EXPERIMENT_LOG）。
#if !defined(USE_MUSA) || defined(HAVE_MUSA_FP16_H)
  #define KERNEL_HALF_TYPE __half
#endif
#if defined(USE_MUSA) && !defined(HAVE_MUSA_FP16_H)
  #define KERNEL_HALF_TYPE gpu_half_bits
#endif

// ---------------------------------------------------------------------------
// 自实现 binary16（IEEE 754 round-to-nearest-even）——MUSA 平台边界 fallback
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
    if (exp32 == 0xFF) {  // Inf / NaN
        out = (unsigned short)(mant ? 0xFE00u : 0x7C00u);
    } else if (exp32 == 0 && mant == 0) {
        out = 0;  // ±0
    } else {
        int e = (int)exp32 - 127;  // 无偏指数
        if (e > 15) {
            out = 0x7C00u;  // 溢出 → Inf
        } else if (e >= -14) {
            // 规格数：10 位尾数 + RNE
            unsigned int hm = mant >> 13;
            unsigned int rem = mant & 0x1FFFu;
            unsigned int h = ((unsigned int)(e + 15) << 10) | hm;
            if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) h += 1;
            out = (unsigned short)h;
        } else if (e >= -25) {
            // 次规格数：value = full × 2^(e-23)，目标 = s × 2^-24
            // s = round(full × 2^(e+1))，右移 (-e-1) 位 + RNE
            unsigned int full = 0x800000u | mant;
            int shift = -e - 1;  // ≥ 0
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
            out = 0;  // 下溢 → 0
        }
    }
    return (unsigned short)(sign | out);
}

#endif  // MUSA fallback 类型

// ---------------------------------------------------------------------------
// 统一转换入口（模板特化：float 主线 + gpu_half 对照）
// ---------------------------------------------------------------------------
template <typename T>
__device__ __host__ inline T to_type(float v);

template <typename T>
__device__ __host__ inline float to_float_val(T v);

template <>
__device__ __host__ inline float to_type<float>(float v) { return v; }
template <>
__device__ __host__ inline float to_float_val<float>(float v) { return v; }

#if !defined(DISABLE_FP16)
#if defined(USE_MUSA)
#if defined(HAVE_MUSA_FP16_H)  // musa_fp16.h 可用：假定与 CUDA 同名 intrinsics（实机验证）
template <>
__device__ __host__ inline __half to_type<__half>(float v) { return __float2half(v); }
template <>
__device__ __host__ inline float to_float_val<__half>(__half v) { return __half2float(v); }
#else  // 位级转换 fallback
template <>
__device__ __host__ inline gpu_half_bits to_type<gpu_half_bits>(float v) {
    gpu_half_bits h;
    h.bits = float_to_half_rne(v);
    return h;
}
template <>
__device__ __host__ inline float to_float_val<gpu_half_bits>(gpu_half_bits v) {
    // binary16 → binary32（精确，无舍入）
    unsigned short h = v.bits;
    unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // 次规格：value = mant × 2^-24；0x200 对应 e=-15，每左移 1 位 e 减 1
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
#else  // CUDA
template <>
__device__ __host__ inline __half to_type<__half>(float v) { return __float2half(v); }
template <>
__device__ __host__ inline float to_float_val<__half>(__half v) { return __half2float(v); }
#endif
#endif  // !DISABLE_FP16

// ---------------------------------------------------------------------------
// 错误检查
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
// Epilogue：0=无 1=Bias 2=Bias+GELU(tanh) 3=Bias+SiLU
//（公式与 implementations/cpu/naive_gemm.py 完全同源）
// ---------------------------------------------------------------------------
__device__ __host__ inline float gelu_tanh_f(float v) {
    return 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
__device__ __host__ inline float silu_f(float v) {
    return v / (1.0f + expf(-v));
}

template <int EPI>
__device__ inline float apply_epilogue(float v, int gn, const float* bias) {
    if (EPI >= 1) v += bias[gn];
    if (EPI == 2) v = gelu_tanh_f(v);
    if (EPI == 3) v = silu_f(v);
    return v;
}

// ---------------------------------------------------------------------------
// Kernel 1：零组织基线（无 tiling，直读 global）
// 每个 thread 算一个输出元素：读 A 的整行 + B 的整列（B 列访问跨行、
// 不合并访存）——这是"没把数据切进片上"的代价，扫描时作 tiling 收益锚点。
// ---------------------------------------------------------------------------
template <typename TIn, typename TOut>
__global__ void gemm_naive_kernel(const TIn* __restrict__ A, const TIn* __restrict__ B,
                                  TOut* __restrict__ C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += to_float_val(A[(size_t)row * K + k]) * to_float_val(B[(size_t)k * N + col]);
    }
    C[(size_t)row * N + col] = to_type<TOut>(acc);
}

// ---------------------------------------------------------------------------
// Kernel 2：单层 tiling + 片上缓冲 + 寄存器分块（本期主实现）
//
//   block 负责输出 TILE_M×TILE_N；block 内线程排成 (TM/RM)×(TN/RN)，
//   每线程持有 REG_M×REG_N 个 float 累加器（寄存器分块）。
//   每 k-tile：协作加载 A-tile(TM×TK) 与 B-tile(TK×TN) 进 shared memory，
//   __syncthreads() 后线程网格各自做 RM×RN×TK 条 FMA。
//
//   bank conflict 规避：As 行距 TK+1、Bs 行距 TN+1（+1 padding）。
//   As[row][kk]：同 tx 异 ty 的地址差 (TK+1)——与 32 互质步长 → 无冲突；
//   同 ty 异 tx 读同一地址（广播）；Bs[kk][col]：异 tx 相邻地址 → 无冲突。
//   （第 19 期再对比"无 padding 的原始冲突"与异步流水。）
// ---------------------------------------------------------------------------
template <typename TIn, typename TOut, int TILE_M, int TILE_N, int TILE_K, int REG_M, int REG_N,
          int EPI>
__global__ void gemm_tiled_kernel(const TIn* __restrict__ A, const TIn* __restrict__ B,
                                  const float* __restrict__ bias, TOut* __restrict__ C, int M,
                                  int N, int K) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;  // 列方向（快维）
    const int ty = threadIdx.y;  // 行方向
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
        // 协作加载 A-tile（快维 = K 列 → global 合并访存），越界补 0
        for (int idx = tid; idx < TILE_M * TILE_K; idx += nthreads) {
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int gm = m0 + r;
            int gk = k0 + c;
            As[r][c] = (gm < M && gk < K) ? to_float_val(A[(size_t)gm * K + gk]) : 0.0f;
        }
        // 协作加载 B-tile（快维 = N 列 → global 合并访存），越界补 0
        for (int idx = tid; idx < TILE_K * TILE_N; idx += nthreads) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gk = k0 + r;
            int gn = n0 + c;
            Bs[r][c] = (gk < K && gn < N) ? to_float_val(B[(size_t)gk * N + gn]) : 0.0f;
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

    // epilogue（融合位置）+ 写回（行主序合并访存）
#pragma unroll
    for (int m = 0; m < REG_M; ++m) {
        int gm = m0 + ty * REG_M + m;
#pragma unroll
        for (int n = 0; n < REG_N; ++n) {
            int gn = n0 + tx * REG_N + n;
            if (gm < M && gn < N) {
                float v = apply_epilogue<EPI>(acc[m][n], gn, bias);
                C[(size_t)gm * N + gn] = to_type<TOut>(v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2b：WMMA（Tensor Core fragment，16×16×16）——第 18 期主实现
//
// 与 gemm_tiled_kernel 的对应关系（"算法与 tiling 不变，只换计算单元"）：
//   - block 仍负责 TILE_M×TILE_N 输出，k 维按 KSTEP=16 步进（TILE_K=8 的
//     配置映射为 wm16：一次加载 16 行 k，=2 个 k-fragment）；
//   - warp 布局 = (TILE_M/16)×(TILE_N/16)，每 warp 持 1 个 16×16 累加
//     fragment（对照：朴素版每 thread 持 RM×RN 个标量累加器）；
//   - smem tile 为 half（FP16 输入直达，无需 float 转换）；行距 +8 halves
//     （=16B）同时满足 WMMA 对 ldm 是 8 倍数的对齐要求；
//   - fp32 累加器经 smem 暂存换出（store_matrix_sync 指针类型须与累加器
//     一致），fp16 累加器走同一暂存路径，逐元素 guard 写回 half C。
// ---------------------------------------------------------------------------
#if defined(HAVE_WMMA) && !defined(DISABLE_FP16)
template <int TILE_M, int TILE_N, int KSTEP, int ACC_FLOAT>
__global__ void gemm_wmma_kernel(const KERNEL_HALF_TYPE* __restrict__ A,
                                 const KERNEL_HALF_TYPE* __restrict__ B,
                                 KERNEL_HALF_TYPE* __restrict__ C, int M, int N, int K) {
    __shared__ KERNEL_HALF_TYPE As[TILE_M][KSTEP + 8];
    __shared__ KERNEL_HALF_TYPE Bs[KSTEP][TILE_N + 8];
    __shared__ float Cs_f[TILE_M][TILE_N];
    __shared__ KERNEL_HALF_TYPE Cs_h[TILE_M][TILE_N + 8];

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int warps_n = TILE_N / 16;
    const int warp_row = (tid / 32) / warps_n;
    const int warp_col = (tid / 32) % warps_n;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    using acc_t = typename std::conditional<ACC_FLOAT == 1, float, KERNEL_HALF_TYPE>::type;
    wmma_api::fragment<wmma_api::accumulator, 16, 16, 16, acc_t> acc;
    if constexpr (ACC_FLOAT == 1) {
        wmma_api::fill_fragment(acc, 0.0f);
    } else {
        wmma_api::fill_fragment(acc, to_type<KERNEL_HALF_TYPE>(0.0f));
    }

    for (int k0 = 0; k0 < K; k0 += KSTEP) {
        // 协作加载 A/B 的 half tile（越界补 0；快维沿 K/N 列 → 合并访存）
        for (int idx = tid; idx < TILE_M * KSTEP; idx += nthreads) {
            int r = idx / KSTEP;
            int c = idx % KSTEP;
            int gm = m0 + r;
            int gk = k0 + c;
            As[r][c] = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : to_type<KERNEL_HALF_TYPE>(0.0f);
        }
        for (int idx = tid; idx < KSTEP * TILE_N; idx += nthreads) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gk = k0 + r;
            int gn = n0 + c;
            Bs[r][c] = (gk < K && gn < N) ? B[(size_t)gk * N + gn] : to_type<KERNEL_HALF_TYPE>(0.0f);
        }
        __syncthreads();

        // 计算单元切换点：第 17 期这里是 RM×RN×TK 条标量 FMA，
        // 本期一条 wmma::mma_sync 完成 16×16×16×2 = 8192 FLOP
        wmma_api::fragment<wmma_api::matrix_a, 16, 16, 16, KERNEL_HALF_TYPE, wmma_api::row_major> a_frag;
        wmma_api::fragment<wmma_api::matrix_b, 16, 16, 16, KERNEL_HALF_TYPE, wmma_api::row_major> b_frag;
        wmma_api::load_matrix_sync(a_frag, &As[warp_row * 16][0], KSTEP + 8);
        wmma_api::load_matrix_sync(b_frag, &Bs[0][warp_col * 16], TILE_N + 8);
        wmma_api::mma_sync(acc, a_frag, b_frag, acc);
        __syncthreads();
    }

    // epilogue 位置：fragment → smem 暂存（fp32 累加必须经此换出为 half；
    // 该路径天然提供越界 guard，ragged 形状不会写越界）
    if constexpr (ACC_FLOAT == 1) {
        wmma_api::store_matrix_sync(&Cs_f[warp_row * 16][warp_col * 16], acc, TILE_N,
                                    wmma_api::mem_row_major);
    } else {
        wmma_api::store_matrix_sync(&Cs_h[warp_row * 16][warp_col * 16], acc, TILE_N + 8,
                                    wmma_api::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < TILE_M * TILE_N; idx += nthreads) {
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        int gm = m0 + r;
        int gn = n0 + c;
        if (gm < M && gn < N) {
            float v = (ACC_FLOAT == 1) ? Cs_f[r][c] : to_float_val(Cs_h[r][c]);
            C[(size_t)gm * N + gn] = to_type<KERNEL_HALF_TYPE>(v);
        }
    }
}
#endif  // HAVE_WMMA && !DISABLE_FP16

// ---------------------------------------------------------------------------
// Kernel 3：分离 epilogue 的 elementwise kernel（"不融合"对照路径）
// sep3：GEMM → bias_kernel → act_kernel（两次额外读+写 C）
// sep2：GEMM → bias_act_kernel（一次额外读+写 C）
// ---------------------------------------------------------------------------
template <typename T>
__global__ void bias_kernel(T* __restrict__ C, const float* __restrict__ bias, size_t total,
                            int N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = to_float_val(C[idx]) + bias[idx % N];
        C[idx] = to_type<T>(v);
    }
}

template <typename T, int ACT>  // ACT: 2=GELU 3=SiLU
__global__ void act_kernel(const T* __restrict__ in, T* __restrict__ out, size_t total) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = to_float_val(in[idx]);
        v = (ACT == 2) ? gelu_tanh_f(v) : silu_f(v);
        out[idx] = to_type<T>(v);
    }
}

template <typename T, int ACT>
__global__ void bias_act_kernel(T* __restrict__ C, const float* __restrict__ bias, size_t total,
                                int N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = to_float_val(C[idx]) + bias[idx % N];
        v = (ACT == 2) ? gelu_tanh_f(v) : silu_f(v);
        C[idx] = to_type<T>(v);
    }
}

// ---------------------------------------------------------------------------
// 随机填充 kernel（bench 模式数据；不依赖 host 端 half 转换）
// LCG hash：value = hash(idx) 映射到 [-1, 1)，确定性
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

#if !defined(DISABLE_FP16)
__global__ void fill_random_half(KERNEL_HALF_TYPE* out, size_t count, unsigned long long seed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = to_type<KERNEL_HALF_TYPE>(((float)hash_u32(idx, seed) / 2147483648.0f) - 1.0f);
    }
}
#endif

// ---------------------------------------------------------------------------
// 配置注册表：tile (TM,TN,TK) × REG (RM,RN)，编译期模板实例化
// 线程块 = (TN/RN, TM/RM)；smem = (TM*(TK+1) + TK*(TN+1))*4 字节
// ---------------------------------------------------------------------------
struct Config {
    const char* name;  // 例如 "32x32x8_reg2x2"
    int tm, tn, tk;
    int rm, rn;
    bool epi_variants;  // 是否编译了 Bias/GELU/SiLU epilogue 变体（控制实例化数量）
};

static const std::vector<Config>& config_table() {
    static const std::vector<Config> table = {
        {"16x16x16_reg2x2", 16, 16, 16, 2, 2, false},
        {"16x16x16_reg4x4", 16, 16, 16, 4, 4, false},
        {"32x32x8_reg2x2", 32, 32, 8, 2, 2, true},
        {"32x32x8_reg4x4", 32, 32, 8, 4, 4, false},
        {"32x64x8_reg2x2", 32, 64, 8, 2, 2, false},
        {"32x64x8_reg4x4", 32, 64, 8, 4, 4, false},
        {"64x32x8_reg2x2", 64, 32, 8, 2, 2, false},
        {"64x32x8_reg4x4", 64, 32, 8, 4, 4, false},
        {"64x64x8_reg2x2", 64, 64, 8, 2, 2, false},
        {"64x64x8_reg4x4", 64, 64, 8, 4, 4, true},
        {"64x64x8_reg8x8", 64, 64, 8, 8, 8, false},
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

// ---------------------------------------------------------------------------
// tiled kernel 分发（模板实例化表）
// ---------------------------------------------------------------------------
template <typename TIn, typename TOut>
static bool launch_tiled(const Config& cfg, int epilogue, dim3 grid, dim3 block, const TIn* dA,
                         const TIn* dB, const float* dBias, TOut* dC, int M, int N, int K) {
#define LAUNCH_EPI(TM, TN, TK, RM, RN, E, allowed)                              \
    if (cfg.tm == TM && cfg.tn == TN && cfg.tk == TK && cfg.rm == RM && cfg.rn == RN && \
        (E == 0 || allowed)) {                                                 \
        if (E == epilogue) {                                                   \
            gemm_tiled_kernel<TIn, TOut, TM, TN, TK, RM, RN, E>                \
                <<<grid, block>>>(dA, dB, dBias, dC, M, N, K);                 \
            return true;                                                       \
        }                                                                      \
    }
    LAUNCH_EPI(16, 16, 16, 2, 2, 0, false);
    LAUNCH_EPI(16, 16, 16, 4, 4, 0, false);
    LAUNCH_EPI(32, 32, 8, 2, 2, 0, true);
    LAUNCH_EPI(32, 32, 8, 2, 2, 1, true);
    LAUNCH_EPI(32, 32, 8, 2, 2, 2, true);
    LAUNCH_EPI(32, 32, 8, 2, 2, 3, true);
    LAUNCH_EPI(32, 32, 8, 4, 4, 0, false);
    LAUNCH_EPI(32, 64, 8, 2, 2, 0, false);
    LAUNCH_EPI(32, 64, 8, 4, 4, 0, false);
    LAUNCH_EPI(64, 32, 8, 2, 2, 0, false);
    LAUNCH_EPI(64, 32, 8, 4, 4, 0, false);
    LAUNCH_EPI(64, 64, 8, 2, 2, 0, false);
    LAUNCH_EPI(64, 64, 8, 4, 4, 0, true);
    LAUNCH_EPI(64, 64, 8, 4, 4, 1, true);
    LAUNCH_EPI(64, 64, 8, 4, 4, 2, true);
    LAUNCH_EPI(64, 64, 8, 4, 4, 3, true);
    LAUNCH_EPI(64, 64, 8, 8, 8, 0, false);
#undef LAUNCH_EPI
    return false;
}

template <typename TIn, typename TOut>
static void launch_naive(dim3 grid, dim3 block, const TIn* dA, const TIn* dB, TOut* dC, int M,
                         int N, int K) {
    gemm_naive_kernel<TIn, TOut><<<grid, block>>>(dA, dB, dC, M, N, K);
}

// ---------------------------------------------------------------------------
// 分离 epilogue 链
// ---------------------------------------------------------------------------
template <typename TIn, typename TOut>
static void launch_sep(const Config& cfg, int act, int sep_count, dim3 grid, dim3 block,
                       const TIn* dA, const TIn* dB, const float* dBias, TOut* dC, int M, int N,
                       int K, size_t total) {
    // 先跑纯 GEMM（EPI=0）
    if (!launch_tiled<TIn, TOut>(cfg, 0, grid, block, dA, dB, nullptr, dC, M, N, K)) {
        fprintf(stderr, "[GPU-ERROR] sep 模式的 GEMM 配置不可用\n");
        exit(2);
    }
    dim3 ew_grid((unsigned)((total + 255) / 256));
    dim3 ew_block(256);
    if (act == 1) {
        // bias-only 分离链：GEMM → bias_kernel（1 次额外读+写 C）
        if (sep_count == 2) {
            bias_kernel<TOut><<<ew_grid, ew_block>>>(dC, dBias, total, N);
            return;
        }
        fprintf(stderr, "[GPU-ERROR] bias-only 分离链仅支持 sep=2\n");
        exit(2);
    }
    if (sep_count == 3) {
        bias_kernel<TOut><<<ew_grid, ew_block>>>(dC, dBias, total, N);
        if (act == 2)
            act_kernel<TOut, 2><<<ew_grid, ew_block>>>(dC, dC, total);
        else
            act_kernel<TOut, 3><<<ew_grid, ew_block>>>(dC, dC, total);
    } else {  // sep2：bias+act 合并为一个分离 kernel
        if (act == 2)
            bias_act_kernel<TOut, 2><<<ew_grid, ew_block>>>(dC, dBias, total, N);
        else
            bias_act_kernel<TOut, 3><<<ew_grid, ew_block>>>(dC, dBias, total, N);
    }
}

// ---------------------------------------------------------------------------
// WMMA 分发（5 个 tile 形状 × 2 种累加 dtype；KSTEP 恒为 16）
// ---------------------------------------------------------------------------
#if defined(HAVE_WMMA) && !defined(DISABLE_FP16)
struct WmmaConfig {
    int tm, tn;
    const char* name;
};

static const std::vector<WmmaConfig>& wmma_table() {
    static const std::vector<WmmaConfig> table = {
        {16, 16, "wm16x16"}, {32, 32, "wm32x32"}, {32, 64, "wm32x64"},
        {64, 32, "wm64x32"}, {64, 64, "wm64x64"},
    };
    return table;
}

static bool launch_wmma(int tm, int tn, bool acc_float, dim3 grid, dim3 block,
                        const KERNEL_HALF_TYPE* dA, const KERNEL_HALF_TYPE* dB,
                        KERNEL_HALF_TYPE* dC, int M, int N, int K) {
#define LAUNCH_WMMA(TMM, TNN)                                                              \
    if (tm == TMM && tn == TNN) {                                                          \
        if (acc_float)                                                                     \
            gemm_wmma_kernel<TMM, TNN, 16, 1><<<grid, block>>>(dA, dB, dC, M, N, K);       \
        else                                                                               \
            gemm_wmma_kernel<TMM, TNN, 16, 0><<<grid, block>>>(dA, dB, dC, M, N, K);       \
        return true;                                                                       \
    }
    LAUNCH_WMMA(16, 16);
    LAUNCH_WMMA(32, 32);
    LAUNCH_WMMA(32, 64);
    LAUNCH_WMMA(64, 32);
    LAUNCH_WMMA(64, 64);
#undef LAUNCH_WMMA
    return false;
}
#endif  // HAVE_WMMA && !DISABLE_FP16

// ---------------------------------------------------------------------------
// 计时（事件对口径，与 torch OperatorTimer / 第 01 期合同一致）
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
// JSON 输出 helper（单行，stdout 最后一行；stderr 走日志）
// ---------------------------------------------------------------------------
static std::string fnum(float v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.6f", v);
    return buf;
}

// ---------------------------------------------------------------------------
// bench 模式：单（shape × 配置 × dtype × epilogue）测延迟
// ---------------------------------------------------------------------------
static void mode_bench(const std::map<std::string, std::string>& args) {
    int m = std::stoi(args.at("--m"));
    int n = std::stoi(args.at("--n"));
    int k = std::stoi(args.at("--k"));
    std::string dtype = args.count("--dtype") ? args.at("--dtype") : "fp32";
    std::string kernel_kind = args.count("--kernel") ? args.at("--kernel") : "tiled";
    std::string tile = args.count("--tile") ? args.at("--tile") : "32x32x8";
    std::string reg = args.count("--reg") ? args.at("--reg") : "2x2";
    int epilogue = args.count("--epilogue") ? std::stoi(args.at("--epilogue")) : 0;
    int sep = args.count("--sep") ? std::stoi(args.at("--sep")) : 0;  // 0 融合 / 2 / 3
    int warmup = args.count("--warmup") ? std::stoi(args.at("--warmup")) : 10;
    int repeats = args.count("--repeats") ? std::stoi(args.at("--repeats")) : 50;
    unsigned long long seed = args.count("--seed") ? std::stoull(args.at("--seed")) : 20261118ULL;

    // WMMA 分支用 tile 字符串的前两维直接分发（"32x32x16" 等 wm16 映射名不在
    // 朴素版配置表中），find_config 仅对 tiled/naive 分支要求
    const Config* cfg = (kernel_kind == "wmma") ? find_config("32x32x8", "2x2")
                                                : find_config(tile, reg);
    if (!cfg) {
        fprintf(stderr, "[ERROR] 未知配置 tile=%s reg=%s\n", tile.c_str(), reg.c_str());
        exit(2);
    }

    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, 0));

    size_t total = (size_t)m * n;
    bool use_half = (dtype == "fp16");
#if defined(DISABLE_FP16)
    if (use_half) {
        fprintf(stderr, "[ERROR] 本二进制编译时禁用了 FP16（-DDISABLE_FP16）\n");
        exit(2);
    }
#endif
    size_t in_bytes = (size_t)m * k * (use_half ? 2 : 4);
    size_t bn_bytes = (size_t)k * n * (use_half ? 2 : 4);
    size_t out_bytes = total * (use_half ? 2 : 4);

    char* dA;
    char* dB;
    char* dC;
    float* dBias;
    GPU_CHECK(gpuMalloc(&dA, in_bytes));
    GPU_CHECK(gpuMalloc(&dB, bn_bytes));
    GPU_CHECK(gpuMalloc(&dC, out_bytes));
    GPU_CHECK(gpuMalloc((char**)&dBias, (size_t)n * 4));
    {
        dim3 fb(256);
        if (use_half) {
#if !defined(DISABLE_FP16)
            fill_random_half<<<dim3((unsigned)(((size_t)m * k + 255) / 256)), fb>>>(
                (KERNEL_HALF_TYPE*)dA, (size_t)m * k, seed);
            check_launch("fill A");
            fill_random_half<<<dim3((unsigned)(((size_t)k * n + 255) / 256)), fb>>>(
                (KERNEL_HALF_TYPE*)dB, (size_t)k * n, seed + 1);
            check_launch("fill B");
#endif
        } else {
            fill_random_float<<<dim3((unsigned)(((size_t)m * k + 255) / 256)), fb>>>(
                (float*)dA, (size_t)m * k, seed);
            check_launch("fill A");
            fill_random_float<<<dim3((unsigned)(((size_t)k * n + 255) / 256)), fb>>>(
                (float*)dB, (size_t)k * n, seed + 1);
            check_launch("fill B");
        }
        fill_random_float<<<dim3((unsigned)((n + 255) / 256)), fb>>>(dBias, n, seed + 2);
        check_launch("fill bias");
        GPU_CHECK(gpuDeviceSynchronize());
    }

    dim3 block(cfg->tn / cfg->rn, cfg->tm / cfg->rm);
    dim3 grid((unsigned)((n + cfg->tn - 1) / cfg->tn), (unsigned)((m + cfg->tm - 1) / cfg->tm));

    TimingStats st;
    bool acc_float = true;
    if (args.count("--acc")) acc_float = (args.at("--acc") == "fp32");
    if (kernel_kind == "wmma") {
#if defined(HAVE_WMMA) && !defined(DISABLE_FP16)
        if (!use_half) {
            fprintf(stderr, "[ERROR] WMMA fragment 输入为 FP16（请 --dtype fp16）\n");
            exit(2);
        }
        if (epilogue != 0 || sep != 0) {
            fprintf(stderr, "[ERROR] 第 18 期 WMMA 仅 gemm_only（epilogue 融合归第 17 期朴素版）\n");
            exit(2);
        }
        int tm = 0, tn = 0, tk_parse = 0;
        if (sscanf(tile.c_str(), "%dx%dx%d", &tm, &tn, &tk_parse) != 3) {
            fprintf(stderr, "[ERROR] tile 参数格式应为 TMxTNxTK：%s\n", tile.c_str());
            exit(2);
        }
        dim3 wblock((tm / 16) * (tn / 16) * 32);
        dim3 wgrid((unsigned)((n + tn - 1) / tn), (unsigned)((m + tm - 1) / tm));
        st = time_launches(warmup, repeats, [&] {
            if (!launch_wmma(tm, tn, acc_float, wgrid, wblock, (const KERNEL_HALF_TYPE*)dA,
                             (const KERNEL_HALF_TYPE*)dB, (KERNEL_HALF_TYPE*)dC, m, n, k)) {
                fprintf(stderr, "[ERROR] 未知 WMMA tile %dx%d\n", tm, tn);
                exit(2);
            }
            check_launch("wmma launch");
        });
        block = wblock;
        grid = wgrid;
        // k-step 映射：TILE_K=8 → wm16（2 个 k-fragment），JSON 中记录实际值
        tile = std::to_string(tm) + "x" + std::to_string(tn) + "x16";
        reg = "wm";
#else
        fprintf(stderr, "[ERROR] 本二进制未编译 WMMA（DISABLE_WMMA 或平台头文件缺失）\n");
        exit(2);
#endif
    } else if (kernel_kind == "naive") {
        dim3 nblock(16, 16);
        dim3 ngrid((unsigned)((n + 15) / 16), (unsigned)((m + 15) / 16));
        if (use_half) {
#if !defined(DISABLE_FP16)
            st = time_launches(warmup, repeats,
                               [&] { launch_naive<KERNEL_HALF_TYPE, KERNEL_HALF_TYPE>(ngrid, nblock, (const KERNEL_HALF_TYPE*)dA, (const KERNEL_HALF_TYPE*)dB, (KERNEL_HALF_TYPE*)dC, m, n, k); });
#endif
        } else {
            st = time_launches(warmup, repeats,
                               [&] { launch_naive<float, float>(ngrid, nblock, (const float*)dA, (const float*)dB, (float*)dC, m, n, k); });
        }
        block = nblock;
        grid = ngrid;
    } else if (sep > 0 && epilogue >= 1) {
        // 分离 epilogue 链（sep 个 kernel：GEMM + bias(+act)）
        if (use_half) {
#if !defined(DISABLE_FP16)
            st = time_launches(warmup, repeats, [&] {
                launch_sep<KERNEL_HALF_TYPE, KERNEL_HALF_TYPE>(*cfg, epilogue, sep, grid, block,
                                                               (const KERNEL_HALF_TYPE*)dA,
                                                               (const KERNEL_HALF_TYPE*)dB, dBias,
                                                               (KERNEL_HALF_TYPE*)dC, m, n, k, total);
            });
#endif
        } else {
            st = time_launches(warmup, repeats, [&] {
                launch_sep<float, float>(*cfg, epilogue, sep, grid, block, (const float*)dA,
                                         (const float*)dB, dBias, (float*)dC, m, n, k, total);
            });
        }
    } else {
        if (use_half) {
#if !defined(DISABLE_FP16)
            st = time_launches(warmup, repeats, [&] {
                if (!launch_tiled<KERNEL_HALF_TYPE, KERNEL_HALF_TYPE>(
                        *cfg, epilogue, grid, block, (const KERNEL_HALF_TYPE*)dA,
                        (const KERNEL_HALF_TYPE*)dB, dBias, (KERNEL_HALF_TYPE*)dC, m, n, k)) {
                    fprintf(stderr, "[ERROR] 配置 %s 不支持 epilogue=%d（未实例化）\n", cfg->name,
                            epilogue);
                    exit(2);
                }
                check_launch("tiled launch");
            });
#endif
        } else {
            st = time_launches(warmup, repeats, [&] {
                if (!launch_tiled<float, float>(*cfg, epilogue, grid, block, (const float*)dA,
                                                (const float*)dB, dBias, (float*)dC, m, n, k)) {
                    fprintf(stderr, "[ERROR] 配置 %s 不支持 epilogue=%d（未实例化）\n", cfg->name,
                            epilogue);
                    exit(2);
                }
                check_launch("tiled launch");
            });
        }
    }

    double flops = 2.0 * m * n * k;
    // 名义字节口径：
    //   - GEMM 本体：读 A + 读 B + 写 C（bias 向量 N*4B 忽略，与 roofline_gemm.py 同口径）；
    //   - 分离 epilogue（sep=2/3）：额外 (sep-1) 次"读 C + 写 C"→ 2*(sep-1)*MN*itemsize；
    //   - 融合 epilogue（sep=0）：无额外 C 流量。
    double bytes = ((double)m * k + (double)k * n + (double)m * n) * (use_half ? 2 : 4);
    if (sep > 0) bytes += (double)(2 * (sep - 1)) * total * (use_half ? 2 : 4);

    size_t json_smem = smem_bytes_of(*cfg);
#if defined(HAVE_WMMA) && !defined(DISABLE_FP16)
    if (kernel_kind == "wmma") {
        int wm_tm = 0, wm_tn = 0;
        sscanf(tile.c_str(), "%dx%d", &wm_tm, &wm_tn);
        json_smem = (size_t)(wm_tm * (16 + 8) + 16 * (wm_tn + 8)) * 2 +
                    (acc_float ? (size_t)wm_tm * wm_tn * 4 : (size_t)wm_tm * (wm_tn + 8) * 2);
    }
#endif

    std::string line;
    line += "{\"mode\":\"bench\"";
    line += ",\"backend_binary\":\"naive_gemm_bench\"";
    line += ",\"dtype\":\"" + dtype + "\"";
    line += ",\"kernel\":\"" + kernel_kind + "\"";
    line += ",\"tile\":\"" + tile + "\"";
    line += ",\"reg\":\"" + reg + "\"";
    if (kernel_kind == "wmma") {
        line += std::string(",\"acc\":\"") + (acc_float ? "fp32" : "fp16") + "\",\"kstep\":16";
    }
    line += ",\"epilogue\":" + std::to_string(epilogue);
    line += ",\"sep\":" + std::to_string(sep);
    char tmp[256];
    snprintf(tmp, sizeof(tmp), ",\"m\":%d,\"n\":%d,\"k\":%d,\"grid\":[%u,%u],\"block\":[%u,%u]", m, n,
             k, grid.x, grid.y, block.x, block.y);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"warps_per_block\":%u", (block.x * block.y + 31) / 32);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"smem_bytes_static\":%zu", json_smem);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"flops\":%.0f,\"nominal_bytes\":%.0f", flops, bytes);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"warmup\":%d,\"repeats\":%d", st.warmup, st.repeats);
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
    snprintf(tmp, sizeof(tmp), ",\"tflops_p50\":%.4f", flops / (st.p50_ms / 1000.0) / 1e12);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"effective_gb_s_p50\":%.2f", bytes / (st.p50_ms / 1000.0) / 1e9);
    line += tmp;
    line += ",\"device_name\":\"" + std::string(prop.name) + "\"";
    // clockRate：CUDA 13 的 cudaDeviceProp 已无该字段（ch02 实测记录），统一记 -1
    snprintf(tmp, sizeof(tmp), ",\"sm_count\":%d,\"warp_size\":%d,\"clock_khz\":-1",
             prop.multiProcessorCount, prop.warpSize);
    line += tmp;
    line += "}";
    printf("%s\n", line.c_str());

    GPU_CHECK(gpuFree(dA));
    GPU_CHECK(gpuFree(dB));
    GPU_CHECK(gpuFree(dC));
    GPU_CHECK(gpuFree(dBias));
}

// ---------------------------------------------------------------------------
// run 模式：单次执行（正确性验证用），文件输入输出（raw bytes）
//   --in-a / --in-b / [--in-bias] / --out-c；A/B 的 dtype 随 --dtype
//   （fp16 输入文件是 float16 raw bytes，本二进制不做 host 端 half 转换）
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

static void mode_run(const std::map<std::string, std::string>& args) {
    int m = std::stoi(args.at("--m"));
    int n = std::stoi(args.at("--n"));
    int k = std::stoi(args.at("--k"));
    std::string dtype = args.count("--dtype") ? args.at("--dtype") : "fp32";
    std::string kernel_kind = args.count("--kernel") ? args.at("--kernel") : "tiled";
    std::string tile = args.count("--tile") ? args.at("--tile") : "32x32x8";
    std::string reg = args.count("--reg") ? args.at("--reg") : "2x2";
    int epilogue = args.count("--epilogue") ? std::stoi(args.at("--epilogue")) : 0;
    int sep = args.count("--sep") ? std::stoi(args.at("--sep")) : 0;

    // WMMA 分支用 tile 字符串的前两维直接分发（"32x32x16" 等 wm16 映射名不在
    // 朴素版配置表中），find_config 仅对 tiled/naive 分支要求
    const Config* cfg = (kernel_kind == "wmma") ? find_config("32x32x8", "2x2")
                                                : find_config(tile, reg);
    if (!cfg) {
        fprintf(stderr, "[ERROR] 未知配置 tile=%s reg=%s\n", tile.c_str(), reg.c_str());
        exit(2);
    }
    if ((kernel_kind == "naive" || kernel_kind == "wmma") && (epilogue != 0 || sep != 0)) {
        fprintf(stderr, "[ERROR] naive/wmma kernel 仅支持 epilogue=0（epilogue 对照归第 17 期朴素版）\n");
        exit(2);
    }
#if defined(DISABLE_FP16)
    if (use_half) {
        fprintf(stderr, "[ERROR] 本二进制编译时禁用了 FP16（-DDISABLE_FP16）\n");
        exit(2);
    }
#endif

    bool use_half = (dtype == "fp16");
    std::vector<char> a_data = read_file_bytes(args.at("--in-a"));
    std::vector<char> b_data = read_file_bytes(args.at("--in-b"));
    std::vector<char> bias_data;
    bool has_bias = args.count("--in-bias") > 0;
    if (has_bias) bias_data = read_file_bytes(args.at("--in-bias"));
    size_t expect_ab = (size_t)m * k * (use_half ? 2 : 4);
    size_t expect_bb = (size_t)k * n * (use_half ? 2 : 4);
    if (a_data.size() != expect_ab || b_data.size() != expect_bb) {
        fprintf(stderr, "[ERROR] 输入尺寸不符：A=%zu（期望 %zu） B=%zu（期望 %zu）\n",
                a_data.size(), expect_ab, b_data.size(), expect_bb);
        exit(2);
    }
    if (epilogue >= 1 && !has_bias) {
        fprintf(stderr, "[ERROR] epilogue>=1 需要 --in-bias\n");
        exit(2);
    }

    char* dA;
    char* dB;
    char* dC;
    float* dBias = nullptr;
    size_t total = (size_t)m * n;
    GPU_CHECK(gpuMalloc(&dA, expect_ab));
    GPU_CHECK(gpuMalloc(&dB, expect_bb));
    GPU_CHECK(gpuMalloc(&dC, total * (use_half ? 2 : 4)));
    GPU_CHECK(gpuMemcpy(dA, a_data.data(), expect_ab, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(dB, b_data.data(), expect_bb, gpuMemcpyHostToDevice));
    if (has_bias) {
        GPU_CHECK(gpuMalloc((char**)&dBias, (size_t)n * 4));
        GPU_CHECK(gpuMemcpy(dBias, bias_data.data(), (size_t)n * 4, gpuMemcpyHostToDevice));
    }

    dim3 block(cfg->tn / cfg->rn, cfg->tm / cfg->rm);
    dim3 grid((unsigned)((n + cfg->tn - 1) / cfg->tn), (unsigned)((m + cfg->tm - 1) / cfg->tm));

    if (kernel_kind == "wmma") {
#if defined(HAVE_WMMA) && !defined(DISABLE_FP16)
        if (!use_half) {
            fprintf(stderr, "[ERROR] WMMA fragment 输入为 FP16（请 --dtype fp16）\n");
            exit(2);
        }
        bool acc_float = (args.count("--acc") == 0 || args.at("--acc") == "fp32");
        int tm = 0, tn = 0, tk_parse = 0;
        if (sscanf(tile.c_str(), "%dx%dx%d", &tm, &tn, &tk_parse) != 3) {
            fprintf(stderr, "[ERROR] tile 参数格式应为 TMxTNxTK：%s\n", tile.c_str());
            exit(2);
        }
        dim3 wblock((tm / 16) * (tn / 16) * 32);
        dim3 wgrid((unsigned)((n + tn - 1) / tn), (unsigned)((m + tm - 1) / tm));
        if (!launch_wmma(tm, tn, acc_float, wgrid, wblock, (const KERNEL_HALF_TYPE*)dA,
                         (const KERNEL_HALF_TYPE*)dB, (KERNEL_HALF_TYPE*)dC, m, n, k)) {
            fprintf(stderr, "[ERROR] 未知 WMMA tile %dx%d\n", tm, tn);
            exit(2);
        }
        check_launch("wmma launch");
#else
        fprintf(stderr, "[ERROR] 本二进制未编译 WMMA（DISABLE_WMMA 或平台头文件缺失）\n");
        exit(2);
#endif
    } else if (kernel_kind == "naive") {
        dim3 nblock(16, 16);
        dim3 ngrid((unsigned)((n + 15) / 16), (unsigned)((m + 15) / 16));
        if (use_half) {
#if !defined(DISABLE_FP16)
            launch_naive<KERNEL_HALF_TYPE, KERNEL_HALF_TYPE>(ngrid, nblock,
                                                             (const KERNEL_HALF_TYPE*)dA,
                                                             (const KERNEL_HALF_TYPE*)dB,
                                                             (KERNEL_HALF_TYPE*)dC, m, n, k);
#endif
        } else {
            launch_naive<float, float>(ngrid, nblock, (const float*)dA, (const float*)dB,
                                       (float*)dC, m, n, k);
        }
        check_launch("naive launch");
    } else if (sep > 0 && epilogue >= 1) {
        if (use_half) {
#if !defined(DISABLE_FP16)
            launch_sep<KERNEL_HALF_TYPE, KERNEL_HALF_TYPE>(*cfg, epilogue, sep, grid, block,
                                                           (const KERNEL_HALF_TYPE*)dA,
                                                           (const KERNEL_HALF_TYPE*)dB, dBias,
                                                           (KERNEL_HALF_TYPE*)dC, m, n, k, total);
#endif
        } else {
            launch_sep<float, float>(*cfg, epilogue, sep, grid, block, (const float*)dA,
                                     (const float*)dB, dBias, (float*)dC, m, n, k, total);
        }
        check_launch("sep launch");
    } else {
        if (use_half) {
#if !defined(DISABLE_FP16)
            if (!launch_tiled<KERNEL_HALF_TYPE, KERNEL_HALF_TYPE>(*cfg, epilogue, grid, block,
                                                                  (const KERNEL_HALF_TYPE*)dA,
                                                                  (const KERNEL_HALF_TYPE*)dB,
                                                                  dBias, (KERNEL_HALF_TYPE*)dC, m,
                                                                  n, k)) {
                fprintf(stderr, "[ERROR] 配置 %s 不支持 epilogue=%d\n", cfg->name, epilogue);
                exit(2);
            }
#endif
        } else {
            if (!launch_tiled<float, float>(*cfg, epilogue, grid, block, (const float*)dA,
                                            (const float*)dB, dBias, (float*)dC, m, n, k)) {
                fprintf(stderr, "[ERROR] 配置 %s 不支持 epilogue=%d\n", cfg->name, epilogue);
                exit(2);
            }
        }
        check_launch("tiled launch");
    }
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<char> out(total * (use_half ? 2 : 4));
    GPU_CHECK(gpuMemcpy(out.data(), dC, out.size(), gpuMemcpyDeviceToHost));
    write_file_bytes(args.at("--out-c"), out.data(), out.size());

    GPU_CHECK(gpuFree(dA));
    GPU_CHECK(gpuFree(dB));
    GPU_CHECK(gpuFree(dC));
    if (dBias) GPU_CHECK(gpuFree(dBias));
}

// ---------------------------------------------------------------------------
// selftest 模式：无 fixtures 的最小自检（host 三重循环 vs 各配置，FP32）
// FP16 的自检不在此做（避免 host half 依赖），由 fixtures 正确性链覆盖。
// ---------------------------------------------------------------------------
static void mode_selftest(const std::map<std::string, std::string>& args) {
    (void)args;
    struct Shape {
        int m, k, n;
    };
    std::vector<Shape> shapes = {{32, 24, 40}, {17, 19, 13}, {4, 8, 4}};
    int passed = 0, failed = 0;

    for (const auto& s : shapes) {
        size_t count_a = (size_t)s.m * s.k, count_b = (size_t)s.k * s.n;
        std::vector<float> hA(count_a), hB(count_b);
        unsigned long long seed = 20261118ULL + s.m * 131 + s.k * 17 + s.n;
        for (size_t i = 0; i < count_a; ++i)
            hA[i] = ((float)hash_u32(i, seed) / 2147483648.0f) - 1.0f;
        for (size_t i = 0; i < count_b; ++i)
            hB[i] = ((float)hash_u32(i, seed + 1) / 2147483648.0f) - 1.0f;
        // host 参考：k 升序 float 累加
        std::vector<float> hRef((size_t)s.m * s.n, 0.0f);
        for (int i = 0; i < s.m; ++i)
            for (int p = 0; p < s.k; ++p) {
                float a = hA[(size_t)i * s.k + p];
                for (int j = 0; j < s.n; ++j) hRef[(size_t)i * s.n + j] += a * hB[(size_t)p * s.n + j];
            }

        float *dA, *dB, *dC;
        GPU_CHECK(gpuMalloc(&dA, count_a * 4));
        GPU_CHECK(gpuMalloc(&dB, count_b * 4));
        GPU_CHECK(gpuMalloc(&dC, (size_t)s.m * s.n * 4));
        GPU_CHECK(gpuMemcpy(dA, hA.data(), count_a * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(dB, hB.data(), count_b * 4, gpuMemcpyHostToDevice));

        // naive kernel
        {
            dim3 nb(16, 16);
            dim3 ng((unsigned)((s.n + 15) / 16), (unsigned)((s.m + 15) / 16));
            launch_naive<float, float>(ng, nb, dA, dB, dC, s.m, s.n, s.k);
            check_launch("selftest naive");
            GPU_CHECK(gpuDeviceSynchronize());
            std::vector<float> out((size_t)s.m * s.n);
            GPU_CHECK(gpuMemcpy(out.data(), dC, out.size() * 4, gpuMemcpyDeviceToHost));
            float max_abs = 0;
            for (size_t i = 0; i < out.size(); ++i) max_abs = std::max(max_abs, fabsf(out[i] - hRef[i]));
            bool ok = max_abs < 1e-3f;
            printf("SELFTEST kernel=naive shape=%dx%dx%d max_abs=%.3e %s\n", s.m, s.k, s.n, max_abs,
                   ok ? "PASS" : "FAIL");
            ok ? passed++ : failed++;
        }
        // 各 tiled 配置
        for (const auto& cfg : config_table()) {
            dim3 block(cfg.tn / cfg.rn, cfg.tm / cfg.rm);
            dim3 grid((unsigned)((s.n + cfg.tn - 1) / cfg.tn), (unsigned)((s.m + cfg.tm - 1) / cfg.tm));
            if (!launch_tiled<float, float>(cfg, 0, grid, block, dA, dB, nullptr, dC, s.m, s.n, s.k)) {
                printf("SELFTEST config=%s shape=%dx%dx%d SKIP\n", cfg.name, s.m, s.k, s.n);
                continue;
            }
            check_launch("selftest tiled");
            GPU_CHECK(gpuDeviceSynchronize());
            std::vector<float> out((size_t)s.m * s.n);
            GPU_CHECK(gpuMemcpy(out.data(), dC, out.size() * 4, gpuMemcpyDeviceToHost));
            float max_abs = 0;
            for (size_t i = 0; i < out.size(); ++i) max_abs = std::max(max_abs, fabsf(out[i] - hRef[i]));
            bool ok = max_abs < 1e-3f;
            printf("SELFTEST config=%s shape=%dx%dx%d max_abs=%.3e %s\n", cfg.name, s.m, s.k, s.n,
                   max_abs, ok ? "PASS" : "FAIL");
            ok ? passed++ : failed++;
        }
        GPU_CHECK(gpuFree(dA));
        GPU_CHECK(gpuFree(dB));
        GPU_CHECK(gpuFree(dC));
    }
    printf("{\"mode\":\"selftest\",\"passed\":%d,\"failed\":%d}\n", passed, failed);
    if (failed > 0) exit(3);
}

// ---------------------------------------------------------------------------
// selftest 补充段：WMMA kernel 自检（第 18 期新增；无 fixtures 最小验证）
// fp32 累加容差同 tiled；fp16 累加容差放宽（K=40 的 half 累加舍入 ~0.25）
// ---------------------------------------------------------------------------
static void mode_selftest_wmma() {
#if defined(HAVE_WMMA) && !defined(DISABLE_FP16)
    struct Shape {
        int m, k, n;
    };
    std::vector<Shape> shapes = {{32, 24, 40}, {17, 19, 13}, {4, 8, 4}};
    int passed = 0, failed = 0;
    for (const auto& s : shapes) {
        size_t count_a = (size_t)s.m * s.k, count_b = (size_t)s.k * s.n;
        std::vector<float> hA(count_a), hB(count_b);
        unsigned long long seed = 20261119ULL + s.m * 131 + s.k * 17 + s.n;
        for (size_t i = 0; i < count_a; ++i)
            hA[i] = ((float)hash_u32(i, seed) / 2147483648.0f) - 1.0f;
        for (size_t i = 0; i < count_b; ++i)
            hB[i] = ((float)hash_u32(i, seed + 1) / 2147483648.0f) - 1.0f;
        std::vector<float> hRef((size_t)s.m * s.n, 0.0f);
        for (int i = 0; i < s.m; ++i)
            for (int p = 0; p < s.k; ++p) {
                float a = hA[(size_t)i * s.k + p];
                for (int j = 0; j < s.n; ++j) hRef[(size_t)i * s.n + j] += a * hB[(size_t)p * s.n + j];
            }

        // host 生成 half 输入（位级 RNE，与 kernel 侧同源）
        std::vector<KERNEL_HALF_TYPE> hAh(count_a), hBh(count_b);
        for (size_t i = 0; i < count_a; ++i) hAh[i] = to_type<KERNEL_HALF_TYPE>(hA[i]);
        for (size_t i = 0; i < count_b; ++i) hBh[i] = to_type<KERNEL_HALF_TYPE>(hB[i]);

        KERNEL_HALF_TYPE *dA, *dB, *dC;
        GPU_CHECK(gpuMalloc(&dA, count_a * sizeof(KERNEL_HALF_TYPE)));
        GPU_CHECK(gpuMalloc(&dB, count_b * sizeof(KERNEL_HALF_TYPE)));
        GPU_CHECK(gpuMalloc(&dC, (size_t)s.m * s.n * sizeof(KERNEL_HALF_TYPE)));
        GPU_CHECK(gpuMemcpy(dA, hAh.data(), count_a * sizeof(KERNEL_HALF_TYPE), gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(dB, hBh.data(), count_b * sizeof(KERNEL_HALF_TYPE), gpuMemcpyHostToDevice));

        for (int acc_float = 1; acc_float >= 0; --acc_float) {
            for (const auto& wc : wmma_table()) {
                dim3 block((wc.tm / 16) * (wc.tn / 16) * 32);
                dim3 grid((unsigned)((s.n + wc.tn - 1) / wc.tn),
                          (unsigned)((s.m + wc.tm - 1) / wc.tm));
                if (!launch_wmma(wc.tm, wc.tn, acc_float == 1, grid, block, dA, dB, dC, s.m, s.n,
                                 s.k)) {
                    printf("SELFTEST wmma=%s acc=%s shape=%dx%dx%d SKIP\n", wc.name,
                           acc_float ? "fp32" : "fp16", s.m, s.k, s.n);
                    continue;
                }
                check_launch("selftest wmma");
                GPU_CHECK(gpuDeviceSynchronize());
                std::vector<KERNEL_HALF_TYPE> out((size_t)s.m * s.n);
                GPU_CHECK(gpuMemcpy(out.data(), dC, out.size() * sizeof(KERNEL_HALF_TYPE),
                                    gpuMemcpyDeviceToHost));
                float max_abs = 0;
                for (size_t i = 0; i < out.size(); ++i) {
                    float diff = fabsf(to_float_val(out[i]) - hRef[i]);
                    max_abs = std::max(max_abs, diff);
                }
                // 输入已 half 化：fp32 累加也含输入舍入，容差放宽到输入舍入水平
                float tol = acc_float ? 0.05f : 0.5f;
                bool ok = max_abs < tol;
                printf("SELFTEST wmma=%s acc=%s shape=%dx%dx%d max_abs=%.4e %s\n", wc.name,
                       acc_float ? "fp32" : "fp16", s.m, s.k, s.n, max_abs, ok ? "PASS" : "FAIL");
                ok ? passed++ : failed++;
            }
        }
        GPU_CHECK(gpuFree(dA));
        GPU_CHECK(gpuFree(dB));
        GPU_CHECK(gpuFree(dC));
    }
    printf("{\"mode\":\"selftest_wmma\",\"passed\":%d,\"failed\":%d}\n", passed, failed);
    if (failed > 0) exit(3);
#else
    printf("{\"mode\":\"selftest_wmma\",\"note\":\"WMMA 未编译（平台边界或 DISABLE_WMMA）\"}\n");
#endif
}

// ---------------------------------------------------------------------------
// attrs 模式：逐配置寄存器/共享内存占用（寄存器占用估算的实测证据）
// ---------------------------------------------------------------------------
static void mode_attrs(const std::map<std::string, std::string>& args) {
    (void)args;
    std::string json = "{\"mode\":\"attrs\",\"configs\":[";
    bool first = true;
    for (const auto& cfg : config_table()) {
        dim3 block(cfg.tn / cfg.rn, cfg.tm / cfg.rm);
        // fp32 EPI=0 实例的属性（fp16/epilogue 变体属性在编译日志 --resource-usage 里）
        const void* func = nullptr;
#define GET_FUNC(TM, TN, TK, RM, RN)                                    \
    if (cfg.tm == TM && cfg.tn == TN && cfg.tk == TK && cfg.rm == RM && cfg.rn == RN) \
        func = (const void*)&gemm_tiled_kernel<float, float, TM, TN, TK, RM, RN, 0>;
        GET_FUNC(16, 16, 16, 2, 2);
        GET_FUNC(16, 16, 16, 4, 4);
        GET_FUNC(32, 32, 8, 2, 2);
        GET_FUNC(32, 32, 8, 4, 4);
        GET_FUNC(32, 64, 8, 2, 2);
        GET_FUNC(32, 64, 8, 4, 4);
        GET_FUNC(64, 32, 8, 2, 2);
        GET_FUNC(64, 32, 8, 4, 4);
        GET_FUNC(64, 64, 8, 2, 2);
        GET_FUNC(64, 64, 8, 4, 4);
        GET_FUNC(64, 64, 8, 8, 8);
#undef GET_FUNC
        if (!func) continue;
        char buf[512];
#if !defined(DISABLE_FUNC_ATTRS)
        gpuFuncAttributes attr;
        gpuError_t err = gpuFuncGetAttributes(&attr, func);
        if (err == gpuSuccess) {
            snprintf(buf, sizeof(buf),
                     "%s{\"name\":\"%s\",\"block\":[%u,%u],\"threads\":%u,\"num_regs\":%d,"
                     "\"shared_bytes_static\":%zu,\"shared_bytes_attr\":%zu,"
                     "\"local_bytes\":%zu,\"max_threads_per_block\":%d}",
                     first ? "" : ",", cfg.name, block.x, block.y, block.x * block.y, attr.numRegs,
                     smem_bytes_of(cfg), (size_t)attr.sharedSizeBytes, (size_t)attr.localSizeBytes,
                     attr.maxThreadsPerBlock);
        } else
#endif
        {
            snprintf(buf, sizeof(buf),
                     "%s{\"name\":\"%s\",\"block\":[%u,%u],\"threads\":%u,\"num_regs\":null,"
                     "\"shared_bytes_static\":%zu,\"shared_bytes_attr\":null,\"local_bytes\":null,"
                     "\"max_threads_per_block\":null}",
                     first ? "" : ",", cfg.name, block.x, block.y, block.x * block.y,
                     smem_bytes_of(cfg));
        }
        json += buf;
        first = false;
    }
#if defined(HAVE_WMMA) && !defined(DISABLE_FP16)
    // 第 18 期：WMMA kernel 的寄存器/共享内存占用（fp32-acc 实例）
    for (const auto& wc : wmma_table()) {
        dim3 wblock((wc.tm / 16) * (wc.tn / 16) * 32);
        const void* func = nullptr;
#define GET_WMMA_FUNC(TMM, TNN)                                                     \
    if (wc.tm == TMM && wc.tn == TNN)                                               \
        func = (const void*)&gemm_wmma_kernel<TMM, TNN, 16, 1>;
        GET_WMMA_FUNC(16, 16);
        GET_WMMA_FUNC(32, 32);
        GET_WMMA_FUNC(32, 64);
        GET_WMMA_FUNC(64, 32);
        GET_WMMA_FUNC(64, 64);
#undef GET_WMMA_FUNC
        if (!func) continue;
        size_t wsmem = (size_t)(wc.tm * (16 + 8) + 16 * (wc.tn + 8)) * 2 +
                       (size_t)wc.tm * wc.tn * 4;
        char wbuf[512];
#if !defined(DISABLE_FUNC_ATTRS)
        gpuFuncAttributes wattr;
        gpuError_t werr = gpuFuncGetAttributes(&wattr, func);
        if (werr == gpuSuccess) {
            snprintf(wbuf, sizeof(wbuf),
                     "%s{\"kind\":\"wmma\",\"name\":\"%s\",\"block\":[%u],\"threads\":%u,"
                     "\"num_regs\":%d,\"shared_bytes_static\":%zu,\"shared_bytes_attr\":%zu,"
                     "\"local_bytes\":%zu,\"max_threads_per_block\":%d,\"warps\":%u}",
                     first ? "" : ",", wc.name, wblock.x, wblock.x, wattr.numRegs, wsmem,
                     (size_t)wattr.sharedSizeBytes, (size_t)wattr.localSizeBytes,
                     wattr.maxThreadsPerBlock, wblock.x / 32);
        } else
#endif
        {
            snprintf(wbuf, sizeof(wbuf),
                     "%s{\"kind\":\"wmma\",\"name\":\"%s\",\"block\":[%u],\"threads\":%u,"
                     "\"num_regs\":null,\"shared_bytes_static\":%zu,\"shared_bytes_attr\":null,"
                     "\"local_bytes\":null,\"max_threads_per_block\":null,\"warps\":%u}",
                     first ? "" : ",", wc.name, wblock.x, wblock.x, wsmem, wblock.x / 32);
        }
        json += wbuf;
        first = false;
    }
#endif
    json += "]}";
    printf("%s\n", json.c_str());
}

// ---------------------------------------------------------------------------
// main：极简 key=value 参数解析
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "用法: naive_gemm_bench <selftest|attrs|run|bench> [--key value ...]\n"
                "  bench/run 额外支持 --kernel wmma --acc fp32|fp16（第 18 期）\n");
        return 1;
    }
    std::string mode = argv[1];
    std::map<std::string, std::string> args;
    for (int i = 2; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];

    if (mode == "selftest") {
        mode_selftest(args);
        mode_selftest_wmma();
    } else if (mode == "attrs")
        mode_attrs(args);
    else if (mode == "run")
        mode_run(args);
    else if (mode == "bench")
        mode_bench(args);
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
