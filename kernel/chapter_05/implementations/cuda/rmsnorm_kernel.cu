// ============================================================================
// rmsnorm_kernel.cu —— 第 05 期 CUDA 手写 RMSNorm：dtype 三路径版
// （第 02 期 warpSize-generic Kernel 的扩展：T ∈ {fp32, fp16, bf16} ×
//   累加 dtype ACC ∈ {fp32, fp16} × TILE × VEC 全部编译期特化）
//
// 本期的核心问题：dtype 不只是"精度"——它同时决定访存字节数、转换指令、
// 数值范围与舍入行为。三条 dtype 编译路径的 specialization 差异（对照表）：
//
//   FP32 路径：直接读 float（零转换），平方/累加/归一化全 FP32，直接写回；
//   FP16 路径：读 __half -> __half2float 转 FP32 -> 计算全 FP32 ->
//              写回 __float2half（RNE）一次舍入到 T；
//   BF16 路径：读 __nv_bfloat16 -> __bfloat162float 转 FP32 -> 计算全 FP32 ->
//              写回 __float2bfloat16（RNE，本质是 FP32 位型 + 0x7FFF + 尾数
//              LSB 的偶舍截断）一次舍入到 T；
//   累加路径（ACC 编译期可选，实验 4 的被测变量）：
//     ACC=float（合同默认）：平方、每步累加、warp 蝶形、跨 warp 合并
//         全部 FP32 —— 第 02 期口径不变；
//     ACC=half（对照实现，故意保留的"错误写法"）：平方与每步累加在
//         binary16 内舍入（__hmul/__hadd），warp 蝶形与跨 warp 合并也在
//         half（__shfl_xor_sync 有 __half 重载），仅最后 total->FP32 求
//         mean+eps+rsqrt —— FP16 的 5 位指数意味着平方和 > 65504 即溢出
//         为 inf、10 位尾数意味着每步累加舍入 2^-11 —— 实验演示两者。
//
// 访存组织（沿用第 02 期，本期不改 —— 唯一核心变量是数值表示）：
//   - 每 block 一行（grid-stride），warp 内 __shfl_xor_sync 蝶形归约，
//     跨 warp shared 汇总广播；
//   - 指针 16B 对齐且 H 整除向量宽度时向量化：FP32 float4（4 元素/16B），
//     FP16/BF16 uint4（8 元素/16B）；否则标量回退；
//   - 第一遍把 x 行缓存进 shared（H<=2048），第二遍从 shared 读、写 y。
//
// BF16 的设备类型：CUDA 用 cuda_bf16.h 的 __nv_bfloat16（sm_80+ 原生支持，
// 4090 = sm_89）；MUSA 迁移版对应 musa_bf16.h 的 __mt_bfloat16（转换函数
// __float2bfloat16_rn/__bfloat162float 同名，见 MUSA_PORT_NOTES.md）。
//
// 本文件刻意不依赖 torch：可被 nvcc 单独编译（rmsnorm_bench.cu 使用），
// torch 扩展通过 rmsnorm_ext.cu 包一层。
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mini_infra {

#define RMSNORM_CUDA_CHECK(expr)                                       \
    do {                                                               \
        cudaError_t _err = (expr);                                     \
        if (_err != cudaSuccess) {                                     \
            throw std::runtime_error(std::string("CUDA error at ") +   \
                                     __FILE__ + ":" +                  \
                                     std::to_string(__LINE__) + " " +  \
                                     #expr + ": " +                    \
                                     cudaGetErrorString(_err));        \
        }                                                              \
    } while (0)

// 合同：支持的最大 H（H<=2048 时 Pass 2 从 shared 读 x，免重读 global）。
constexpr int kMaxCacheElems = 2048;
constexpr long kMaxGridBlocks = 1L << 20;

// ---------------------------------------------------------------------------
// 向量化访存适配层（specialization 差异之一）：
//   FP32: float4 16B 载荷（4 元素）；FP16/BF16: uint4 16B 载荷（8 元素）。
// ---------------------------------------------------------------------------
template <typename T>
struct VecTraits;

template <>
struct VecTraits<float> {
    static constexpr int kWidth = 4;
    using Pack = float4;
};

template <>
struct VecTraits<__half> {
    static constexpr int kWidth = 8;
    using Pack = uint4;
};

template <>
struct VecTraits<__nv_bfloat16> {
    static constexpr int kWidth = 8;
    using Pack = uint4;
};

// dtype 转换适配：读入转 FP32 / 写回从 FP32 舍入到 T（specialization 差异之二）。
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T cast_from_float(float v);

template <>
__device__ __forceinline__ float cast_from_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ __half cast_from_float<__half>(float v) { return __float2half(v); }

template <>
__device__ __forceinline__ __nv_bfloat16 cast_from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);  // RNE
}

// 向量化载荷 <-> 标量（模板化：float4 走 x/y/z/w 成员；uint4 走 reinterpret，
// __half 与 __nv_bfloat16 均为 2 字节 POD，位宽一致）。
template <typename T>
__device__ __forceinline__ float pack_get(const typename VecTraits<T>::Pack& p, int k);

template <>
__device__ __forceinline__ float pack_get<float>(const float4& p, int k) {
    return (&p.x)[k];
}

template <>
__device__ __forceinline__ float pack_get<__half>(const uint4& p, int k) {
    return __half2float(reinterpret_cast<const __half*>(&p)[k]);
}

template <>
__device__ __forceinline__ float pack_get<__nv_bfloat16>(const uint4& p, int k) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&p)[k]);
}

template <typename T>
__device__ __forceinline__ void pack_set(typename VecTraits<T>::Pack& p, int k, float v);

template <>
__device__ __forceinline__ void pack_set<float>(float4& p, int k, float v) {
    (&p.x)[k] = v;
}

template <>
__device__ __forceinline__ void pack_set<__half>(uint4& p, int k, float v) {
    reinterpret_cast<__half*>(&p)[k] = __float2half(v);
}

template <>
__device__ __forceinline__ void pack_set<__nv_bfloat16>(uint4& p, int k, float v) {
    reinterpret_cast<__nv_bfloat16*>(&p)[k] = __float2bfloat16(v);
}

// ---------------------------------------------------------------------------
// 累加 dtype 策略（specialization 差异之三 —— 实验 4 的被测变量）：
//   ACC=float: 平方、累加全 FP32（合同默认）；
//   ACC=half : 平方 __hmul、累加 __hadd 都在 binary16 内舍入（故意保留的
//              "错误写法"，用于量化 FP32 累加的必要性）。
// 边界语义（如实记录）：ACC=half 时 |x| > ~256 的元素平方即溢出为 inf，
//   平方和 > 65504 溢出为 inf；total(inf) -> FP32 后 rsqrtf(inf)=0 -> 输出全零
//   （与 torch rsqrt(inf)=0 行为一致，无 NaN 传播）。
// ---------------------------------------------------------------------------
template <typename ACC>
struct AccTraits;

template <>
struct AccTraits<float> {
    using type = float;
    static constexpr const char* kName = "fp32";
    __device__ __forceinline__ static float zero() { return 0.0f; }
    __device__ __forceinline__ static float square_add(float acc, float f) {
        return acc + f * f;  // 平方与加法都在 FP32（每步舍入 2^-24 相对）
    }
    __device__ __forceinline__ static float to_float_total(float acc) { return acc; }
};

template <>
struct AccTraits<__half> {
    using type = __half;
    static constexpr const char* kName = "fp16";
    __device__ __forceinline__ static __half zero() { return __float2half(0.0f); }
    __device__ __forceinline__ static __half square_add(__half acc, float f) {
        const __half fh = __float2half(f);            // FP32 -> FP16（T=fp32 时有损）
        return __hadd(acc, __hmul(fh, fh));           // 平方与累加都在 FP16 内舍入
    }
    __device__ __forceinline__ static float to_float_total(__half acc) {
        return __half2float(acc);  // inf 按位传染到 FP32
    }
};

// ---------------------------------------------------------------------------
// 归约原语（先声明后用；两个调用点共用）：
//   shfl_xor：float / __half 各自走 __shfl_xor_sync 的原生重载；
//   acc_combine：蝶形合并，在 ACC 算术内相加（half 蝶形 = __hadd 舍入链）。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float shfl_xor(float v, int off) {
    return __shfl_xor_sync(0xffffffffu, v, off);
}

__device__ __forceinline__ __half shfl_xor(__half v, int off) {
    return __shfl_xor_sync(0xffffffffu, v, off);
}

template <typename ACC>
__device__ __forceinline__ ACC acc_combine(ACC a, ACC b);

template <>
__device__ __forceinline__ float acc_combine<float>(float a, float b) { return a + b; }

template <>
__device__ __forceinline__ __half acc_combine<__half>(__half a, __half b) {
    return __hadd(a, b);  // binary16 舍入的合并
}

template <typename ACC>
__device__ __forceinline__ ACC warp_butterfly_step(ACC v, int off) {
    return acc_combine<ACC>(v, shfl_xor(v, off));
}

// ---------------------------------------------------------------------------
// 核心模板：T × ACC × TILE × VEC 全部编译期参数（实例化 3×2×5×2 = 60 个）。
// 归约代码以 warpSize 为准（CUDA=32 / MUSA 按设备内建），硬编码 32 在
// warpSize != 32 的设备上语义错误 —— 同一份 device 代码跨两平台成立
// （第 02 期结论）。
// ---------------------------------------------------------------------------
template <typename T, typename ACC, int TILE, bool VEC>
__global__ void rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ w,
                               T* __restrict__ y, int rows, int h, float eps) {
    constexpr int kW = VecTraits<T>::kWidth;
    const int warp_size = warpSize;  // 设备内建：CUDA 32 / MUSA 实测见台账
    __shared__ float s_x[kMaxCacheElems];  // 行缓存（FP32 形态，所有 dtype 共用）
    __shared__ typename AccTraits<ACC>::type s_warp[TILE / 32];  // 跨 warp 部分和（ACC 形态）

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const T* xrow = x + static_cast<size_t>(row) * h;
        T* yrow = y + static_cast<size_t>(row) * h;

        // ---- Pass 1：读 x（转 FP32），平方和在 ACC 算术内累加 ----
        ACC acc = AccTraits<ACC>::zero();
        if (VEC) {
            int j = threadIdx.x * kW;
            while (j < h) {
                typename VecTraits<T>::Pack p =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(xrow + j);
                #pragma unroll
                for (int k = 0; k < kW; ++k) {
                    const float f = pack_get<T>(p, k);   // T -> FP32
                    if (h <= kMaxCacheElems) s_x[j + k] = f;
                    acc = AccTraits<ACC>::square_add(acc, f);
                }
                j += TILE * kW;
            }
        } else {
            for (int j = threadIdx.x; j < h; j += TILE) {  // 块内 stride（非整 Tile）
                const float f = to_float(xrow[j]);
                if (h <= kMaxCacheElems) s_x[j] = f;
                acc = AccTraits<ACC>::square_add(acc, f);
            }
        }

        // ---- warp 内蝶形归约：步数由 warpSize 推导；ACC 形态一路保持 ----
        for (int off = warp_size / 2; off > 0; off >>= 1) {
            acc = warp_butterfly_step<ACC>(acc, off);
        }
        const int warp_count = (TILE + warp_size - 1) / warp_size;
        const int warp = threadIdx.x / warp_size;
        const int lane = threadIdx.x % warp_size;
        if (lane == 0) s_warp[warp] = acc;
        __syncthreads();

        // ---- 跨 warp 归约：warp 0 在 ACC 算术内汇总并广播 ----
        if (warp == 0) {
            ACC v = (lane < warp_count) ? s_warp[lane] : AccTraits<ACC>::zero();
            for (int off = warp_size / 2; off > 0; off >>= 1) {
                v = warp_butterfly_step<ACC>(v, off);
            }
            if (lane == 0) s_warp[0] = v;
        }
        __syncthreads();
        const float total = AccTraits<ACC>::to_float_total(s_warp[0]);
        // mean + eps + rsqrt 固定 FP32（本期变量只有"平方和累加"这一处；
        // total=inf 时 rsqrtf(inf)=0 -> 输出全零，语义如实记录）。
        const float scale = rsqrtf(total / static_cast<float>(h) + eps);

        // ---- Pass 2：y = x * scale * w（FP32 计算，一次舍入写回 T）----
        if (VEC) {
            int j = threadIdx.x * kW;
            while (j < h) {
                typename VecTraits<T>::Pack wv =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(w + j);
                typename VecTraits<T>::Pack yv;
                #pragma unroll
                for (int k = 0; k < kW; ++k) {
                    const float xv = (h <= kMaxCacheElems) ? s_x[j + k]
                                                           : to_float(xrow[j + k]);
                    pack_set<T>(yv, k, xv * scale * to_float(pack_get<T>(wv, k)));
                }
                *reinterpret_cast<typename VecTraits<T>::Pack*>(yrow + j) = yv;
                j += TILE * kW;
            }
        } else {
            for (int j = threadIdx.x; j < h; j += TILE) {
                const float xv = (h <= kMaxCacheElems) ? s_x[j] : to_float(xrow[j]);
                yrow[j] = cast_from_float<T>(xv * scale * to_float(w[j]));
            }
        }
        // grid-stride 下先保证所有 warp 读完 shared，再允许下一行覆盖。
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// host 端 dispatch：按 (T, ACC, tile, 可否向量化) 选择编译期特化。
// 编译期实例化集合：TILE ∈ {64,128,256,512,1024} × VEC ∈ {false,true} ×
// dtype ∈ {fp32, fp16, bf16} × acc ∈ {fp32, fp16}。
// ---------------------------------------------------------------------------
inline bool is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

inline int device_warp_size() {
    static int cached = 0;
    if (cached == 0) {
        int device = 0;
        RMSNORM_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop;
        RMSNORM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        cached = prop.warpSize;
    }
    return cached;
}

template <typename T>
bool vec_feasible(const void* x, const void* w, const void* y, int h) {
    if (h % VecTraits<T>::kWidth != 0) return false;
    return is_aligned16(x) && is_aligned16(w) && is_aligned16(y);
}

template <typename T, typename ACC>
void launch_tile_acc(const void* x, const void* w, void* y, int rows, int h, float eps,
                     int tile, cudaStream_t stream) {
    const int warp_size = device_warp_size();
    if (tile < warp_size || tile % warp_size != 0) {
        throw std::runtime_error("TILE=" + std::to_string(tile) +
                                 " 必须是本设备 warpSize(" + std::to_string(warp_size) +
                                 ") 的整数倍（蝶形归约要求整 warp 覆盖）");
    }
    const bool vec = vec_feasible<T>(x, w, y, h);
    const T* xt = static_cast<const T*>(x);
    const T* wt = static_cast<const T*>(w);
    T* yt = static_cast<T*>(y);
    dim3 block(tile);
    dim3 grid(static_cast<unsigned>(rows < kMaxGridBlocks ? rows : kMaxGridBlocks));

    #define RMSNORM_LAUNCH(T_NAME, TILE_V, VEC_V)                                  \
        do {                                                                       \
            rmsnorm_kernel<T_NAME, ACC, TILE_V, VEC_V>                             \
                <<<grid, block, 0, stream>>>(xt, wt, yt, rows, h, eps);            \
        } while (0)
    #define RMSNORM_DISPATCH_VEC(T_NAME, TILE_V)                                   \
        do {                                                                       \
            if (vec) { RMSNORM_LAUNCH(T_NAME, TILE_V, true); }                     \
            else     { RMSNORM_LAUNCH(T_NAME, TILE_V, false); }                    \
        } while (0)
    switch (tile) {
        case 64:   RMSNORM_DISPATCH_VEC(T, 64);   break;
        case 128:  RMSNORM_DISPATCH_VEC(T, 128);  break;
        case 256:  RMSNORM_DISPATCH_VEC(T, 256);  break;
        case 512:  RMSNORM_DISPATCH_VEC(T, 512);  break;
        case 1024: RMSNORM_DISPATCH_VEC(T, 1024); break;
        default:
            throw std::runtime_error("unsupported TILE=" + std::to_string(tile) +
                                     "（编译期实例化集合：{64,128,256,512,1024}）");
    }
    #undef RMSNORM_LAUNCH
    #undef RMSNORM_DISPATCH_VEC
}

template <typename ACC>
void launch_acc(const void* x, const void* w, void* y, int dtype_code, int rows, int h,
                float eps, int tile, cudaStream_t stream) {
    switch (dtype_code) {
        case 0:
            launch_tile_acc<float, ACC>(x, w, y, rows, h, eps, tile, stream);
            break;
        case 1:
            launch_tile_acc<__half, ACC>(x, w, y, rows, h, eps, tile, stream);
            break;
        case 2:
            launch_tile_acc<__nv_bfloat16, ACC>(x, w, y, rows, h, eps, tile, stream);
            break;
        default:
            throw std::runtime_error("dtype_code 只支持 0=fp32 / 1=fp16 / 2=bf16");
    }
}

// 统一入口（torch 扩展与 bench 共用）：
//   dtype_code: 0=fp32, 1=fp16, 2=bf16；acc_code: 0=fp32（合同默认）, 1=fp16。
void launch_rmsnorm(const void* x, const void* w, void* y, int dtype_code, int acc_code,
                    int rows, int h, float eps, int tile, cudaStream_t stream) {
    switch (acc_code) {
        case 0:
            launch_acc<float>(x, w, y, dtype_code, rows, h, eps, tile, stream);
            break;
        case 1:
            launch_acc<__half>(x, w, y, dtype_code, rows, h, eps, tile, stream);
            break;
        default:
            throw std::runtime_error("acc_code 只支持 0=fp32 / 1=fp16");
    }
    RMSNORM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace mini_infra
