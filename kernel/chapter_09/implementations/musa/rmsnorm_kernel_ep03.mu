// ============================================================================
// rmsnorm_kernel_ep03.mu —— 第 02 期 RMSNorm Kernel MUSA 版原样复用
//                          （第 03 期分离路径第 2 步；CUDA → MUSA 逐文件迁移）
//
// 复用声明（第 03 期，源 = 本目录 cuda/rmsnorm_kernel_ep03.cu 的机械替换迁移版）：
//   device 代码（两遍扫描 / warpSize-generic 蝶形归约 / 向量化路径）逐字不变
//   （warpSize 内建由 MUSA 侧 = 128，蝶形步数自动 7 步 —— M-11 已实测）；
//   迁移差异全部为机械替换：cuda*→musa*（Runtime API 与头文件）、
//   <cuda_runtime.h>→<musa_runtime.h>、<cuda_fp16.h>→<musa_fp16.h>；
//   host 侧仅两处适配（与 CUDA 版相同，M-17 ctypes 直传模式要求 kernel 库
//   零 Runtime API 调用）：
//     (a) device_warp_size()（musaGetDevice/musaGetDeviceProperties）删除，
//         warpSize 由调用方以参数传入（Python 侧用 probe kernel 实测取得）；
//     (b) launch_rmsnorm_* 尾部的 musaGetLastError 检查删除（错误在 Python
//         侧同步边界暴露）。
//   其余与 CUDA 版逐字相同 —— 差异逐行记入 cross-backend-comparison.md。
//
// 组织方式（对应大纲 §13 第 02 期"实现"条目）：
//   - 每 block 负责一行（grid-stride 循环遍历行，支持任意 rows）；
//   - warp 内归约用 __shfl_xor_sync（蝶形，5 步）；
//   - 跨 warp 归约用 shared memory（各 warp 部分和 -> warp 0 汇总 -> 广播）；
//   - 平方和与均值固定 FP32 累加（合同：累加 dtype = FP32）；
//   - 指针 16B 对齐且 H 整除向量化宽度时走向量化路径：
//       FP32: float4（每线程每步 4 元素），FP16: uint4 装 8 个 half；
//     否则回退标量路径。块内 stride 循环覆盖 H=127~2048 的非整 Tile；
//   - 第一遍扫描把 x 行缓存进 shared memory（H<=2048 时），
//     第二遍从 shared 读 x、写 y —— 理论访存 = 读 x + 写 y + 读 w，
//     避免 naive 两遍全局重读 x 的 3x 流量（这正是"手写单 Kernel"相对
//     组合表达式链的动机，第 01 期结论）。
//
// Tile（block 维度）是编译期参数：模板实例化 TILE ∈ {64,128,256,512,1024}，
// host 端 dispatch 表按运行时参数选择 —— 扫描无需重编译，但每个配置都是
// 编译期特化（合同：编译期 specialization 条目）。
//
// 数值语义：
//   acc 用 FP32 串入 warp 树形归约；scale = rsqrtf(mean_square + eps)
//   （rsqrtf 为硬件近似，~2^-22 相对误差 —— 对比 CPU 真值时可见，
//   与 FP16 dtype 误差（~1e-2）分开归因）；
//   y = T(x * scale * w)：归一化与权重乘法在 FP32 完成，写回时一次舍入
//   到 T（比组合表达式先舍入到 T 再乘 w 少一次中间舍入，见对照实验）。
//
// 确定性：固定 Tile 与固定形状下归约分组固定 => 位级可复现（第 07 期主题）。
//
// 本文件刻意不依赖 torch：可被 mcc 单独编译（fused_bench.mu 使用），
// 第 04 期 torch_musa 扩展再包一层（本期不涉及注册）。
// ============================================================================

#include <musa_runtime.h>
#include <musa_fp16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mini_infra {

// ---------------------------------------------------------------------------
// 错误检查（host 侧）。MUSA 迁移版把 musa* 前缀逐个替换，差异记迁移台账。
// ---------------------------------------------------------------------------
#define RMSNORM_MUSA_CHECK(expr)                                       \
    do {                                                               \
        musaError_t _err = (expr);                                     \
        if (_err != musaSuccess) {                                     \
            throw std::runtime_error(std::string("MUSA error at ") +   \
                                     __FILE__ + ":" +                  \
                                     std::to_string(__LINE__) + " " +  \
                                     #expr + ": " +                    \
                                     musaGetErrorString(_err));        \
        }                                                              \
    } while (0)

// 合同：支持的最大 H（超出则 Pass 2 回退重读 global x，仍正确）。
constexpr int kMaxCacheElems = 2048;
// grid-stride 一次发射的最大 block 数（行数更多时循环接管）。
constexpr long kMaxGridBlocks = 1L << 20;

// ---------------------------------------------------------------------------
// 向量化访存适配层：FP32 用 float4（4 元素/16B），FP16 用 uint4（8 half/16B）。
// ---------------------------------------------------------------------------
template <typename T>
struct VecTraits;

template <>
struct VecTraits<float> {
    static constexpr int kWidth = 4;   // 每次向量化访存的元素数
    using Pack = float4;               // 16B 载荷
};

template <>
struct VecTraits<__half> {
    static constexpr int kWidth = 8;
    using Pack = uint4;                // 8 个 half 的 16B 载荷
};

// FP32 载荷 <-> 标量。
__device__ __forceinline__ float pack_get(const float4& p, int k) {
    return (&p.x)[k];
}

__device__ __forceinline__ void pack_set(float4& p, int k, float v) {
    (&p.x)[k] = v;
}

// FP16 载荷 <-> 标量（reinterpret 型双关：uint4 16B 对齐，8 half 连续排布；
// 比 union 双关对编译器更友好，mcc/nvcc 均接受）。
__device__ __forceinline__ float pack_get(const uint4& p, int k) {
    return __half2float(reinterpret_cast<const __half*>(&p)[k]);
}

// 值参数为 __half（MUSA 13 起无 __half→float 隐式转换，调用方经
// cast_from_float<T> 已转换到 T；float 路径走 float4 重载）。
__device__ __forceinline__ void pack_set(uint4& p, int k, __half v) {
    reinterpret_cast<__half*>(&p)[k] = v;
}

// dtype 适配：读入转 FP32 / 写回从 FP32 舍入到 T。
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T cast_from_float(float v);

template <>
__device__ __forceinline__ float cast_from_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ __half cast_from_float<__half>(float v) { return __float2half(v); }

// 核心模板：TILE = block 维度（编译期），VEC = 是否向量化路径（编译期）。
// 归约代码以 warpSize 为准（MUSA=32，MUSA=128，第 01 期实测）：蝶形步数、
// lane/warp 索引、跨 warp 部分和数量全部由 warpSize 推导 —— 硬编码 32 在
// MUSA 上语义错误（同一份 device 代码因此跨两平台成立，见迁移台账）。
template <typename T, int TILE, bool VEC>
__global__ void rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ w,
                               T* __restrict__ y, int rows, int h, float eps) {
    constexpr int kW = VecTraits<T>::kWidth;
    const int warp_size = warpSize;  // 设备内建：MUSA 32 / MUSA 128
    __shared__ float s_x[kMaxCacheElems];  // 行缓存（H<=2048 时 Pass 2 免重读 x）
    __shared__ float s_warp[TILE / 32];    // 跨 warp 归约的部分和（上界：warpSize>=32）

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const T* xrow = x + static_cast<size_t>(row) * h;
        T* yrow = y + static_cast<size_t>(row) * h;

        // ---- Pass 1：读 x（向量化/标量），缓存进 shared，FP32 累加平方和 ----
        float acc = 0.0f;
        if (VEC) {
            // host 端已保证：h % kW == 0 且三个指针 16B 对齐。
            int j = threadIdx.x * kW;
            while (j < h) {
                typename VecTraits<T>::Pack p =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(xrow + j);
                #pragma unroll
                for (int k = 0; k < kW; ++k) {
                    const float f = pack_get(p, k);
                    if (h <= kMaxCacheElems) s_x[j + k] = f;
                    acc += f * f;
                }
                j += TILE * kW;
            }
        } else {
            for (int j = threadIdx.x; j < h; j += TILE) {  // 块内 stride 循环（非整 Tile）
                const float f = to_float(xrow[j]);
                if (h <= kMaxCacheElems) s_x[j] = f;
                acc += f * f;
            }
        }

        // ---- warp 内蝶形归约：步数由 warpSize 推导（MUSA 5 步 / MUSA 7 步）----
        for (int off = warp_size / 2; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        }
        const int warp_count = (TILE + warp_size - 1) / warp_size;
        const int warp = threadIdx.x / warp_size;
        const int lane = threadIdx.x % warp_size;
        if (lane == 0) s_warp[warp] = acc;
        __syncthreads();

        // ---- 跨 warp 归约：warp 0 汇总各 warp 部分和并广播 ----
        if (warp == 0) {
            float v = (lane < warp_count) ? s_warp[lane] : 0.0f;
            for (int off = warp_size / 2; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, off);
            }
            if (lane == 0) s_warp[0] = v;
        }
        __syncthreads();
        const float total = s_warp[0];
        // scale = 1/sqrt(mean(x^2) + eps)，eps 在根号内（全零行输出全零，无 NaN）。
        const float scale = rsqrtf(total / static_cast<float>(h) + eps);

        // ---- Pass 2：y = x * scale * w（x 优先从 shared 读；w 每行重读，驻留 L2）----
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
                    pack_set(yv, k, cast_from_float<T>(xv * scale * to_float(pack_get(wv, k))));
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
        // grid-stride 下本 block 可能继续处理下一行：先保证所有 warp 读完
        // s_x / s_warp，再允许下一行 Pass 1 覆盖它们。
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// host 端 dispatch：按 (T, tile, 可否向量化) 选择编译期特化。
// 编译期实例化集合：TILE ∈ {64,128,256,512,1024} × VEC ∈ {false,true} ×
// dtype ∈ {float32, float16}。
// ---------------------------------------------------------------------------
inline bool is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

// 第 03 期适配 (a)：原 device_warp_size()（musaGetDevice/musaGetDeviceProperties）
// 删除 —— ctypes 直传模式要求 kernel 库零 Runtime API 调用（M-17）。
// warpSize 改由调用方以参数传入（Python 侧 probe kernel 实测：
// MUSA 32 / MUSA 128，第 01/02 期实测结论一致）。
// 蝶形归约要求 TILE 是 warpSize 整数倍 —— MUSA warpSize=128，TILE=64
// 显式拒绝而非静默错误。

template <typename T>
bool vec_feasible(const void* x, const void* w, const void* y, int h) {
    if (h % VecTraits<T>::kWidth != 0) return false;
    return is_aligned16(x) && is_aligned16(w) && is_aligned16(y);
}

template <typename T>
void launch_tile(const void* x, const void* w, void* y, int rows, int h, float eps,
                 int tile, musaStream_t stream, int warp_size) {
    if (tile < warp_size || tile % warp_size != 0) {
        throw std::runtime_error("TILE=" + std::to_string(tile) +
                                 " 必须是本设备 warpSize(" + std::to_string(warp_size) +
                                 ") 的整数倍（蝶形归约要求整 warp 覆盖；MUSA warpSize=128）");
    }
    const bool vec = vec_feasible<T>(x, w, y, h);
    const T* xt = static_cast<const T*>(x);
    const T* wt = static_cast<const T*>(w);
    T* yt = static_cast<T*>(y);
    dim3 block(tile);
    dim3 grid(static_cast<unsigned>(rows < kMaxGridBlocks ? rows : kMaxGridBlocks));

    #define RMSNORM_LAUNCH(T_NAME, TILE_V, VEC_V)                                   \
        do {                                                                        \
            rmsnorm_kernel<T_NAME, TILE_V, VEC_V>                                   \
                <<<grid, block, 0, stream>>>(xt, wt, yt, rows, h, eps);             \
        } while (0)
    #define RMSNORM_DISPATCH_VEC(T_NAME, TILE_V)                                    \
        do {                                                                        \
            if (vec) { RMSNORM_LAUNCH(T_NAME, TILE_V, true); }                      \
            else     { RMSNORM_LAUNCH(T_NAME, TILE_V, false); }                     \
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

// 第 03 期适配 (b)：尾部 RMSNORM_MUSA_CHECK(musaGetLastError()) 删除
// （ctypes 模式零 Runtime API；launch 错误在 Python 侧同步边界暴露）。

void launch_rmsnorm_fp32(const float* x, const float* w, float* y, int rows, int h,
                         float eps, int tile, musaStream_t stream, int warp_size) {
    launch_tile<float>(x, w, y, rows, h, eps, tile, stream, warp_size);
}

void launch_rmsnorm_fp16(const void* x, const void* w, void* y, int rows, int h,
                         float eps, int tile, musaStream_t stream, int warp_size) {
    launch_tile<__half>(x, w, y, rows, h, eps, tile, stream, warp_size);
}


// dtype 分发辅助（第 03 期 extern "C" 启动器用；指针按字节传入）。
void launch_rmsnorm_any(const void* x, const void* w, void* y, bool is_fp16,
                        int rows, int h, float eps, int tile, musaStream_t stream,
                        int warp_size) {
    if (is_fp16) {
        launch_rmsnorm_fp16(x, w, y, rows, h, eps, tile, stream, warp_size);
    } else {
        launch_rmsnorm_fp32(static_cast<const float*>(x), static_cast<const float*>(w),
                            static_cast<float*>(y), rows, h, eps, tile, stream, warp_size);
    }
}

}  // namespace mini_infra
