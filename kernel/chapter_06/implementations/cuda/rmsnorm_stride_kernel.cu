// ============================================================================
// rmsnorm_stride_kernel.cu —— 第 06 期策略 C：stride 寻址（泛化 index 计算）
//
// 在第 02 期连续 kernel（rmsnorm_kernel.cu，本目录同款复制）之上做的最小
// 泛化：给 x 行加两个元素步长参数
//     x 的元素 (row, j) 物理地址 = x + row*stride_row + j*stride_col
// 于是同一份 device 代码可以吃：
//   - 行主序连续（stride_row=H, stride_col=1）—— 与 ep03 kernel 完全等价；
//   - 转置视图（stride_row=1, stride_col=rows）—— 本期实验的"转置后输入"；
//   - 行切片视图（stride_row=H+pad, stride_col=1）—— 大模型 slice 的布局。
//
// 合并访存教学点（本期实测主题）：
//   - stride_col == 1 时行内相邻线程访存相邻 => 向量化路径仍可能成立
//     （host 端额外要求行首 16B 对齐：x + row*stride_row 按 16B 对齐）；
//   - stride_col == rows（转置视图）时同一 warp 的 32 个线程读的地址相距
//     rows*itemsize 字节 => 一次 32B sector 只贡献 1 个元素，带宽塌缩到
//     ~1/8（FP32）—— 这就是"kernel 支持非连续的代价"，无法用任何软件
//     技巧补救，只能靠改变布局（策略 B 的拷贝）。
// w 恒连续、y 恒行主序连续（合同：输出布局固定，见 experiment-contract.yaml）。
//
// 归约结构、warpSize 泛化（device builtin，CUDA 32 / MUSA 32-lane 分组）、
// TILE 模板 {64,128,256,512,1024}、FP32 累加 —— 全部继承第 02 期 kernel。
// ============================================================================

#include "rmsnorm_kernel.cu"  // 第 02 期连续 kernel + VecTraits/归约基础设施

namespace mini_infra {

template <typename T, int TILE, bool VEC>
__global__ void rmsnorm_kernel_strided(const T* __restrict__ x, const T* __restrict__ w,
                                       T* __restrict__ y, int rows, int h,
                                       long stride_row, long stride_col, float eps) {
    constexpr int kW = VecTraits<T>::kWidth;
    const int warp_size = warpSize;
    __shared__ float s_x[kMaxCacheElems];
    __shared__ float s_warp[TILE / 32];

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const T* xrow = x + static_cast<long>(row) * stride_row;
        T* yrow = y + static_cast<size_t>(row) * h;

        // ---- Pass 1：读 x（物理地址 = xrow + j*stride_col），FP32 累加平方和 ----
        float acc = 0.0f;
        if (VEC) {
            // host 端已保证：stride_col == 1、h % kW == 0、行首 16B 对齐。
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
            for (int j = threadIdx.x; j < h; j += TILE) {
                const float f = to_float(xrow[static_cast<long>(j) * stride_col]);
                if (h <= kMaxCacheElems) s_x[j] = f;
                acc += f * f;
            }
        }

        // ---- 归约（与第 02 期逐行相同：warpSize 泛化蝶形 + 跨 warp 汇总）----
        for (int off = warp_size / 2; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        }
        const int warp_count = (TILE + warp_size - 1) / warp_size;
        const int warp = threadIdx.x / warp_size;
        const int lane = threadIdx.x % warp_size;
        if (lane == 0) s_warp[warp] = acc;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < warp_count) ? s_warp[lane] : 0.0f;
            for (int off = warp_size / 2; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, off);
            }
            if (lane == 0) s_warp[0] = v;
        }
        __syncthreads();
        const float total = s_warp[0];
        const float scale = rsqrtf(total / static_cast<float>(h) + eps);

        // ---- Pass 2：y = x * scale * w（y 行主序连续；x 优先 shared 缓存）----
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
                const float xv = (h <= kMaxCacheElems)
                                     ? s_x[j]
                                     : to_float(xrow[static_cast<long>(j) * stride_col]);
                yrow[j] = cast_from_float<T>(xv * scale * to_float(w[j]));
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// host 端 dispatch（策略 C）：stride 由调用方（ext 层从 tensor.strides()）传入。
// 向量化可行性 = stride_col==1 且 h%kW==0 且各指针 16B 对齐且行首 16B 对齐
//（行首对齐：x + row*stride_row 对每行都要成立 => stride_row*itemsize % 16 == 0）。
// ---------------------------------------------------------------------------
template <typename T>
void launch_strided_tile(const void* x, const void* w, void* y, int rows, int h,
                         long stride_row, long stride_col, float eps,
                         int tile, cudaStream_t stream) {
    const int warp_size = device_warp_size();
    if (tile < warp_size || tile % warp_size != 0) {
        throw std::runtime_error("TILE=" + std::to_string(tile) +
                                 " 必须是本设备 warpSize(" + std::to_string(warp_size) +
                                 ") 的整数倍（蝶形归约要求整 warp 覆盖）");
    }
    constexpr int kW = VecTraits<T>::kWidth;
    const size_t itemsize = sizeof(T);
    const bool row_start_aligned =
        (stride_row * itemsize) % 16 == 0;  // 每行首地址 16B 对齐（float4/uint4 前提）
    const bool vec = stride_col == 1 && h % kW == 0 && row_start_aligned &&
                     is_aligned16(x) && is_aligned16(w) && is_aligned16(y);
    const T* xt = static_cast<const T*>(x);
    const T* wt = static_cast<const T*>(w);
    T* yt = static_cast<T*>(y);
    dim3 block(tile);
    dim3 grid(static_cast<unsigned>(rows < kMaxGridBlocks ? rows : kMaxGridBlocks));

    #define RMSNORM_STRIDED_LAUNCH(T_NAME, TILE_V, VEC_V)                               \
        do {                                                                            \
            rmsnorm_kernel_strided<T_NAME, TILE_V, VEC_V>                               \
                <<<grid, block, 0, stream>>>(xt, wt, yt, rows, h, stride_row,           \
                                             stride_col, eps);                          \
        } while (0)
    #define RMSNORM_STRIDED_DISPATCH_VEC(T_NAME, TILE_V)                                \
        do {                                                                            \
            if (vec) { RMSNORM_STRIDED_LAUNCH(T_NAME, TILE_V, true); }                  \
            else     { RMSNORM_STRIDED_LAUNCH(T_NAME, TILE_V, false); }                 \
        } while (0)
    switch (tile) {
        case 64:   RMSNORM_STRIDED_DISPATCH_VEC(T, 64);   break;
        case 128:  RMSNORM_STRIDED_DISPATCH_VEC(T, 128);  break;
        case 256:  RMSNORM_STRIDED_DISPATCH_VEC(T, 256);  break;
        case 512:  RMSNORM_STRIDED_DISPATCH_VEC(T, 512);  break;
        case 1024: RMSNORM_STRIDED_DISPATCH_VEC(T, 1024); break;
        default:
            throw std::runtime_error("unsupported TILE=" + std::to_string(tile));
    }
    #undef RMSNORM_STRIDED_LAUNCH
    #undef RMSNORM_STRIDED_DISPATCH_VEC
}

void launch_rmsnorm_strided_any(const void* x, const void* w, void* y, bool is_fp16,
                                int rows, int h, long stride_row, long stride_col,
                                float eps, int tile, cudaStream_t stream) {
    if (is_fp16) {
        launch_strided_tile<__half>(x, w, y, rows, h, stride_row, stride_col, eps,
                                    tile, stream);
    } else {
        launch_strided_tile<float>(x, w, y, rows, h, stride_row, stride_col, eps,
                                   tile, stream);
    }
    RMSNORM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace mini_infra
