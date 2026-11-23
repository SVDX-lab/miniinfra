// ============================================================================
// rmsnorm_stride_kernel.cu —— 第 06 期 MUSA 策略 C：stride 寻址（泛化 index）
//
// CUDA 版（implementations/cuda/rmsnorm_stride_kernel.cu）的逐文件迁移：
//   - device 代码（kernel 模板）逐字相同——warpSize 由 device builtin 推导
//     （M-20：S4000 shuffle 按 32 lane 分组），蝶形/跨 warp 归约零改动；
//   - host 侧机械替换：cudaStream_t→musaStream_t、RMSNORM_CUDA_CHECK→
//     RMSNORM_MUSA_CHECK（musaError_t/musaSuccess/musaGetErrorString）、
//     cudaGetLastError→musaGetLastError（第 02 期实机核对结论复用）。
// 教学点与 CUDA 版一致：stride_col==rows 时合并访存失效，带宽塌缩——
// 本期用 S4000 实测量化"同一 kernel、同一布局、两个平台塌缩比例是否一致"。
// ============================================================================

#include "rmsnorm_kernel.cu"  // 第 02/06 期 MUSA 连续 kernel（本目录同款）

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

        // ---- Pass 2：y = x * scale * w（y 行主序连续）----
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
// host 端 dispatch（策略 C，musa 版）：与 CUDA 版逐字对应，仅错误检查宏不同。
// ---------------------------------------------------------------------------
template <typename T>
void launch_strided_tile(const void* x, const void* w, void* y, int rows, int h,
                         long stride_row, long stride_col, float eps,
                         int tile, musaStream_t stream) {
    const int warp_size = device_warp_size();
    if (tile < warp_size || tile % warp_size != 0) {
        throw std::runtime_error("TILE=" + std::to_string(tile) +
                                 " 必须是本设备 warpSize(" + std::to_string(warp_size) +
                                 ") 的整数倍（蝶形归约要求整 warp 覆盖）");
    }
    constexpr int kW = VecTraits<T>::kWidth;
    const size_t itemsize = sizeof(T);
    const bool row_start_aligned = (stride_row * itemsize) % 16 == 0;
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
                                float eps, int tile, musaStream_t stream) {
    if (is_fp16) {
        launch_strided_tile<__half>(x, w, y, rows, h, stride_row, stride_col, eps,
                                    tile, stream);
    } else {
        launch_strided_tile<float>(x, w, y, rows, h, stride_row, stride_col, eps,
                                   tile, stream);
    }
    RMSNORM_MUSA_CHECK(musaGetLastError());
}

}  // namespace mini_infra
