// ============================================================================
// attention_ext.cu —— 第 20 期 torch 扩展封装（CUDA 列）。
//
// 暴露四个入口（合同校验显式报错，不静默回退——第 02 期风格）：
//   gqa_prefill_attention(q, k, v, scale)      → 形态一（tile + online softmax）
//   gqa_decode_attention(q, k, v, scale)       → 形态二（单 query，chunked）
//   gqa_prefill_matmul(q, k, v, scale)         → 物化对照变体（三 kernel）
//   warp_size_device()                         → 设备内置 warpSize 探针
// WARP 归约实例按探针选择（32 / 128 双实例编译，运行期二选一——M-11/M-20）。
// ============================================================================

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attention_kernels.cuh"

namespace {

int g_device_warp_size = 0;   // 探针结果缓存（0 = 未探测；进程级，单 GPU 会话）

void ensure_warp_probed() {
    if (g_device_warp_size > 0) {
        return;
    }
    auto buf = at::empty({1}, at::TensorOptions()
                                 .dtype(at::kInt)
                                 .device(at::kCUDA));
    atn::warp_size_probe<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        buf.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    g_device_warp_size = buf.cpu().item<int>();
}

template <int WARP>
void launch_decode(const at::Tensor& q, const at::Tensor& k,
                   const at::Tensor& v, at::Tensor* o, int hq, int hkv, int n,
                   float scale, cudaStream_t stream) {
    if (q.scalar_type() == at::kHalf) {
        atn::gqa_decode_kernel<__half, WARP><<<static_cast<unsigned>(q.size(0) * hq),
                                               atn::THREADS, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(o->data_ptr<at::Half>()),
            hq, hkv, n, scale);
    } else {
        atn::gqa_decode_kernel<float, WARP><<<static_cast<unsigned>(q.size(0) * hq),
                                              atn::THREADS, 0, stream>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            o->data_ptr<float>(), hq, hkv, n, scale);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_common(const at::Tensor& q, const at::Tensor& k,
                  const at::Tensor& v, bool prefill) {
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "合同：q/k/v dtype 必须一致（float16 或 float32），实际 ",
                q.scalar_type(), "/", k.scalar_type(), "/", v.scalar_type());
    TORCH_CHECK((q.scalar_type() == at::kHalf) || (q.scalar_type() == at::kFloat),
                "合同：本期只接受 float16/float32，实际 ", q.scalar_type());
    TORCH_CHECK(q.device().type() == c10::DeviceType::CUDA &&
                k.device().type() == c10::DeviceType::CUDA &&
                v.device().type() == c10::DeviceType::CUDA,
                "本扩展是 CUDA 实现，不接受 ", q.device(), " Tensor");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "合同：本期 Kernel 只接受连续输入（非连续布局是第 06 期主题）");
    if (prefill) {
        TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                    "prefill 合同：q/k/v 均为 4D [B, H, S, D]，实际 ",
                    q.sizes(), "/", k.sizes(), "/", v.sizes());
        const int64_t b = q.size(0), hq = q.size(1), s = q.size(2), d = k.size(3);
        TORCH_CHECK(d == atn::D, "head_dim 合同 = ", atn::D, "，实际 ", d);
        TORCH_CHECK(k.size(0) == b && v.size(0) == b, "batch 不一致");
        TORCH_CHECK(k.size(1) == v.size(1), "k/v 的 kv 头数不一致");
        TORCH_CHECK(hq % k.size(1) == 0, "HQ 必须 HKV 整除（GQA），实际 ",
                    hq, " vs ", k.size(1));
        TORCH_CHECK(k.size(2) == s && v.size(2) == s,
                    "prefill 合同：K/V 序列长 = Q 序列长（整段 self-attention）");
        TORCH_CHECK(s >= 1, "S ≥ 1 合同");
    } else {
        TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.dim() == 4,
                    "decode 合同：q [B, HQ, D]，k/v [B, HKV, N, D]，实际 ",
                    q.sizes(), "/", k.sizes(), "/", v.sizes());
        const int64_t b = q.size(0), hq = q.size(1), d = q.size(2);
        TORCH_CHECK(d == atn::D, "head_dim 合同 = ", atn::D, "，实际 ", d);
        TORCH_CHECK(k.size(0) == b && v.size(0) == b, "batch 不一致");
        TORCH_CHECK(k.size(1) == v.size(1), "k/v 的 kv 头数不一致");
        TORCH_CHECK(hq % k.size(1) == 0, "HQ 必须 HKV 整除（GQA），实际 ",
                    hq, " vs ", k.size(1));
        TORCH_CHECK(k.size(2) == v.size(2) && k.size(2) >= 1,
                    "N ≥ 1 且 k/v 一致");
    }
}

at::Tensor dispatch_prefill(const at::Tensor& q, const at::Tensor& k,
                            const at::Tensor& v, double scale, bool materialized) {
    check_common(q, k, v, /*prefill=*/true);
    const int b = static_cast<int>(q.size(0));
    const int hq = static_cast<int>(q.size(1));
    const int hkv = static_cast<int>(k.size(1));
    const int s = static_cast<int>(q.size(2));

    at::Tensor o = at::empty_like(q);
    const at::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float fscale = static_cast<float>(scale);
    const unsigned tiles = static_cast<unsigned>((s + atn::BR - 1) / atn::BR);

    if (materialized) {
        auto scores = at::empty({b * hq, s, s}, q.options());
        auto attn = at::empty({b * hq, s, s}, q.options());
        auto lrow = at::empty({b * hq, s}, q.options().dtype(at::kFloat));
        dim3 grid(s, static_cast<unsigned>(b * hq));
        if (q.scalar_type() == at::kHalf) {
            atn::mat_scores_kernel<__half><<<grid, atn::THREADS, 0, stream>>>(
                reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(scores.data_ptr<at::Half>()),
                hq, hkv, s, fscale);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            if (g_device_warp_size == 128) {
                atn::mat_softmax_kernel<__half, 128><<<grid, atn::THREADS, 0, stream>>>(
                    reinterpret_cast<const __half*>(scores.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(attn.data_ptr<at::Half>()),
                    lrow.data_ptr<float>(), hq, s);
            } else {
                atn::mat_softmax_kernel<__half, 32><<<grid, atn::THREADS, 0, stream>>>(
                    reinterpret_cast<const __half*>(scores.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(attn.data_ptr<at::Half>()),
                    lrow.data_ptr<float>(), hq, s);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            atn::mat_pv_kernel<__half><<<grid, atn::THREADS, 0, stream>>>(
                reinterpret_cast<const __half*>(attn.data_ptr<at::Half>()),
                lrow.data_ptr<float>(),
                reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(o.data_ptr<at::Half>()), hq, hkv, s);
        } else {
            atn::mat_scores_kernel<float><<<grid, atn::THREADS, 0, stream>>>(
                q.data_ptr<float>(), k.data_ptr<float>(),
                scores.data_ptr<float>(), hq, hkv, s, fscale);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            if (g_device_warp_size == 128) {
                atn::mat_softmax_kernel<float, 128><<<grid, atn::THREADS, 0, stream>>>(
                    scores.data_ptr<float>(), attn.data_ptr<float>(),
                    lrow.data_ptr<float>(), hq, s);
            } else {
                atn::mat_softmax_kernel<float, 32><<<grid, atn::THREADS, 0, stream>>>(
                    scores.data_ptr<float>(), attn.data_ptr<float>(),
                    lrow.data_ptr<float>(), hq, s);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            atn::mat_pv_kernel<float><<<grid, atn::THREADS, 0, stream>>>(
                attn.data_ptr<float>(), lrow.data_ptr<float>(),
                v.data_ptr<float>(), o.data_ptr<float>(), hq, hkv, s);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return o;
    }

    if (q.scalar_type() == at::kHalf) {
        atn::gqa_prefill_kernel<__half><<<dim3(tiles, static_cast<unsigned>(b * hq)),
                                          atn::THREADS, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(o.data_ptr<at::Half>()),
            hq, hkv, s, fscale);
    } else {
        atn::gqa_prefill_kernel<float><<<dim3(tiles, static_cast<unsigned>(b * hq)),
                                         atn::THREADS, 0, stream>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            o.data_ptr<float>(), hq, hkv, s, fscale);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}

}  // namespace

at::Tensor gqa_prefill_attention(const at::Tensor& q, const at::Tensor& k,
                                 const at::Tensor& v, double scale) {
    ensure_warp_probed();
    return dispatch_prefill(q, k, v, scale, /*materialized=*/false);
}

at::Tensor gqa_prefill_matmul(const at::Tensor& q, const at::Tensor& k,
                              const at::Tensor& v, double scale) {
    ensure_warp_probed();
    return dispatch_prefill(q, k, v, scale, /*materialized=*/true);
}

at::Tensor gqa_decode_attention(const at::Tensor& q, const at::Tensor& k,
                                const at::Tensor& v, double scale) {
    check_common(q, k, v, /*prefill=*/false);
    ensure_warp_probed();
    const int hq = static_cast<int>(q.size(1));
    const int hkv = static_cast<int>(k.size(1));
    const int n = static_cast<int>(k.size(2));
    at::Tensor o = at::empty_like(q);
    const at::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float fscale = static_cast<float>(scale);
    if (g_device_warp_size == 128) {
        launch_decode<128>(q, k, v, &o, hq, hkv, n, fscale, stream);
    } else {
        launch_decode<32>(q, k, v, &o, hq, hkv, n, fscale, stream);
    }
    return o;
}

int64_t warp_size_device() {
    ensure_warp_probed();
    return static_cast<int64_t>(g_device_warp_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("gqa_prefill_attention", &gqa_prefill_attention,
               py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
               "第 20 期 naive prefill attention（tile + online softmax，causal，GQA）");
    module.def("gqa_decode_attention", &gqa_decode_attention,
               py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
               "第 20 期 naive decode attention（单 query，chunked + 稳定 softmax，GQA）");
    module.def("gqa_prefill_matmul", &gqa_prefill_matmul,
               py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
               "物化对照变体（S×S scores/attn 满阵三 kernel）");
    module.def("warp_size_device", &warp_size_device,
               "设备内置 warpSize 探针（M-20 口径）");
}
