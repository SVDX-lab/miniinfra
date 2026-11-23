// ============================================================================
// paged_ext.cu —— 第 21 期 torch 扩展封装（CUDA 列，ch21 风格）。
//
// 暴露入口（合同校验显式报错，不静默回退）：
//   paged_decode_attention(q, k_pool, v_pool, block_table, seq_lens,
//                          block_size, scale)     → [B, HQ, D]
//   contig_decode_attention(q, k_cache, v_cache, seq_lens, scale)
//                                                 → [B, HQ, D]（控制组）
//   paged_prefill_attention(q, k_pool, v_pool, block_table, block_size, scale)
//                                                 → [B, HQ, S, D]（可选对照）
//   contig_prefill_attention(q, k, v, scale)      → [B, HQ, S, D]（控制组）
//   gather_kv(k_pool, v_pool, block_table, seq_lens, block_size)
//                                                 → (k_out, v_out)（归因分层）
//   validate_tables(block_table, seq_lens, num_blocks, block_size)
//                                                 → 块表合同预检（host 侧）
//   warp_size_device()
//
// 契约：decode 的 per-request 长度从 seq_lens[b]（device int32）在核内读取
// ——引擎口径（变长 batch 一条 kernel 路径）。表内容校验是 host 预检入口
// （validate_tables，correctness 与 bench 前各调一次）；性能路径内不做逐项
// 检查（kernel 无逐元素 guard 的性能口径声明，见 experiment-contract）。
// ============================================================================

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "paged_kernels.cuh"

namespace {

int g_device_warp_size = 0;   // 探针缓存（进程级，单 GPU 会话）

void ensure_warp_probed() {
    if (g_device_warp_size > 0) return;
    auto buf = at::empty({1}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
    pkg::warp_size_probe<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        buf.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    g_device_warp_size = buf.cpu().item<int>();
}

void check_common_qkv(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                      bool decode) {
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "合同：q/k(/v) dtype 必须一致（float16 或 float32），实际 ",
                q.scalar_type(), "/", k.scalar_type(), "/", v.scalar_type());
    TORCH_CHECK((q.scalar_type() == at::kHalf) || (q.scalar_type() == at::kFloat),
                "合同：本期只接受 float16/float32，实际 ", q.scalar_type());
    TORCH_CHECK(q.device().is_cuda() && k.device().is_cuda() && v.device().is_cuda(),
                "本扩展是 CUDA 实现，不接受 ", q.device(), " Tensor");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "合同：输入必须连续（非连续布局是第 06 期主题）");
    TORCH_CHECK(q.dim() == (decode ? 3 : 4), decode
                ? "decode 合同：q [B, HQ, D]" : "prefill 合同：q [B, HQ, S, D]",
                "，实际 ", q.sizes());
    TORCH_CHECK(q.size(-1) == pkg::D, "head_dim 合同 = ", pkg::D, "，实际 ", q.size(-1));
    TORCH_CHECK(k.size(1) == v.size(1), "k/v 的 kv 头数不一致");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "HQ 必须 HKV 整除（GQA），实际 ",
                q.size(1), " vs ", k.size(1));
}

void check_pool_tables(const at::Tensor& k_pool, const at::Tensor& v_pool,
                       const at::Tensor& block_table, const at::Tensor& seq_lens,
                       const at::Tensor& q, int64_t block_size) {
    TORCH_CHECK(k_pool.dim() == 4 && v_pool.dim() == 4,
                "池合同：[num_blocks, HKV, block_size, D]，实际 ",
                k_pool.sizes(), "/", v_pool.sizes());
    TORCH_CHECK(k_pool.sizes() == v_pool.sizes(), "k_pool/v_pool 形状不一致");
    TORCH_CHECK(k_pool.size(3) == pkg::D, "池 head_dim 合同 = ", pkg::D);
    TORCH_CHECK(k_pool.size(2) == block_size,
                "block_size 与池形状不一致：池 ", k_pool.size(2), " vs 参数 ", block_size);
    TORCH_CHECK(block_size >= 1 && block_size <= 4096,
                "block_size 合同域 [1, 4096]，实际 ", block_size);
    TORCH_CHECK(block_table.scalar_type() == at::kInt && seq_lens.scalar_type() == at::kInt,
                "block_table/seq_lens 合同 dtype = int32，实际 ",
                block_table.scalar_type(), "/", seq_lens.scalar_type());
    TORCH_CHECK(block_table.is_cuda() && block_table.is_contiguous(),
                "block_table 必须是连续 CUDA int32");
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.is_contiguous(),
                "seq_lens 必须是连续 CUDA int32");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == q.size(0),
                "block_table 合同 [B, max_blocks]，B 与 q 一致，实际 ", block_table.sizes());
    TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == q.size(0),
                "seq_lens 合同 [B]，实际 ", seq_lens.sizes());
    TORCH_CHECK(q.size(1) % k_pool.size(1) == 0,
                "HQ 必须 HKV 整除（GQA），实际 ", q.size(1), " vs ", k_pool.size(1));
}

template <typename T, int WARP>
void launch_paged_decode(const at::Tensor& q, const at::Tensor& kp,
                         const at::Tensor& vp, const at::Tensor& table,
                         const at::Tensor& lens, at::Tensor* o,
                         int hq, int hkv, int bs, int max_blocks,
                         float scale, cudaStream_t stream) {
    // data_ptr()（void*）：T 由调用方按 dtype 分派保证一致——data_ptr<at::Half>()
    // 的类型检查会让 fp32 路径误抛 "expected scalar type Half"（KI-22-09）
    pkg::paged_decode_kernel<T, WARP><<<static_cast<unsigned>(q.size(0) * hq),
                                        pkg::THREADS, 0, stream>>>(
        reinterpret_cast<const T*>(q.data_ptr()),
        reinterpret_cast<const T*>(kp.data_ptr()),
        reinterpret_cast<const T*>(vp.data_ptr()),
        table.data_ptr<int>(), lens.data_ptr<int>(),
        reinterpret_cast<T*>(o->data_ptr()),
        hq, hkv, bs, max_blocks, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int WARP>
void launch_contig_decode(const at::Tensor& q, const at::Tensor& k,
                          const at::Tensor& v, const at::Tensor& lens,
                          at::Tensor* o, int hq, int hkv, int n_max,
                          float scale, cudaStream_t stream) {
    pkg::contig_decode_kernel<T, WARP><<<static_cast<unsigned>(q.size(0) * hq),
                                         pkg::THREADS, 0, stream>>>(
        reinterpret_cast<const T*>(q.data_ptr()),
        reinterpret_cast<const T*>(k.data_ptr()),
        reinterpret_cast<const T*>(v.data_ptr()),
        lens.data_ptr<int>(),
        reinterpret_cast<T*>(o->data_ptr()),
        hq, hkv, n_max, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

at::Tensor paged_decode_attention(const at::Tensor& q, const at::Tensor& k_pool,
                                  const at::Tensor& v_pool,
                                  const at::Tensor& block_table,
                                  const at::Tensor& seq_lens,
                                  int64_t block_size, double scale) {
    check_common_qkv(q, k_pool, v_pool, /*decode=*/true);
    check_pool_tables(k_pool, v_pool, block_table, seq_lens, q, block_size);
    at::Tensor o = at::empty_like(q);
    ensure_warp_probed();
    const at::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int hq = static_cast<int>(q.size(1));
    const int hkv = static_cast<int>(k_pool.size(1));
    const int bs = static_cast<int>(block_size);
    const int max_blocks = static_cast<int>(block_table.size(1));
    const float fscale = static_cast<float>(scale);
    if (q.scalar_type() == at::kHalf) {
        if (g_device_warp_size == 128)
            launch_paged_decode<ATTN_HALF_TYPE, 128>(q, k_pool, v_pool, block_table,
                                                     seq_lens, &o, hq, hkv, bs,
                                                     max_blocks, fscale, stream);
        else
            launch_paged_decode<ATTN_HALF_TYPE, 32>(q, k_pool, v_pool, block_table,
                                                    seq_lens, &o, hq, hkv, bs,
                                                    max_blocks, fscale, stream);
    } else {
        if (g_device_warp_size == 128)
            launch_paged_decode<float, 128>(q, k_pool, v_pool, block_table, seq_lens,
                                            &o, hq, hkv, bs, max_blocks, fscale, stream);
        else
            launch_paged_decode<float, 32>(q, k_pool, v_pool, block_table, seq_lens,
                                           &o, hq, hkv, bs, max_blocks, fscale, stream);
    }
    return o;
}

at::Tensor contig_decode_attention(const at::Tensor& q, const at::Tensor& k_cache,
                                   const at::Tensor& v_cache,
                                   const at::Tensor& seq_lens, double scale) {
    check_common_qkv(q, k_cache, v_cache, /*decode=*/true);
    TORCH_CHECK(k_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
                "连续 cache 合同 [B, HKV, Nmax, D]，实际 ", k_cache.sizes());
    TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_cuda()
                && seq_lens.is_contiguous() && seq_lens.size(0) == q.size(0),
                "seq_lens 合同 [B] 连续 CUDA int32");
    at::Tensor o = at::empty_like(q);
    ensure_warp_probed();
    const at::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int hq = static_cast<int>(q.size(1));
    const int hkv = static_cast<int>(k_cache.size(1));
    const int n_max = static_cast<int>(k_cache.size(2));
    const float fscale = static_cast<float>(scale);
    if (q.scalar_type() == at::kHalf) {
        if (g_device_warp_size == 128)
            launch_contig_decode<ATTN_HALF_TYPE, 128>(q, k_cache, v_cache, seq_lens,
                                                      &o, hq, hkv, n_max, fscale, stream);
        else
            launch_contig_decode<ATTN_HALF_TYPE, 32>(q, k_cache, v_cache, seq_lens,
                                                     &o, hq, hkv, n_max, fscale, stream);
    } else {
        if (g_device_warp_size == 128)
            launch_contig_decode<float, 128>(q, k_cache, v_cache, seq_lens,
                                             &o, hq, hkv, n_max, fscale, stream);
        else
            launch_contig_decode<float, 32>(q, k_cache, v_cache, seq_lens,
                                            &o, hq, hkv, n_max, fscale, stream);
    }
    return o;
}

at::Tensor paged_prefill_attention(const at::Tensor& q, const at::Tensor& k_pool,
                                   const at::Tensor& v_pool,
                                   const at::Tensor& block_table,
                                   int64_t block_size, double scale) {
    check_common_qkv(q, k_pool, v_pool, /*decode=*/false);
    TORCH_CHECK(k_pool.dim() == 4 && k_pool.sizes() == v_pool.sizes(),
                "池合同：[num_blocks, HKV, block_size, D]，实际 ", k_pool.sizes());
    TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.is_cuda()
                && block_table.is_contiguous() && block_table.dim() == 2
                && block_table.size(0) == q.size(0),
                "block_table 合同 [B, max_blocks] 连续 CUDA int32");
    TORCH_CHECK(k_pool.size(2) == block_size, "block_size 与池形状不一致");
    const int b = static_cast<int>(q.size(0));
    const int hq = static_cast<int>(q.size(1));
    const int hkv = static_cast<int>(k_pool.size(1));
    const int s = static_cast<int>(q.size(2));
    const int bs = static_cast<int>(block_size);
    const int max_blocks = static_cast<int>(block_table.size(1));
    const float fscale = static_cast<float>(scale);
    at::Tensor o = at::empty_like(q);
    ensure_warp_probed();
    const at::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const unsigned tiles = static_cast<unsigned>((s + pkg::BR - 1) / pkg::BR);
    dim3 grid(tiles, static_cast<unsigned>(b * hq));
    if (q.scalar_type() == at::kHalf) {
        pkg::paged_prefill_kernel<ATTN_HALF_TYPE><<<grid, pkg::THREADS, 0, stream>>>(
            reinterpret_cast<const ATTN_HALF_TYPE*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const ATTN_HALF_TYPE*>(k_pool.data_ptr<at::Half>()),
            reinterpret_cast<const ATTN_HALF_TYPE*>(v_pool.data_ptr<at::Half>()),
            block_table.data_ptr<int>(),
            reinterpret_cast<ATTN_HALF_TYPE*>(o.data_ptr<at::Half>()),
            hq, hkv, s, bs, max_blocks, fscale);
    } else {
        pkg::paged_prefill_kernel<float><<<grid, pkg::THREADS, 0, stream>>>(
            q.data_ptr<float>(), k_pool.data_ptr<float>(), v_pool.data_ptr<float>(),
            block_table.data_ptr<int>(), o.data_ptr<float>(),
            hq, hkv, s, bs, max_blocks, fscale);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}

at::Tensor contig_prefill_attention(const at::Tensor& q, const at::Tensor& k,
                                    const at::Tensor& v, double scale) {
    check_common_qkv(q, k, v, /*decode=*/false);
    TORCH_CHECK(k.size(2) == q.size(2) && v.size(2) == q.size(2),
                "prefill 合同：K/V 序列长 = Q 序列长");
    const int b = static_cast<int>(q.size(0));
    const int hq = static_cast<int>(q.size(1));
    const int hkv = static_cast<int>(k.size(1));
    const int s = static_cast<int>(q.size(2));
    const float fscale = static_cast<float>(scale);
    at::Tensor o = at::empty_like(q);
    ensure_warp_probed();
    const at::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const unsigned tiles = static_cast<unsigned>((s + pkg::BR - 1) / pkg::BR);
    dim3 grid(tiles, static_cast<unsigned>(b * hq));
    if (q.scalar_type() == at::kHalf) {
        pkg::contig_prefill_kernel<ATTN_HALF_TYPE><<<grid, pkg::THREADS, 0, stream>>>(
            reinterpret_cast<const ATTN_HALF_TYPE*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const ATTN_HALF_TYPE*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const ATTN_HALF_TYPE*>(v.data_ptr<at::Half>()),
            reinterpret_cast<ATTN_HALF_TYPE*>(o.data_ptr<at::Half>()),
            hq, hkv, s, fscale);
    } else {
        pkg::contig_prefill_kernel<float><<<grid, pkg::THREADS, 0, stream>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            o.data_ptr<float>(), hq, hkv, s, fscale);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}

std::vector<at::Tensor> gather_kv(const at::Tensor& k_pool, const at::Tensor& v_pool,
                                  const at::Tensor& block_table,
                                  const at::Tensor& seq_lens, int64_t block_size,
                                  int64_t n_max_hint) {
    TORCH_CHECK(k_pool.dim() == 4 && k_pool.sizes() == v_pool.sizes(),
                "池合同：[num_blocks, HKV, block_size, D]，实际 ", k_pool.sizes());
    TORCH_CHECK(k_pool.scalar_type() == at::kHalf || k_pool.scalar_type() == at::kFloat,
                "池 dtype 合同 float16/float32");
    TORCH_CHECK(k_pool.is_cuda() && k_pool.is_contiguous(), "池必须连续 CUDA");
    TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.is_cuda()
                && block_table.is_contiguous(), "block_table 合同连续 CUDA int32");
    TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_cuda()
                && seq_lens.is_contiguous(), "seq_lens 合同连续 CUDA int32");
    const int b = static_cast<int>(block_table.size(0));
    const int max_blocks = static_cast<int>(block_table.size(1));
    const int hkv = static_cast<int>(k_pool.size(1));
    const int bs = static_cast<int>(block_size);
    int n_max;
    if (n_max_hint > 0) {
        n_max = static_cast<int>(n_max_hint);   // bench 路径：host 已知，免同步
    } else {
        auto cput = seq_lens.cpu();             // correctness 路径：host 预检同步可接受
        n_max = 0;
        const int* lp = cput.data_ptr<int>();
        for (int i = 0; i < b; ++i) n_max = n_max > lp[i] ? n_max : lp[i];
    }
    TORCH_CHECK(n_max >= 1, "Nmax ≥ 1 合同（seq_lens 全 0）");
    at::Tensor k_out = at::empty({b, hkv, (long long)n_max, (long long)pkg::D},
                                 k_pool.options());
    at::Tensor v_out = at::empty_like(k_out);
    const long long total = 2LL * b * hkv * n_max * pkg::D;
    const unsigned grid = static_cast<unsigned>((total + pkg::THREADS - 1) / pkg::THREADS);
    const at::cuda::CUDAGuard guard(k_pool.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (k_pool.scalar_type() == at::kHalf) {
        pkg::gather_kv_kernel<ATTN_HALF_TYPE><<<grid, pkg::THREADS, 0, stream>>>(
            reinterpret_cast<const ATTN_HALF_TYPE*>(k_pool.data_ptr<at::Half>()),
            reinterpret_cast<const ATTN_HALF_TYPE*>(v_pool.data_ptr<at::Half>()),
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
            reinterpret_cast<ATTN_HALF_TYPE*>(k_out.data_ptr<at::Half>()),
            reinterpret_cast<ATTN_HALF_TYPE*>(v_out.data_ptr<at::Half>()),
            b, hkv, bs, max_blocks, n_max, total);
    } else {
        pkg::gather_kv_kernel<float><<<grid, pkg::THREADS, 0, stream>>>(
            k_pool.data_ptr<float>(), v_pool.data_ptr<float>(),
            block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
            k_out.data_ptr<float>(), v_out.data_ptr<float>(),
            b, hkv, bs, max_blocks, n_max, total);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {k_out, v_out};
}

int64_t validate_tables(const at::Tensor& block_table, const at::Tensor& seq_lens,
                        int64_t num_blocks, int64_t block_size) {
    // host 预检入口（合同：越界/有效区 -1/物理块重复/padding 非 -1/表长不足）。
    // 性能路径不调用（.cpu() 同步）——correctness 与 bench 前各预检一次。
    TORCH_CHECK(block_table.scalar_type() == at::kInt && seq_lens.scalar_type() == at::kInt,
                "合同 dtype int32");
    auto t = block_table.cpu();
    auto l = seq_lens.cpu();
    const int b = static_cast<int>(t.size(0));
    const int max_blocks = static_cast<int>(t.size(1));
    const int* tp = t.data_ptr<int>();
    const int* lp = l.data_ptr<int>();
    std::vector<char> seen(static_cast<size_t>(num_blocks), 0);
    for (int i = 0; i < b; ++i) {
        const int n = lp[i];
        TORCH_CHECK(n >= 1, "请求 ", i, " seq_len 必须 ≥ 1（N≥1 合同），实际 ", n);
        const int nb = static_cast<int>((n + block_size - 1) / block_size);
        TORCH_CHECK(nb <= max_blocks, "请求 ", i, " 需要 ", nb, " 块 > 块表宽 ", max_blocks);
        for (int tb = 0; tb < max_blocks; ++tb) {
            const int p = tp[static_cast<size_t>(i) * max_blocks + tb];
            if (tb < nb) {
                TORCH_CHECK(p >= 0 && p < num_blocks, "请求 ", i, " 逻辑块 ", tb,
                            " 表项越界: ", p, "（num_blocks=", num_blocks, "）");
                TORCH_CHECK(!seen[static_cast<size_t>(p)], "物理块 ", p,
                            " 跨请求重复（池合同违例）");
                seen[static_cast<size_t>(p)] = 1;
            } else {
                TORCH_CHECK(p == -1, "请求 ", i, " padding 表项合同值 -1，实际 ", p);
            }
        }
    }
    return 0;
}

int64_t warp_size_device() {
    ensure_warp_probed();
    return static_cast<int64_t>(g_device_warp_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("paged_decode_attention", &paged_decode_attention,
               py::arg("q"), py::arg("k_pool"), py::arg("v_pool"),
               py::arg("block_table"), py::arg("seq_lens"), py::arg("block_size"),
               py::arg("scale"),
               "第 21 期 paged decode attention（单 query + 块表寻址，GQA）");
    module.def("contig_decode_attention", &contig_decode_attention,
               py::arg("q"), py::arg("k_cache"), py::arg("v_cache"),
               py::arg("seq_lens"), py::arg("scale"),
               "控制组：同核去块表化（连续布局寻址）");
    module.def("paged_prefill_attention", &paged_prefill_attention,
               py::arg("q"), py::arg("k_pool"), py::arg("v_pool"),
               py::arg("block_table"), py::arg("block_size"), py::arg("scale"),
               "paged prefill（tile + online softmax + 块表 K/V，可选对照）");
    module.def("contig_prefill_attention", &contig_prefill_attention,
               py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
               "控制组：ch21 prefill 核原样（连续布局）");
    module.def("gather_kv", &gather_kv,
               py::arg("k_pool"), py::arg("v_pool"), py::arg("block_table"),
               py::arg("seq_lens"), py::arg("block_size"),
               py::arg("n_max_hint") = 0,
               "纯 gather：池 → 连续布局（归因分层用；n_max_hint>0 免 host 同步）");
    module.def("validate_tables", &validate_tables,
               py::arg("block_table"), py::arg("seq_lens"),
               py::arg("num_blocks"), py::arg("block_size"),
               "块表合同 host 预检（越界/空洞/重复/padding）");
    module.def("warp_size_device", &warp_size_device,
               "设备内置 warpSize 探针（M-20 口径）");
}
