// ============================================================================
// mla_absorbed.cu —— 第 24 期 MLA 吸收形式 decode attention 核（CUDA）。
//
// 结构：flash-decoding 式 split-K 两 kernel（大纲"少量大 GEMM + 单个 attention
// 核"的核部分；q 投影 / 吸收 bmm / 输出吸收 bmm / o_proj 走 torch+cuBLAS）。
//
//   输入（全部 fp16、连续、CUDA 设备）：
//     q_abs  [H, 512]   q_nope 已吸收 W_UK 后的 latent 空间 query（torch bmm 产出）
//     q_rope [H, 64]    已按 query 位置旋转的 rope 头（rotation 在 torch 侧完成）
//     c_kv   [N, 512]   latent cache（未吸收形式里要经它物化 K/V；这里直接读）
//     k_rope [N, 64]    已逐位置旋转的 rope cache
//   输出：
//     s      [H, 512]   加权 latent（softmax 归一化后 Σ_t a_t · c_kv[t]，fp16）
//
//   kernel 1 (mla_split_kernel)：grid (H, S)，block 256 线程（8 warp）。
//     每 warp 串行处理 chunk 内 t = t_start + warp, 步长 8 的 token：
//       score_t = (Σ_d q_abs[d]·c_kv[t][d] + Σ_d q_rope[d]·k_rope[t][d]) × scale
//     online softmax（per-warp 运行 max/sum）+ per-warp 加权 latent 累加
//     （acc[8][512] 放 shared，每 lane 独占 16 维，无竞争）。
//     块内合并 8 个 warp 部分和 → 写 psum[H,S,512] / pl[H,S]（未归一化）。
//   kernel 2 (mla_merge_kernel)：grid (H,)，跨 S 合并 + 归一化 → s。
//
//   数值口径：io fp16、片上 fp32 累加（与 torch bmm fp16 路径一致）；
//   expf（非 fast-math 近似）。确定性：无 atomics、归约顺序固定 → 位级可复现
//   （compare_correctness 有 3 次连跑位级比对）。
//
//   合同维度：D_LAT=512 / D_ROPE=64 编译期固定（DeepSeek-V2-Lite 公开 config，
//   见 mla_config.py）；H 与 N 运行期任意（H 进 grid.x，N 进循环边界）。
// ============================================================================

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int D_LAT = 512;             // kv_lora_rank
constexpr int D_ROPE = 64;             // qk_rope_head_dim
constexpr int THREADS = 256;           // 8 warps
constexpr int WARPS = 8;
constexpr unsigned int FULL_MASK = 0xffffffffu;
constexpr int DIMS_PER_LANE = D_LAT / 32;   // 16：每 lane 独占的 latent 维
constexpr int PAIRS_PER_LANE = DIMS_PER_LANE / 2;  // 8：lane 以 (2L+64k, 2L+64k+1) 对持有
constexpr int ROPE_LANES = D_ROPE / 8;      // 8：rope 段由 lane 0..7 各持 8 维
constexpr int MAX_SPLITS = 64;
constexpr int CHUNK_TARGET = 256;           // 每 split 目标 token 数（启发式）

// 维度所有权（防 shared bank conflict 的关键，见 EXPERIMENT_LOG §12.0）：
// lane L 持有 latent 维 (2L+64k, 2L+64k+1)，k=0..7 —— acc/行加载按 word 索引
// (2L+64k)/2 = L+32k mod 32 = L → 32 lane 落 32 个不同 bank，零冲突；
// warp 级每 k 一段 128B 连续加载（32 lane × 1 uint），完全 coalesced。
// 旧布局 lane*16+i 的 bank = 16L mod 32 ∈ {0,16} → 16-way conflict，
// shared RMW 慢 16 倍（N=32768 实测 0.81ms 的根因）。

// uint（2 fp16）→ 2 个 float
__device__ __forceinline__ void unpack2(const unsigned int v, float* out) {
    const __half2 h2 = *reinterpret_cast<const __half2*>(&v);
    const float2 f = __half22float2(h2);
    out[0] = f.x;
    out[1] = f.y;
}

// 16B（8 fp16）→ 8 个 float（rope 段用）
__device__ __forceinline__ void unpack8(const uint4& v, float* out) {
    const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h2[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(FULL_MASK, v, off);
    }
    return v;
}

__global__ void mla_split_kernel(
    const __half* __restrict__ q_abs,    // [H, 512]
    const __half* __restrict__ q_rope,   // [H, 64]
    const __half* __restrict__ c_kv,     // [N, 512]
    const __half* __restrict__ k_rope,   // [N, 64]
    float* __restrict__ psum,            // [H, S, 512]（本 split 局部 max 归一的未归一化加权 latent）
    float* __restrict__ pl,              // [H, S]（本 split 局部 max 归一的 exp 和）
    float* __restrict__ pm,              // [H, S]（本 split 局部 max；空 split = -inf）
    const int n,
    const int splits,
    const int chunk,
    const float scale) {
    const int h = static_cast<int>(blockIdx.x);
    const int sp = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int t_start = sp * chunk;
    const int t_end = min(t_start + chunk, n);

    __shared__ float acc[WARPS][D_LAT];  // per-warp 加权 latent 累加器（16KB）
    __shared__ float warp_m[WARPS];
    __shared__ float warp_l[WARPS];
    __shared__ float warp_w[WARPS];      // 块内合并的重归一因子 w_i = exp(m_i - M)；
                                         // acc 的合并必须乘 w_i（不是 l_i！——开发期
                                         // 仿真抓出的第二个 bug，见 EXPERIMENT_LOG §12.0）
    __shared__ float merge_scratch[2];

    // 每 lane 独占 dims (2L+64k, 2L+64k+1)：acc 清零 + q_abs 装寄存器
    float qreg[DIMS_PER_LANE];
    {
        const unsigned int* qv = reinterpret_cast<const unsigned int*>(
            q_abs + static_cast<size_t>(h) * D_LAT);
#pragma unroll
        for (int k = 0; k < PAIRS_PER_LANE; ++k) {
            unpack2(qv[lane + 32 * k], qreg + 2 * k);
        }
    }
#pragma unroll
    for (int k = 0; k < PAIRS_PER_LANE; ++k) {
        acc[warp][2 * lane + 64 * k] = 0.0f;
        acc[warp][2 * lane + 64 * k + 1] = 0.0f;
    }
    // rope 段：lane 0..7 各持 q_rope 的 8 维
    float qrope_reg[8];
    if (lane < ROPE_LANES) {
        const uint4* qr = reinterpret_cast<const uint4*>(
            q_rope + static_cast<size_t>(h) * D_ROPE);
        unpack8(qr[lane], qrope_reg);
    }

    float m = -INFINITY;
    float l = 0.0f;
    // ---- 软件流水（双缓冲）：t 的行在寄存器里，t+WARPS 的行预取 issued——
    // 串行 token 链的 DRAM 延迟与计算重叠（开发期实测：无预取时核是
    // DRAM-latency-bound，N=32768 仅 40 GB/s，见 EXPERIMENT_LOG §12.0）。
    int t = t_start + warp;
    unsigned int cur[PAIRS_PER_LANE];
    unsigned int nxt[PAIRS_PER_LANE];
    uint4 rcur0 = make_uint4(0u, 0u, 0u, 0u), rnext0 = rcur0;
    if (t < t_end) {
        const unsigned int* cv = reinterpret_cast<const unsigned int*>(
            c_kv + static_cast<size_t>(t) * D_LAT);
#pragma unroll
        for (int k = 0; k < PAIRS_PER_LANE; ++k) {
            cur[k] = cv[lane + 32 * k];
        }
        if (lane < ROPE_LANES) {
            rcur0 = reinterpret_cast<const uint4*>(k_rope + static_cast<size_t>(t) * D_ROPE)[lane];
        }
    }
    while (t < t_end) {
        const int tn = t + WARPS;
        if (tn < t_end) {  // 预取下一 token（与本 token 的计算重叠）
            const unsigned int* cvn = reinterpret_cast<const unsigned int*>(
                c_kv + static_cast<size_t>(tn) * D_LAT);
#pragma unroll
            for (int k = 0; k < PAIRS_PER_LANE; ++k) {
                nxt[k] = cvn[lane + 32 * k];
            }
            if (lane < ROPE_LANES) {
                rnext0 = reinterpret_cast<const uint4*>(k_rope + static_cast<size_t>(tn) * D_ROPE)[lane];
            }
        }
        // ---- latent 段点积（q_abs · c_kv[t]，512 维按 lane 分片）----
        float creg[DIMS_PER_LANE];
#pragma unroll
        for (int k = 0; k < PAIRS_PER_LANE; ++k) {
            unpack2(cur[k], creg + 2 * k);
        }
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            dot += qreg[i] * creg[i];
        }
        dot = warp_reduce_sum(dot);

        // ---- rope 段点积（64 维，lane 0..7）----
        float rope_dot = 0.0f;
        if (lane < ROPE_LANES) {
            float kreg[8];
            unpack8(rcur0, kreg);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                rope_dot += qrope_reg[i] * kreg[i];
            }
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                rope_dot += __shfl_xor_sync(0x000000ffu, rope_dot, off);
            }
        }
        rope_dot = __shfl_sync(FULL_MASK, rope_dot, 0);  // 广播到全 warp

        // ---- online softmax 更新 + 加权 latent 累加 ----
        const float score = (dot + rope_dot) * scale;
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m - m_new);   // m=-inf 时 expf(-inf)=0
        const float p = expf(score - m_new);
        l = l * alpha + p;
        if (m_new > m) {
            // 重归一只在运行 max 真正抬升时执行（×expf(0)=1.0 是恒等变换，
            // 跳过与执行位级等价）——随机分数下省掉约一半的 shared RMW
            const float rescale = alpha;
#pragma unroll
            for (int k = 0; k < PAIRS_PER_LANE; ++k) {
                acc[warp][2 * lane + 64 * k] *= rescale;
                acc[warp][2 * lane + 64 * k + 1] *= rescale;
            }
        }
        if (p > 0.0f) {  // p=0（exp 下溢）时 += 0 位级等价，跳过
#pragma unroll
            for (int k = 0; k < PAIRS_PER_LANE; ++k) {
                acc[warp][2 * lane + 64 * k] += p * creg[2 * k];
                acc[warp][2 * lane + 64 * k + 1] += p * creg[2 * k + 1];
            }
        }
        m = m_new;
        // 双缓冲交换
#pragma unroll
        for (int k = 0; k < PAIRS_PER_LANE; ++k) {
            cur[k] = nxt[k];
        }
        rcur0 = rnext0;
        t = tn;
    }

    // ---- 块内合并 8 个 warp 部分和 ----
    warp_m[warp] = m;
    warp_l[warp] = l;
    __syncthreads();
    if (tid == 0) {
        float m_global = -INFINITY;
        for (int i = 0; i < WARPS; ++i) {
            m_global = fmaxf(m_global, warp_m[i]);
        }
        float l_total = 0.0f;
        for (int i = 0; i < WARPS; ++i) {
            // 空 warp（chunk 内无 token，m=-inf）权重视 0，避免 expf(-inf-(-inf))=NaN；
            // 用户强制 splits > N 时整块空 split 也安全（pl=0、psum=0）。
            const float w = (warp_m[i] == -INFINITY)
                                ? 0.0f
                                : expf(warp_m[i] - m_global);
            warp_w[i] = w;       // acc 合并用（乘重归一因子，不含 l_i）
            warp_l[i] *= w;      // 分母用（exp 和 × 因子）
            l_total += warp_l[i];
        }
        merge_scratch[0] = l_total;
    }
    __syncthreads();
    const float l_total = merge_scratch[0];

    float* dst = psum + (static_cast<size_t>(h) * splits + sp) * D_LAT;
    for (int d0 = tid * 2; d0 < D_LAT; d0 += THREADS * 2) {
        float s0 = 0.0f;
        float s1 = 0.0f;
#pragma unroll
        for (int i = 0; i < WARPS; ++i) {
            s0 += warp_w[i] * acc[i][d0];
            s1 += warp_w[i] * acc[i][d0 + 1];
        }
        dst[d0] = s0;
        dst[d0 + 1] = s1;
    }
    if (tid == 0) {
        pl[static_cast<size_t>(h) * splits + sp] = l_total;
        // 本 split 的局部 max（8 个 warp 的 max 已统一到 m_global；
        // 空 split 时 m_global=-inf，merge 侧按 -inf 守卫跳过）
        float m_split = -INFINITY;
        for (int i = 0; i < WARPS; ++i) {
            m_split = fmaxf(m_split, warp_m[i]);
        }
        pm[static_cast<size_t>(h) * splits + sp] = m_split;
    }
}

__global__ void mla_merge_kernel(
    const float* __restrict__ psum,   // [H, S, 512]（各 split 局部 max 归一）
    const float* __restrict__ pl,     // [H, S]
    const float* __restrict__ pm,     // [H, S]（各 split 局部 max）
    __half* __restrict__ s_out,       // [H, 512]
    const int splits) {
    const int h = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    // flash-decoding 跨 split 合并：先把各 split 部分和重归一到全局 max
    // （pm 缺失会把"各自局部 max 归一"的部分和直接相加，结果错误——
    //   本期开发期 CPU 算法仿真抓出的原始 bug，见 kernel_sim.py）。
    float m_global = -INFINITY;
    for (int j = 0; j < splits; ++j) {
        m_global = fmaxf(m_global, pm[static_cast<size_t>(h) * splits + j]);
    }
    float l_total = 0.0f;
    for (int j = 0; j < splits; ++j) {
        const float m_j = pm[static_cast<size_t>(h) * splits + j];
        const float w = (m_j == -INFINITY) ? 0.0f : expf(m_j - m_global);
        l_total += w * pl[static_cast<size_t>(h) * splits + j];
    }
    const float inv = (l_total > 0.0f) ? 1.0f / l_total : 0.0f;  // n≥1 合同下不触发
    for (int d0 = tid * 2; d0 < D_LAT; d0 += THREADS * 2) {
        float s0 = 0.0f;
        float s1 = 0.0f;
        for (int j = 0; j < splits; ++j) {
            const float m_j = pm[static_cast<size_t>(h) * splits + j];
            const float w = (m_j == -INFINITY) ? 0.0f : expf(m_j - m_global);
            const size_t base = (static_cast<size_t>(h) * splits + j) * D_LAT;
            s0 += w * psum[base + d0];
            s1 += w * psum[base + d0 + 1];
        }
        s_out[static_cast<size_t>(h) * D_LAT + d0] = __float2half(s0 * inv);
        s_out[static_cast<size_t>(h) * D_LAT + d0 + 1] = __float2half(s1 * inv);
    }
}

}  // namespace

// ----------------------------------------------------------------------------
// torch 扩展入口：合同校验显式报错，不静默回退（第 02 期风格）。
// splits=0 → 启发式（ceil(N/256) 夹在 [1,64]）；>0 → 指定（夹在 [1,64]）。
// ----------------------------------------------------------------------------

at::Tensor mla_absorbed_attention(
    const at::Tensor& q_abs,
    const at::Tensor& q_rope,
    const at::Tensor& c_kv,
    const at::Tensor& k_rope,
    double scale,
    int64_t splits_req) {
    TORCH_CHECK(q_abs.scalar_type() == at::kHalf &&
                q_rope.scalar_type() == at::kHalf &&
                c_kv.scalar_type() == at::kHalf &&
                k_rope.scalar_type() == at::kHalf,
                "合同：本期 CUDA 核 io dtype = float16，实际 ",
                q_abs.scalar_type(), "/", q_rope.scalar_type(), "/",
                c_kv.scalar_type(), "/", k_rope.scalar_type());
    TORCH_CHECK(q_abs.device().type() == c10::DeviceType::CUDA &&
                c_kv.device().type() == c10::DeviceType::CUDA,
                "本扩展是 CUDA 实现，不接受 ", q_abs.device(), " Tensor");
    TORCH_CHECK(q_abs.is_contiguous() && q_rope.is_contiguous() &&
                c_kv.is_contiguous() && k_rope.is_contiguous(),
                "合同：本期 Kernel 只接受连续输入（非连续布局是第 06 期主题）");
    TORCH_CHECK(q_abs.dim() == 2 && q_abs.size(1) == D_LAT,
                "q_abs 必须是 [H, 512]，实际 ", q_abs.sizes());
    TORCH_CHECK(q_rope.dim() == 2 && q_rope.size(1) == D_ROPE,
                "q_rope 必须是 [H, 64]，实际 ", q_rope.sizes());
    TORCH_CHECK(c_kv.dim() == 2 && c_kv.size(1) == D_LAT,
                "c_kv 必须是 [N, 512]，实际 ", c_kv.sizes());
    TORCH_CHECK(k_rope.dim() == 2 && k_rope.size(1) == D_ROPE,
                "k_rope 必须是 [N, 64]，实际 ", k_rope.sizes());
    const int heads = static_cast<int>(q_abs.size(0));
    TORCH_CHECK(q_rope.size(0) == heads && c_kv.size(0) > 0,
                "H 不一致或 N=0（合同：N≥1）");
    TORCH_CHECK(k_rope.size(0) == c_kv.size(0),
                "c_kv 与 k_rope 的 N 不一致：", c_kv.size(0), " vs ", k_rope.size(0));

    const int n = static_cast<int>(c_kv.size(0));
    int splits;
    if (splits_req > 0) {
        splits = static_cast<int>(std::min<int64_t>(splits_req, MAX_SPLITS));
    } else {
        splits = (n + CHUNK_TARGET - 1) / CHUNK_TARGET;
        splits = std::max(1, std::min(splits, MAX_SPLITS));
    }
    int chunk = (n + splits - 1) / splits;
    chunk = ((chunk + WARPS - 1) / WARPS) * WARPS;  // warp 步长对齐
    chunk = std::max(chunk, WARPS);

    auto s_out = at::empty({heads, D_LAT}, q_abs.options());
    auto psum = at::empty({heads, splits, D_LAT},
                          q_abs.options().dtype(at::kFloat));
    auto pl = at::empty({heads, splits}, q_abs.options().dtype(at::kFloat));
    auto pm = at::empty({heads, splits}, q_abs.options().dtype(at::kFloat));

    const at::cuda::CUDAGuard device_guard(q_abs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(splits));
    mla_split_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __half*>(q_abs.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(q_rope.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(c_kv.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_rope.data_ptr<at::Half>()),
        psum.data_ptr<float>(),
        pl.data_ptr<float>(),
        pm.data_ptr<float>(),
        n, splits, chunk, static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    mla_merge_kernel<<<dim3(static_cast<unsigned>(heads)), THREADS, 0, stream>>>(
        psum.data_ptr<float>(),
        pl.data_ptr<float>(),
        pm.data_ptr<float>(),
        reinterpret_cast<__half*>(s_out.data_ptr<at::Half>()),
        splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return s_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("mla_absorbed_attention", &mla_absorbed_attention,
               py::arg("q_abs"), py::arg("q_rope"), py::arg("c_kv"),
               py::arg("k_rope"), py::arg("scale"), py::arg("splits") = 0,
               "第 24 期 MLA 吸收形式 decode attention 核（split-K flash-decoding 两 kernel）");
}
