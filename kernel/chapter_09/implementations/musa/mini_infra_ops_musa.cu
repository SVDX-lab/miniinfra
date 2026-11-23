// ============================================================================
// mini_infra_ops_musa.cu —— 第 04 期 MUSA 注册 TU（CUDA 版
// mini_infra_ops_cuda.cu 的迁移对应物）：torch_musa 扩展注册路径，
// PrivateUse1 后端 dispatch。
//
// 与 CUDA 版的逐项结构对照（详见 cross-backend-comparison.md）：
//   CUDA                                | MUSA（本文件）
//   ------------------------------------|----------------------------------
//   TORCH_LIBRARY_IMPL(..., CUDA, ...)  | TORCH_LIBRARY_IMPL(..., PrivateUse1, ...)
//                                       （torch_musa 张量的 dispatch key；
//                                        报错原文/key 名以实机为准并记录）
//   c10::cuda::getCurrentCUDAStream()   | 流获取两变体（build.py 依次尝试）：
//                                       变体 0：c10::musa::getCurrentMUSAStream()
//                                         ——第 02 期 M-10 已知与 torch 官方头
//                                           redefinition，预期失败（证据留档）；
//                                       变体 1：默认流（kernel<<<...>>> 无 stream
//                                         参数；torch_musa eager 单流场景与
//                                         框架当前流等价——ep03/ep04 实测口径，
//                                         多 stream 语义第 08 期核对）
//   at::cuda::CUDAGuard                 | 变体 1 无 guard（单卡场景，ep03 先例）
//   C10_CUDA_KERNEL_LAUNCH_CHECK()      | musaGetLastError 同构检查（如可用）
//   nvcc cpp_extension                  | torch_musa.utils.musa_extension.load
//                                       （M-08/M-09：单编译单元 .mu、include
//                                         行改写，build.py 处理）
//
// Kernel 复用：#include "fused_rmsnorm_kernels.mu"（第 03 期 MUSA 迁移版
// kernel 库逐字原样，device 代码零改动）；模板由本 TU 的
// launch_fused_add_rmsnorm_streamed 实例化。
// CPU composite fallback impl 与 CUDA 版逐字相同（同一段 ATen 表达式）。
// ============================================================================

#include <torch/extension.h>

#include <cstdint>
#include <cstdio>

#include "fused_rmsnorm_kernels.mu"

// 流获取变体（build.py 依次尝试；默认变体 1 = 默认流）。
#if !defined(MINI_INFRA_MUSA_STREAM_VARIANT)
#define MINI_INFRA_MUSA_STREAM_VARIANT 1
#endif

#if MINI_INFRA_MUSA_STREAM_VARIANT == 0
#include <c10/musa/MUSAStream.h>
#define MINI_INFRA_CURRENT_MUSA_STREAM() c10::musa::getCurrentMUSAStream().stream()
#elif MINI_INFRA_MUSA_STREAM_VARIANT == 1
// 变体 1：默认流。torch_musa eager 默认单流执行（ep01–ep04 脚本均未用 side
// stream），本发射与框架当前流等价；M-10（MUSAStream.h redefinition）规避。
#define MINI_INFRA_CURRENT_MUSA_STREAM() 0
#else
// 变体 2（第 09 期新增）：默认流 + 显式覆盖流（Graph 捕获兼容层）。
// 实测背景（ep10）：torch_musa CUDAGraph 捕获期内，default-stream（0）上的
// kernel **静默不入图**（MUSAGraph.cpp:160 仅在整图为空时 UserWarning，
// 混合图完全无声→错误结果）；而 c10::musa::getCurrentMUSAStream() 在头文件
// 层面不可编译（M-28/M-31 族）。因此 Python 侧在捕获窗口内经 pybind 注入
// 捕获流句柄（set_launch_stream），kernel 按显式流发射 → 与捕获流同流 →
// 入图。eager 场景覆盖流为 null，行为与变体 1 完全一致。
static ::musaStream_t g_ep10_override_stream = nullptr;
#define MINI_INFRA_CURRENT_MUSA_STREAM() \
    (g_ep10_override_stream ? g_ep10_override_stream : static_cast<::musaStream_t>(0))
#endif

namespace {

using namespace mini_infra;

// ---------------------------------------------------------------------------
// 融合 kernel 的流感知启动器（与 CUDA 版同一 dispatch 结构；
// stream 参数在变体 1 下恒为默认流 0）。
// ---------------------------------------------------------------------------
void launch_fused_add_rmsnorm_streamed(void* r, const void* h_in, const void* w,
                                       void* y, bool is_fp16, int rows, int h_dim,
                                       float eps, int tile,
                                       musaStream_t stream) {
    if (h_dim <= 0 || rows <= 0) {
        return;  // 空 Tensor：不发射 Kernel（合同条目）
    }
    const bool vec = is_fp16 ? vec_feasible4<__half>(r, h_in, w, y, h_dim)
                             : vec_feasible4<float>(r, h_in, w, y, h_dim);
    dim3 block(tile);
    dim3 grid(static_cast<unsigned>(rows < kMaxGridBlocks ? rows : kMaxGridBlocks));

    #define FUSED_LAUNCH_S(T_NAME, TILE_V, VEC_V)                                    \
        do {                                                                          \
            fused_add_rmsnorm_kernel<T_NAME, TILE_V, VEC_V>                           \
                <<<grid, block, 0, stream>>>(static_cast<T_NAME*>(r),                 \
                                             static_cast<const T_NAME*>(h_in),        \
                                             static_cast<const T_NAME*>(w),           \
                                             static_cast<T_NAME*>(y),                 \
                                             rows, h_dim, eps);                       \
        } while (0)
    #define FUSED_DISPATCH_VEC_S(T_NAME, TILE_V)                                     \
        do {                                                                          \
            if (vec) { FUSED_LAUNCH_S(T_NAME, TILE_V, true); }                        \
            else     { FUSED_LAUNCH_S(T_NAME, TILE_V, false); }                       \
        } while (0)
    #define FUSED_DISPATCH_TILE_S(T_NAME)                                            \
        do {                                                                          \
            switch (tile) {                                                           \
                case 64:   FUSED_DISPATCH_VEC_S(T_NAME, 64);   break;                 \
                case 128:  FUSED_DISPATCH_VEC_S(T_NAME, 128);  break;                 \
                case 256:  FUSED_DISPATCH_VEC_S(T_NAME, 256);  break;                 \
                case 512:  FUSED_DISPATCH_VEC_S(T_NAME, 512);  break;                 \
                case 1024: FUSED_DISPATCH_VEC_S(T_NAME, 1024); break;                 \
                default: TORCH_CHECK(false, "tile 必须是 {64,128,256,512,1024} 之一，实际 ", tile); \
            }                                                                         \
        } while (0)
    if (is_fp16) {
        FUSED_DISPATCH_TILE_S(__half);
    } else {
        FUSED_DISPATCH_TILE_S(float);
    }
    #undef FUSED_LAUNCH_S
    #undef FUSED_DISPATCH_VEC_S
    #undef FUSED_DISPATCH_TILE_S
}

// ---------------------------------------------------------------------------
// 合同校验（与 CUDA 版逐字相同——迁移差异为零的部件）。
// ---------------------------------------------------------------------------
void validate_contract(const at::Tensor& input, const at::Tensor& residual,
                       const at::Tensor& weight, double eps) {
    // 形状合同（vLLM fused_add_rms_norm 同款）：input/residual 同形状、末维 H、
    // 任意 leading 维度（模型激活为 [batch, seq, H]，内部展平 rows = numel/H）。
    TORCH_CHECK(input.dim() >= 1, "input 至少一维，实际 dim=", input.dim());
    TORCH_CHECK(residual.sizes() == input.sizes(),
                "residual 与 input 形状必须一致：", residual.sizes(), " vs ", input.sizes());
    TORCH_CHECK(weight.dim() == 1 && weight.numel() == input.size(-1),
                "weight 必须是 [H] 且与 input 的末维一致：w.numel()=", weight.numel(),
                " input.size(-1)=", input.size(-1));
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf,
                "本期合同 dtype ∈ {float32, float16}，实际 ", input.scalar_type(),
                "（bfloat16 属第 05 期；显式报错，不静默回退）");
    TORCH_CHECK(residual.scalar_type() == input.scalar_type() &&
                weight.scalar_type() == input.scalar_type(),
                "input/residual/weight dtype 必须一致：", input.scalar_type(), " / ",
                residual.scalar_type(), " / ", weight.scalar_type());
    TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(),
                "合同：本期只接受连续 Tensor（非连续布局是第 06 期主题）");
    TORCH_CHECK(input.size(-1) > 0, "合同：H=0 为空 Tensor，显式报错（不支持静默处理）");
    TORCH_CHECK(eps > 0.0 && eps < 1.0, "eps 合同范围 (0,1)，实际 ", eps);
}

void validate_aliasing(const at::Tensor& input, const at::Tensor& residual,
                       const at::Tensor& weight) {
    const void* in_p = input.const_data_ptr();
    const void* res_p = residual.const_data_ptr();
    const void* w_p = weight.const_data_ptr();
    TORCH_CHECK(res_p != in_p, "residual 与 input 不能是同一 buffer（op 会分别改写两者）");
    TORCH_CHECK(res_p != w_p && in_p != w_p, "weight 不能与 input/residual 共用 buffer");
}

// ---------------------------------------------------------------------------
// MUSA impl（dispatch key: PrivateUse1）
// ---------------------------------------------------------------------------
void fused_add_rmsnorm_musa_impl(at::Tensor input, at::Tensor residual,
                                 at::Tensor weight, double eps) {
    validate_contract(input, residual, weight, eps);
    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                residual.device().type() == c10::DeviceType::PrivateUse1 &&
                weight.device().type() == c10::DeviceType::PrivateUse1,
                "MUSA impl 只接受 torch_musa（PrivateUse1）Tensor（显式报错，不静默回退）；实际 device=",
                input.device());
    validate_aliasing(input, residual, weight);

    const bool is_fp16 = input.scalar_type() == at::kHalf;
    const int64_t hidden_dim = input.size(-1);
    const int64_t rows = input.numel() / hidden_dim;
    launch_fused_add_rmsnorm_streamed(
        residual.data_ptr(), input.const_data_ptr(), weight.const_data_ptr(),
        input.data_ptr(),  // y 刻意 alias input buffer（op 语义：input ← y）
        is_fp16, static_cast<int>(rows), static_cast<int>(hidden_dim),
        static_cast<float>(eps), /*tile=*/256, MINI_INFRA_CURRENT_MUSA_STREAM());
}

// ---------------------------------------------------------------------------
// CPU composite impl（与 CUDA 版逐字相同：显式注册的 fallback 语义载体）。
// ---------------------------------------------------------------------------
void fused_add_rmsnorm_cpu_impl(at::Tensor input, at::Tensor residual,
                                at::Tensor weight, double eps) {
    validate_contract(input, residual, weight, eps);
    TORCH_CHECK(input.is_cpu() && residual.is_cpu() && weight.is_cpu(),
                "CPU impl 只接受 CPU Tensor；实际 device=", input.device(),
                "（GPU Tensor 应由 PrivateUse1 impl 接管——收到本错误说明 GPU impl "
                "未注册，属显式失败而非静默回退）");
    validate_aliasing(input, residual, weight);
    std::fprintf(stderr,
                 "[mini_infra_ops] CPU composite fallback: fused_add_rmsnorm "
                 "(rows=%lld, H=%lld, dtype=%s) —— 显式留痕\n",
                 static_cast<long long>(input.size(0)),
                 static_cast<long long>(input.size(1)),
                 input.scalar_type() == at::kHalf ? "fp16" : "fp32");
    residual.add_(input);
    auto float_states = residual.to(at::kFloat);
    auto variance = float_states.pow(2).mean(-1, true);
    auto normalized = float_states * at::rsqrt(variance + eps);
    input.copy_(weight * normalized.to(residual.scalar_type()));
}

}  // namespace

// ============================================================================
// 注册（schema 与 CUDA 版逐字相同；PrivateUse1 后端绑定）
// ============================================================================
TORCH_LIBRARY(mini_infra_ops, m) {
    m.def("fused_add_rmsnorm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps) -> ()");
    m.def("fused_add_rmsnorm_no_cpu_fallback(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps) -> ()");
}

TORCH_LIBRARY_IMPL(mini_infra_ops, PrivateUse1, m) {
    m.impl("fused_add_rmsnorm", &fused_add_rmsnorm_musa_impl);
    m.impl("fused_add_rmsnorm_no_cpu_fallback", &fused_add_rmsnorm_musa_impl);
}

TORCH_LIBRARY_IMPL(mini_infra_ops, CPU, m) {
    m.impl("fused_add_rmsnorm", &fused_add_rmsnorm_cpu_impl);
}

// ============================================================================
// pybind 最小入口（musa_extension.load 产物必须可 import；注册在 dlopen 静态
// 初始化完成）。MUSA 版额外暴露流变体编号，供证据链记录。
// 变体 2 追加 set_launch_stream：Graph 捕获兼容层（见文件头注释）。
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("registration_info", []() {
        return std::string(
            "TORCH_LIBRARY(mini_infra_ops) via torch_musa PrivateUse1: "
            "fused_add_rmsnorm [PrivateUse1 kernel + CPU composite fallback], "
            "fused_add_rmsnorm_no_cpu_fallback [PrivateUse1 only]; "
            "stream_variant=") + std::to_string(MINI_INFRA_MUSA_STREAM_VARIANT) +
            std::string(MINI_INFRA_MUSA_STREAM_VARIANT == 0
                            ? " (c10::musa::getCurrentMUSAStream)"
                            : (MINI_INFRA_MUSA_STREAM_VARIANT == 2
                                   ? " (default stream + explicit override for graph capture, ep10)"
                                   : " (default stream, M-10 workaround)"));
    });
    m.def("stream_variant", []() { return MINI_INFRA_MUSA_STREAM_VARIANT; });
#if MINI_INFRA_MUSA_STREAM_VARIANT >= 2
    m.def("set_launch_stream", [](uintptr_t stream_ptr) {
        g_ep10_override_stream = reinterpret_cast<::musaStream_t>(stream_ptr);
    });
#endif
}
