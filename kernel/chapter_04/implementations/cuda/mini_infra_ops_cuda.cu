// ============================================================================
// mini_infra_ops_cuda.cu —— 第 04 期 CUDA 注册 TU：进入框架的 Dispatch。
//
// 注册机制（大纲 §13 第 04 期"实现"条目，逐项对应）：
//   1. schema 声明：TORCH_LIBRARY(mini_infra_ops, m) —— 两个同名族 op：
//        fused_add_rmsnorm            —— CUDA impl + CPU composite fallback impl
//                                        （显式注册的 fallback：谁触发、何时触发
//                                         全部留痕，对照"静默回退"）；
//        fused_add_rmsnorm_no_cpu_fallback —— 只注册 CUDA impl；CPU 张量调用
//                                        它会得到框架的显式 NotImplementedError
//                                        （错误原文进结果 JSON —— "无注册即显式
//                                         报错"的框架级证据）。
//      schema（vLLM fused_add_rms_norm 同款双 mutator、零返回、零分配）：
//        fused_add_rmsnorm(Tensor(a!) input, Tensor(b!) residual,
//                          Tensor weight, float eps) -> ()
//        语义：residual ← residual + input（T 一次舍入）；
//              input    ← RMSNorm(residual) * weight（平方和/均值 FP32）。
//   2. CUDA dispatch key：TORCH_LIBRARY_IMPL(mini_infra_ops, CUDA, m)；
//   3. 当前 Stream：c10::cuda::getCurrentCUDAStream()（与 torch eager 执行流
//      对齐；第 03 期 ctypes 路径用默认流，注册路径必须取框架当前流）；
//   4. 设备/Shape/dtype 检查 + 错误传播：TORCH_CHECK（错误文本即平台边界
//      与合同证据，Python 侧原样捕获进结果 JSON）；
//   5. 构建：torch.utils.cpp_extension.load（build.py）。
//
// Kernel 复用声明：本 TU #include "fused_rmsnorm_kernels.cu"（第 03 期融合
// kernel 库**逐字原样**，内含第 02 期 rmsnorm kernel），device 代码零改动；
// 融合 kernel 模板 fused_add_rmsnorm_kernel<T, TILE, VEC> 由本 TU 的
// launch_fused_add_rmsnorm_streamed 直接实例化——与第 03 期 extern "C"
// 启动器唯一的结构差异是多了 cudaStream_t 参数（ctypes 路径默认流）。
// 第 03 期 extern "C" 入口（fused_add_rmsnorm_rows 等）保留在本 TU 中：
// 同一个 .so 同时可用 torch.ops 与 ctypes 两种调用路径（dispatch 开销对照）。
//
// aliasing 声明（合同 aliasing 条目）：op 语义下 y 写回 input buffer，即
// 融合 kernel 的 y 指针 == h_in 指针（刻意 alias，vLLM 同款零分配）；
// kernel 内 Pass 1 读 h / Pass 2 写 y 以 __syncthreads 分隔、每行独占单
// block，alias 安全；residual/input/weight 三指针必须互不相同（TORCH_CHECK）。
// ============================================================================

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdio>

#include "fused_rmsnorm_kernels.cu"

namespace {

using namespace mini_infra;

// ---------------------------------------------------------------------------
// 融合 kernel 的流感知启动器（与第 03 期 fused_add_rmsnorm_rows 同一 dispatch
// 结构，唯一差异：<<<..., stream>>>；tile 固定 256 由调用方保证合法性）。
// ---------------------------------------------------------------------------
void launch_fused_add_rmsnorm_streamed(void* r, const void* h_in, const void* w,
                                       void* y, bool is_fp16, int rows, int h_dim,
                                       float eps, int tile, cudaStream_t stream) {
    if (h_dim <= 0 || rows <= 0) {
        return;  // 空 Tensor：不发射 Kernel（合同条目；rows<0 已在检查层拒绝）
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
// 合同校验（CUDA/CPU 两个 impl 共用的 shape/dtype/aliasing 检查；
// 设备检查各自做——错误文本即合同证据）。
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
// CUDA impl（dispatch key: CUDA）
// ---------------------------------------------------------------------------
void fused_add_rmsnorm_cuda_impl(at::Tensor input, at::Tensor residual,
                                 at::Tensor weight, double eps) {
    validate_contract(input, residual, weight, eps);
    TORCH_CHECK(input.is_cuda() && residual.is_cuda() && weight.is_cuda(),
                "CUDA impl 只接受 CUDA Tensor（显式报错，不静默回退）；实际 device=",
                input.device());
    validate_aliasing(input, residual, weight);

    const at::cuda::CUDAGuard device_guard(residual.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const bool is_fp16 = input.scalar_type() == at::kHalf;
    const int64_t hidden_dim = input.size(-1);
    const int64_t rows = input.numel() / hidden_dim;
    launch_fused_add_rmsnorm_streamed(
        residual.data_ptr(), input.const_data_ptr(), weight.const_data_ptr(),
        input.data_ptr(),  // y 刻意 alias input buffer（op 语义：input ← y）
        is_fp16, static_cast<int>(rows), static_cast<int>(hidden_dim),
        static_cast<float>(eps), /*tile=*/256, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// CPU composite impl（dispatch key: CPU）——显式注册的 fallback 语义载体。
// 与 implementations/cpu/fused_add_rmsnorm_ref.py 同一数学（ATen 表达式），
// 触发时打印一行留痕（stderr）——"显式 fallback"对照"静默回退"的 C++ 侧证据。
// ---------------------------------------------------------------------------
void fused_add_rmsnorm_cpu_impl(at::Tensor input, at::Tensor residual,
                                at::Tensor weight, double eps) {
    validate_contract(input, residual, weight, eps);
    TORCH_CHECK(input.is_cpu() && residual.is_cpu() && weight.is_cpu(),
                "CPU impl 只接受 CPU Tensor；实际 device=", input.device(),
                "（GPU Tensor 应由 CUDA/PrivateUse1 impl 接管——收到本错误说明"
                "GPU impl 未注册，属显式失败而非静默回退）");
    validate_aliasing(input, residual, weight);
    std::fprintf(stderr,
                 "[mini_infra_ops] CPU composite fallback: fused_add_rmsnorm "
                 "(rows=%lld, H=%lld, dtype=%s) —— 显式留痕\n",
                 static_cast<long long>(input.size(0)),
                 static_cast<long long>(input.size(1)),
                 input.scalar_type() == at::kHalf ? "fp16" : "fp32");
    // Pass 1：t = r + h 原位写回 residual。
    residual.add_(input);
    // Pass 2：FP32 方差 + rsqrt + 权重缩放（第 01 期手写 RMSNorm 同一表达式）。
    auto float_states = residual.to(at::kFloat);
    auto variance = float_states.pow(2).mean(-1, true);
    auto normalized = float_states * at::rsqrt(variance + eps);
    input.copy_(weight * normalized.to(residual.scalar_type()));
}

}  // namespace

// ============================================================================
// 注册（schema 声明 + dispatch key 绑定）
// ============================================================================
TORCH_LIBRARY(mini_infra_ops, m) {
    m.def("fused_add_rmsnorm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps) -> ()");
    m.def("fused_add_rmsnorm_no_cpu_fallback(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps) -> ()");
}

TORCH_LIBRARY_IMPL(mini_infra_ops, CUDA, m) {
    m.impl("fused_add_rmsnorm", &fused_add_rmsnorm_cuda_impl);
    m.impl("fused_add_rmsnorm_no_cpu_fallback", &fused_add_rmsnorm_cuda_impl);
}

TORCH_LIBRARY_IMPL(mini_infra_ops, CPU, m) {
    // 只给 fused_add_rmsnorm 注册 CPU fallback（no_cpu_fallback 族故意不注册，
    // 用于采集框架对"无注册后端调用"的显式报错原文）。
    m.impl("fused_add_rmsnorm", &fused_add_rmsnorm_cpu_impl);
}

// ============================================================================
// pybind 最小入口：cpp_extension.load 的产物必须是可 import 的 python 扩展
// 模块；TORCH_LIBRARY 注册在 dlopen 时的静态初始化中完成，pybind 部分只
// 暴露注册元信息（供正确性脚本记录证据）。
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("registration_info", []() {
        return std::string(
            "TORCH_LIBRARY(mini_infra_ops): "
            "fused_add_rmsnorm [CUDA kernel + CPU composite fallback], "
            "fused_add_rmsnorm_no_cpu_fallback [CUDA only]; "
            "schema: (Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps) -> (); "
            "kernel: chapter_03 fused_add_rmsnorm_kernel<T,TILE,VEC> verbatim; "
            "stream: c10::cuda::getCurrentCUDAStream()");
    });
    m.def("fused_add_rmsnorm_rows_c_abi", &fused_add_rmsnorm_rows,
          "第 03 期 extern \"C\" ctypes 入口（同一 .so 双通道对照用）",
          py::arg("r_ptr"), py::arg("h_ptr"), py::arg("w_ptr"), py::arg("y_ptr"),
          py::arg("is_fp16"), py::arg("rows"), py::arg("h_dim"), py::arg("eps"),
          py::arg("tile"), py::arg("warp_size"));
}
