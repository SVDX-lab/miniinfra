// ============================================================================
// 第 19 期 MUSA 迁移版（MTT S4000 / MUSA SDK 5.2.0 / mcc）——入口包装
//
// 迁移策略：与 CUDA 版【同一份源码】（第 17 期方案延续，API 差异集中在
// USE_MUSA shim）。异步拷贝机制（cp.async 对应物）是本期 MUSA 侧"待实机
// 确认"项（大纲 §14 准入第 10 条）：
//   - async_probe/probe.sh 先行探测（头文件 grep + __cvta_generic_to_shared
//     + cp.async 内联 PTX + pipeline intrinsics 四级探测，全程留证据）；
//   - 探测通过 → build.sh 加 -DHAVE_MUSA_CP_ASYNC 编出 cpasync1/cpasync2/
//     warpspec（wmma_pipe 默认关闭：ep19 M-23 vendor 缺陷，-DHAVE_MUSA_WMMA
//     显式打开 = 修复后复测）；
//   - 探测失败 → 仅编 sync 级，cp.async 记平台边界（报错全文 = 生态结论）。
// ============================================================================
#define USE_MUSA 1
#include "../cuda/naive_gemm_async.cu"
