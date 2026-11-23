// ============================================================================
// synccheck_probe.cu —— 第 10 期：synccheck 可检测面实测探针（4090/CUDA 13
// /sanitizer 2025.3.1/sm_89 实测三形态，结论：全部 0 报告——见 EXPERIMENT_LOG）。
//
//   形态 1 lane_divergent_barrier：barrier 仅由每 warp 的 lane 0 执行（warp
//          内发散），其余线程走函数尾退出 → kernel 正常完成（sm_89 对已退出
//          线程按"隐式到达"处理）→ synccheck 0 errors（--print-level info 同）。
//   形态 2（defect_suite.cu V1，已废弃）：barrier 在偶数分支、奇数线程直接
//          退出 → 完成且结果正确 → synccheck 0 errors。
//   形态 3（defect_suite.cu V3 即现行 case=sync）：奇数线程读 shared 恒真
//          条件自旋（活着、永不到达）→ 真 deadlock（裸跑 timeout=124）→
//          synccheck 无任何报告，进程挂死被外部 timeout 截断。
//
// 教学结论（验收 Q1 的 sync 类答案）：synccheck 名义上是同步误用的对应工具，
// 但 2025 实测栈上三种经典形态全部沉默；实际本地化路径 = 裸跑挂死特征 +
// 代码审查。编译器（nvcc）对 barrier 误用同样零警告。历史教材示例在
// 现代 stack 上"示例失效"是工具生态演化的实证素材。
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>

__global__ void lane_divergent_barrier(float* y) {
    __shared__ float s[256];
    const int tid = threadIdx.x;
    s[tid] = static_cast<float>(tid);
    if (tid % 32 == 0) {          // 每 warp 仅 lane 0 —— warp 内发散
        __syncthreads();          // 仅 1/32 lanes 到达的 barrier
        y[tid] = s[255 - tid];
    } else {
        y[tid] = -1.0f;
    }
    // 其余线程走完函数尾退出：exit-arrival 宽容 → kernel 完成 → 报告可落盘
}

int main() {
    float* dy;
    cudaMalloc(&dy, 256 * 4);
    cudaMemset(dy, 0xEE, 256 * 4);
    lane_divergent_barrier<<<2, 256>>>(dy);
    std::printf("kernel done: %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();
    std::printf("sync ok\n");
    return 0;
}
