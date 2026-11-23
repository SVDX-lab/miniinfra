// ============================================================================
// defect_suite.cu —— 第 10 期 CUDA 列：故意有缺陷的 Kernel 集合（教学体检体）。
//
// 四个 case，每个恰好植入一处缺陷（bad）+ 对应健康版（good）：
//   case=oob    缺陷=越界访存（循环哨兵 <= 导致末元素后越界写/读）
//               工具=memcheck   （Out-of-bounds Write/Read）
//   case=race   缺陷=共享内存累加器缺失 __syncthreads()（WAW+RAW 竞态）
//               工具=racecheck  （Write-Write / Read-After-Write race on shared）
//   case=uninit 缺陷=读取 cudaMalloc 后从未写入的"魔数"缓冲
//               工具=initcheck  （Uninitialized Read）
//   case=sync   缺陷=__syncthreads() 位于线程发散分支内（块内半数线程到达）
//               工具=synccheck  （Barrier 违例；同时预期真挂死——真实教训）
//
// 独立可执行（main 自带 cudaMalloc/种子数据/自检）：compute-sanitizer 以
// 进程为单位包裹，独立二进制是最干净的体检对象（无 torch/驱动噪声）。
// bad 版不 crash 主机进程（越界写落在分配的 slack/邻块，属"未爆雷"状态
// ——这正是 memcheck 的价值：爆不爆雷与错不错是两回事）。
//
// 自检口径：good 版与 host 串行参考 max_abs_error 应为 0（逐位一致）；
// bad 版打印"结果损坏证据"（race 版跨 5 次运行 unique 值计数；其余打印
// 与参考的偏差）——证明缺陷确实改变了程序语义，而非"工具误报"。
// ============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t _err = (expr);                                                \
        if (_err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d (%s)\n",                 \
                         cudaGetErrorString(_err), __FILE__, __LINE__, #expr);    \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

// 确定性数据（splitmix64 + Box-Muller，与第 02 期 bench 同法）
static double splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return static_cast<double>(z ^ (z >> 31)) / 18446744073709551616.0;
}
static void fill_normal(std::vector<float>& data, uint64_t seed) {
    uint64_t state = seed;
    for (size_t i = 0; i + 1 < data.size(); i += 2) {
        double u1 = splitmix64(state), u2 = splitmix64(state);
        if (u1 < 1e-12) u1 = 1e-12;
        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;
        data[i] = static_cast<float>(r * std::cos(theta));
        data[i + 1] = static_cast<float>(r * std::sin(theta));
    }
    if (data.size() % 2 == 1) {
        double u1 = splitmix64(state), u2 = splitmix64(state);
        data[data.size() - 1] = static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                                                   std::cos(2.0 * 3.14159265358979323846 * u2));
    }
}

// ---------------------------------------------------------------------------
// case 1：越界访存（memcheck 体检体）
//   缺陷：哨兵写成 j <= n（应为 j < n）——最末元素后 1 个 float 越界写+读。
//   现实原型：copy/tile kernel 的 off-by-one；无 sanitizer 时 9 成不爆雷。
// ---------------------------------------------------------------------------
__global__ void scale_all_bad(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int stride = gridDim.x * blockDim.x;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j <= n; j += stride) {  // 缺陷行
        y[j] = 2.0f * x[j];
    }
}

__global__ void scale_all_good(const float* __restrict__ x, float* __restrict__ y, int n) {
    const int stride = gridDim.x * blockDim.x;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride) {
        y[j] = 2.0f * x[j];
    }
}

// ---------------------------------------------------------------------------
// case 2：共享内存竞态（racecheck 体检体）
//   缺陷：s_acc 的"线程 0 初始化"与"全体累加"之间缺 __syncthreads()，
//         且累加本身是裸 RMW（非 atomic）→ WAW / RAW 竞态双料。
//   现实原型：手写 block 归约的"忘了 barrier"——比越界更阴险（结果随机对错）。
// ---------------------------------------------------------------------------
__global__ void block_sum_bad(const float* __restrict__ x, float* __restrict__ y, int n) {
    __shared__ float s_acc;
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    const float local = (i < n) ? x[i] : 0.0f;
    if (tid == 0) s_acc = 0.0f;
    // 缺陷：此处缺失 __syncthreads() —— 初始化与累加无同步约束
    s_acc = s_acc + local;          // 全体线程裸读改写共享累加器（竞态本体）
    __syncthreads();
    if (tid == 0) y[blockIdx.x] = s_acc;
}

// 健康版：线程私有槽位 + barrier + 树形归约（无竞态、结果确定）
__global__ void block_sum_good(const float* __restrict__ x, float* __restrict__ y, int n) {
    __shared__ float s_part[256];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    s_part[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) s_part[tid] += s_part[tid + step];
        __syncthreads();
    }
    if (tid == 0) y[blockIdx.x] = s_part[0];
}

// ---------------------------------------------------------------------------
// case 3：未初始化读（initcheck 体检体）
//   缺陷：magic 缓冲 cudaMalloc 后从未写入，kernel 直接读它当乘数。
//   现实原型：忘了 memset/copy 的 accumulator 或 scale buffer。
// ---------------------------------------------------------------------------
__global__ void mul_by_magic_bad(const float* __restrict__ x, const float* __restrict__ magic,
                                 float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float m = magic[0];   // 缺陷行：magic 从未初始化
        y[i] = x[i] * m;
    }
}

__global__ void mul_by_magic_good(const float* __restrict__ x, const float* __restrict__ magic,
                                  float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float m = magic[0];   // 健康版：host 侧 cudaMemset 写入后再 launch
        y[i] = x[i] * m;
    }
}

// ---------------------------------------------------------------------------
// case 4：同步误用（synccheck 体检体）
//   缺陷：__syncthreads() 在线程发散分支内，且分支外线程"活着但永不到达"
//   （自旋等待）→ 块内部分线程到达 barrier 的死锁形态。
//   设计注记（V1 教训）：初版把 barrier 放在偶数分支、奇数线程直接退出——
//   sm_89 硬件对"已退出线程"按隐式到达处理，kernel 正常完成且结果正确，
//   synccheck 0 errors（"退出型误用被硬件宽容"本身是有价值的实测结论，
//   见 EXPERIMENT_LOG）。V2 用活线程自旋构造真死锁：synccheck 应在 barrier
//   执行点报告发散 barrier，随后进程真挂死由外部 timeout 截断。
//   （初版还有第二个意外缺陷：y 只被偶数线程写过——initcheck 在本 case
//   抓到 32 个 cudaMemcpy 源未初始化访问，已改为全元素写齐隔离单缺陷。）
// ---------------------------------------------------------------------------
__global__ void staged_bad(float* __restrict__ y) {
    __shared__ float s[256];
    const int tid = threadIdx.x;
    s[tid] = static_cast<float>(tid);
    if (tid % 2 == 0) {                 // 发散分支
        __syncthreads();                // 缺陷行：仅偶数线程到达
        y[tid] = s[254 - tid];
    } else {
        y[tid] = -1.0f;                 // 全元素写齐（隔离缺陷：只剩 barrier 误用）
        // 活着但永不到达 barrier：自旋条件读 shared（s[255]=255.0f 恒真），
        // 编译器无法常量折叠（V2 教训：局部 volatile 自旋被 nvcc 提升消掉）。
        while (s[255] != 42.0f) { }
    }
}

__global__ void staged_good(float* __restrict__ y) {
    __shared__ float s[256];
    const int tid = threadIdx.x;
    s[tid] = static_cast<float>(tid);
    __syncthreads();                    // 健康版：barrier 在全体路径上（无分支）
    y[tid] = (tid % 2 == 0) ? s[254 - tid] : -1.0f;
}

// ---------------------------------------------------------------------------

struct CaseResult {
    int mismatches = 0;      // good/bad 与 host 参考的最大偏差计数
    double max_abs_err = 0.0;
    int unique_outputs = 0;  // race bad 版 5 次运行 y[0..3] 拼接指纹 unique 计数
};

static int run_oob(bool bad, int n, int blocks, int block) {
    std::vector<float> hx(n), hy(n, -999.0f);
    fill_normal(hx, 20261101u);
    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dy, 0xEE, n * sizeof(float)));
    if (bad) scale_all_bad<<<blocks, block>>>(dx, dy, n);
    else     scale_all_good<<<blocks, block>>>(dx, dy, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    int mm = 0;
    for (int i = 0; i < n; ++i) {
        const double e = std::fabs(static_cast<double>(hy[i]) - 2.0 * static_cast<double>(hx[i]));
        if (e > max_err) max_err = e;
        if (e > 1e-3) ++mm;
    }
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    std::printf("[oob %s] n=%d mismatches=%d max_abs_err=%.3e\n", bad ? "BAD" : "GOOD", n, mm, max_err);
    return 0;
}

static int run_race(bool bad, int n, int blocks, int block) {
    std::vector<float> hx(n), hy_ref(blocks, 0.0f);
    fill_normal(hx, 20261102u);
    std::vector<float> hy(blocks);
    // host 参考：per-block 串行求和（同一 block 内元素顺序求和）
    for (int b = 0; b < blocks; ++b) {
        double acc = 0.0;
        for (int t = 0; t < block; ++t) {
            const int i = b * block + t;
            if (i < n) acc += static_cast<double>(hx[i]);
        }
        hy_ref[b] = static_cast<float>(acc);
    }
    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    double max_err_all = 0.0;
    std::set<std::string> fingerprints;
    for (int rep = 0; rep < 5; ++rep) {
        if (bad) block_sum_bad<<<blocks, block>>>(dx, dy, n);
        else     block_sum_good<<<blocks, block>>>(dx, dy, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hy.data(), dy, blocks * sizeof(float), cudaMemcpyDeviceToHost));
        std::string fp;
        for (int b = 0; b < blocks; ++b) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.6e,", static_cast<double>(hy[b]));
            fp += buf;
            const double e = std::fabs(static_cast<double>(hy[b]) - static_cast<double>(hy_ref[b]));
            if (e > max_err_all) max_err_all = e;
        }
        fingerprints.insert(fp);
    }
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    int mm = 0;
    for (int b = 0; b < blocks; ++b) {
        if (std::fabs(static_cast<double>(hy[b]) - static_cast<double>(hy_ref[b])) > 1e-2) ++mm;
    }
    std::printf("[race %s] blocks=%d mismatches_last_run=%d max_abs_err=%.3e unique_outputs_5rep=%d\n",
                bad ? "BAD" : "GOOD", blocks, mm, max_err_all, static_cast<int>(fingerprints.size()));
    return 0;
}

static int run_uninit(bool bad, int n, int blocks, int block) {
    std::vector<float> hx(n);
    fill_normal(hx, 20261103u);
    float *dx, *dy, *dmagic;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dmagic, sizeof(float)));
    if (!bad) CUDA_CHECK(cudaMemset(dmagic, 0, sizeof(float)));  // 健康版唯一差异：初始化
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    if (bad) mul_by_magic_bad<<<blocks, block>>>(dx, dmagic, dy, n);
    else     mul_by_magic_good<<<blocks, block>>>(dx, dmagic, dy, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hy(n);
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    int mm = 0;
    for (int i = 0; i < n; ++i) if (hy[i] != 0.0f) ++mm;  // magic=0 时 y 应全 0
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dmagic));
    std::printf("[uninit %s] n=%d nonzero_outputs=%d（GOOD 期望 0；BAD 取决于脏内存，不可依赖）\n",
                bad ? "BAD" : "GOOD", n, mm);
    return 0;
}

static int run_sync(bool bad, int blocks, int block) {
    float* dy;
    CUDA_CHECK(cudaMalloc(&dy, 256 * sizeof(float)));
    if (bad) staged_bad<<<blocks, block>>>(dy);
    else     staged_good<<<blocks, block>>>(dy);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hy(256);
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, 256 * sizeof(float), cudaMemcpyDeviceToHost));
    int mm = 0;
    for (int tid = 0; tid < 256; ++tid) {
        const float expect = (tid % 2 == 0) ? static_cast<float>(254 - tid) : -1.0f;
        if (std::fabs(hy[tid] - expect) > 1e-6) ++mm;
    }
    CUDA_CHECK(cudaFree(dy));
    std::printf("[sync %s] mismatches=%d（GOOD 期望 0）\n", bad ? "BAD" : "GOOD", mm);
    return 0;
}

int main(int argc, char** argv) {
    std::string kase, variant = "good";
    int n = 1024, blocks = 4, block = 256;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--case") kase = next();
        else if (a == "--variant") variant = next();
        else if (a == "--n") n = std::stoi(next());
        else if (a == "--blocks") blocks = std::stoi(next());
        else if (a == "--block") block = std::stoi(next());
        else { std::fprintf(stderr, "未知参数 %s\n", a.c_str()); return 2; }
    }
    const bool bad = (variant == "bad");
    if (kase == "oob")    return run_oob(bad, n, blocks, block);
    if (kase == "race")   return run_race(bad, n, blocks, block);
    if (kase == "uninit") return run_uninit(bad, n, blocks, block);
    if (kase == "sync")   return run_sync(bad, blocks, block);
    std::fprintf(stderr, "用法: %s --case oob|race|uninit|sync --variant good|bad [--n N --blocks B --block T]\n", argv[0]);
    return 2;
}
