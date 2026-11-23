// ============================================================================
// rmsnorm_bench.cu —— 第 02 期独立可执行基准 + 正确性自检 + 文件 IO 通道。
//
// 用途：
//   1. ncu / nsys 的被测对象（独立进程，无 torch 干扰）；
//   2. MUSA 侧 torch 扩展路径不通时的退路：独立可执行程序 + 文件 IO
//      对比路径（--load/--out，输入输出统一 FP32 裸二进制，FP16 在边缘
//      转换），与本地 NumPy 真值对照；
//   3. Tile 扫描的第二测量口径（--tiles），与 torch 扩展口径互相印证。
//
// 计时口径（与第 01 期 experiment_utils.OperatorTimer 一致）：
//   warmup 次预热 -> 每次 repeat 用一对 Event 记录 -> 末尾一次 synchronize
//   -> 报告 P50/P95/mean/min（毫秒）。有效带宽 = 理论字节 / P50；
//   达成率 = 有效带宽 / --peak-gbs（cuda: 1008, musa: 768，大纲 §3.3 规格）。
//
// 数据：--load 读 FP32 裸二进制（x.bin: rows*h 个 float；w.bin: h 个 float），
//       或 --seed 用确定性 splitmix64 + Box-Muller 生成 N(0,1) 输入。
// 自检：--check 对前 min(rows,8) 行做 host 串行参考（与 CPU 真值同序），
//       报告 max/mean 绝对误差与余弦相似度。
// 输出：stdout 逐行 JSON（--json-file 可追加写文件），每行一个配置的完整结果。
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace mini_infra {

void launch_rmsnorm_fp32(const float* x, const float* w, float* y, int rows, int h,
                         float eps, int tile, cudaStream_t stream);
void launch_rmsnorm_fp16(const void* x, const void* w, void* y, int rows, int h,
                         float eps, int tile, cudaStream_t stream);

namespace bench {

#define BENCH_CUDA_CHECK(expr)                                                          \
    do {                                                                                \
        cudaError_t _err = (expr);                                                      \
        if (_err != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_err),  \
                         __FILE__, __LINE__);                                           \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

// 确定性输入生成（splitmix64 + Box-Muller），与 fixtures seed 规则独立但同样固定。
double splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return static_cast<double>(z ^ (z >> 31)) / 18446744073709551616.0;
}

void fill_normal(std::vector<float>& data, uint64_t seed) {
    uint64_t state = seed;
    for (size_t i = 0; i + 1 < data.size(); i += 2) {
        double u1 = splitmix64(state);
        double u2 = splitmix64(state);
        if (u1 < 1e-12) u1 = 1e-12;
        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;
        data[i] = static_cast<float>(r * std::cos(theta));
        data[i + 1] = static_cast<float>(r * std::sin(theta));
    }
    if (data.size() % 2 == 1) {
        double u1 = splitmix64(state);
        double u2 = splitmix64(state);
        if (u1 < 1e-12) u1 = 1e-12;
        data[data.size() - 1] = static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                                                   std::cos(2.0 * 3.14159265358979323846 * u2));
    }
}

// host 串行参考：与 CPU 真值（naive_rmsnorm_serial）同一 FP32 累加顺序。
void naive_reference(const std::vector<float>& x, const std::vector<float>& w, int rows, int h,
                     float eps, std::vector<float>& y) {
    for (int i = 0; i < rows; ++i) {
        const float* xr = &x[static_cast<size_t>(i) * h];
        float acc = 0.0f;
        for (int j = 0; j < h; ++j) acc += xr[j] * xr[j];
        const float scale = 1.0f / std::sqrt(acc / static_cast<float>(h) + eps);
        float* yr = &y[static_cast<size_t>(i) * h];
        for (int j = 0; j < h; ++j) yr[j] = (xr[j] * scale) * w[j];
    }
}

double percentile(std::vector<double>& sorted, double q) {
    const size_t n = sorted.size();
    const size_t idx = static_cast<size_t>(std::min(
        static_cast<double>(n - 1), std::floor(q * static_cast<double>(n))));
    return sorted[idx];
}

struct Options {
    int h = 1024;
    int rows = 2048;
    std::string dtype = "fp16";
    std::vector<int> tiles = {256};
    float eps = 1e-6f;
    int warmup = 10;
    int repeats = 100;
    uint64_t seed = 20260301;
    double peak_gbs = 1008.0;
    std::string load_x, load_w, out_y;
    bool check = true;
    int check_rows = 8;
    std::string json_file;
    int tile_for_out = 0;  // 文件 IO 输出时记录的 tile
};

void print_json_line(const std::string& device_name, const Options& opt, int tile, bool vec,
                     const std::vector<double>& samples_ms, long long theoretical_bytes,
                     const std::vector<float>& max_err, const std::vector<float>& mean_err,
                     const std::vector<float>& cos_sim) {
    std::vector<double> sorted(samples_ms);
    std::sort(sorted.begin(), sorted.end());
    double p50 = percentile(sorted, 0.50);
    double p95 = percentile(sorted, 0.95);
    double mean = 0.0;
    for (double v : samples_ms) mean += v;
    mean /= static_cast<double>(samples_ms.size());
    double effective = static_cast<double>(theoretical_bytes) / (p50 / 1000.0);
    double ratio = effective / (opt.peak_gbs * 1e9);

    std::string check_str = "null";
    if (!max_err.empty()) {
        // 逐 tile 自检：取最差行
        float worst_max = 0.0f, worst_mean = 0.0f, worst_cos = 1.0f;
        for (float v : max_err) worst_max = std::max(worst_max, v);
        for (float v : mean_err) worst_mean = std::max(worst_mean, v);
        for (float v : cos_sim) worst_cos = std::min(worst_cos, v);
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "{\"check_rows\":%d,\"max_abs_error\":%.3e,\"mean_abs_error\":%.3e,\"cosine_similarity\":%.8f}",
                      opt.check_rows, worst_max, worst_mean, worst_cos);
        check_str = buf;
    }
    char line[1024];
    std::snprintf(line, sizeof(line),
                  "{\"device\":\"%s\",\"h\":%d,\"rows\":%d,\"dtype\":\"%s\",\"tile\":%d,\"vec\":%s,"
                  "\"warmup\":%d,\"repeats\":%zu,\"p50_ms\":%.4f,\"p95_ms\":%.4f,\"mean_ms\":%.4f,"
                  "\"min_ms\":%.4f,\"theoretical_bytes\":%lld,"
                  "\"effective_bandwidth_gb_s\":%.2f,\"peak_bandwidth_gb_s_spec\":%.1f,"
                  "\"achievement_ratio\":%.4f,\"check\":%s}",
                  device_name.c_str(), opt.h, opt.rows, opt.dtype.c_str(), tile, vec ? "true" : "false",
                  opt.warmup, samples_ms.size(), p50, p95, mean, sorted.front(),
                  theoretical_bytes, effective / 1e9, opt.peak_gbs, ratio, check_str.c_str());
    std::printf("%s\n", line);
    std::fflush(stdout);
    if (!opt.json_file.empty()) {
        std::ofstream file(opt.json_file, std::ios::app);
        file << line << "\n";
    }
}

int run(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--h") opt.h = std::stoi(next());
        else if (a == "--rows") opt.rows = std::stoi(next());
        else if (a == "--dtype") opt.dtype = next();
        else if (a == "--tiles") {
            std::string list = next();
            opt.tiles.clear();
            size_t pos = 0;
            while (pos < list.size()) {
                size_t comma = list.find(',', pos);
                if (comma == std::string::npos) comma = list.size();
                opt.tiles.push_back(std::stoi(list.substr(pos, comma - pos)));
                pos = comma + 1;
            }
        }
        else if (a == "--eps") opt.eps = std::stof(next());
        else if (a == "--warmup") opt.warmup = std::stoi(next());
        else if (a == "--repeats") opt.repeats = std::stoi(next());
        else if (a == "--seed") opt.seed = std::stoull(next());
        else if (a == "--peak-gbs") opt.peak_gbs = std::stod(next());
        else if (a == "--load") { opt.load_x = next(); opt.load_w = next(); }
        else if (a == "--out") opt.out_y = next();
        else if (a == "--check") { opt.check = (next() != "false"); }
        else if (a == "--check-rows") opt.check_rows = std::stoi(next());
        else if (a == "--json-file") opt.json_file = next();
        else if (a == "--device-name") { /* 占位：实际设备名由程序自取 */ }
        else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            return 2;
        }
    }
    if (opt.h <= 0 || opt.rows < 0 || opt.tiles.empty()) {
        std::fprintf(stderr, "非法参数：h>0、rows>=0、tiles 非空\n");
        return 2;
    }
    if (opt.dtype != "fp32" && opt.dtype != "fp16") {
        std::fprintf(stderr, "dtype 只支持 fp32 / fp16\n");
        return 2;
    }

    int device = 0;
    BENCH_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    BENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::string device_name = prop.name;

    const bool is_fp16 = (opt.dtype == "fp16");
    const size_t itemsize = is_fp16 ? 2 : 4;
    const long long theoretical_bytes =
        (static_cast<long long>(opt.rows) * opt.h * 2 + opt.h) * static_cast<long long>(itemsize);

    // ---- 输入数据（host FP32 形态）----
    std::vector<float> x_host(static_cast<size_t>(opt.rows) * opt.h);
    std::vector<float> w_host(opt.h);
    if (!opt.load_x.empty()) {
        std::ifstream fx(opt.load_x, std::ios::binary);
        std::ifstream fw(opt.load_w, std::ios::binary);
        if (!fx || !fw) { std::fprintf(stderr, "打不开输入文件\n"); return 2; }
        fx.read(reinterpret_cast<char*>(x_host.data()),
                static_cast<std::streamsize>(x_host.size() * sizeof(float)));
        fw.read(reinterpret_cast<char*>(w_host.data()),
                static_cast<std::streamsize>(w_host.size() * sizeof(float)));
        if (!fx || !fw) { std::fprintf(stderr, "输入文件长度不足\n"); return 2; }
    } else {
        fill_normal(x_host, opt.seed);
        fill_normal(w_host, opt.seed + 7);
    }

    // ---- 设备内存与上传 ----
    void *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
    size_t x_bytes = x_host.size() * itemsize;
    size_t w_bytes = w_host.size() * itemsize;
    BENCH_CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    BENCH_CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
    BENCH_CUDA_CHECK(cudaMalloc(&d_y, x_bytes));
    if (is_fp16) {
        // FP16：host 转 half 再上传（文件 IO 统一 FP32 形态，转换在边缘完成）。
        std::vector<__half> xh(x_host.size()), wh(w_host.size());
        for (size_t i = 0; i < x_host.size(); ++i) xh[i] = __float2half(x_host[i]);
        for (size_t i = 0; i < w_host.size(); ++i) wh[i] = __float2half(w_host[i]);
        BENCH_CUDA_CHECK(cudaMemcpy(d_x, xh.data(), x_bytes, cudaMemcpyHostToDevice));
        BENCH_CUDA_CHECK(cudaMemcpy(d_w, wh.data(), w_bytes, cudaMemcpyHostToDevice));
    } else {
        BENCH_CUDA_CHECK(cudaMemcpy(d_x, x_host.data(), x_bytes, cudaMemcpyHostToDevice));
        BENCH_CUDA_CHECK(cudaMemcpy(d_w, w_host.data(), w_bytes, cudaMemcpyHostToDevice));
    }

    // ---- host 参考（自检用）----
    std::vector<float> y_ref;
    int check_rows = opt.check ? std::min(opt.rows, opt.check_rows) : 0;
    if (check_rows > 0) {
        y_ref.assign(static_cast<size_t>(check_rows) * opt.h, 0.0f);
        naive_reference(x_host, w_host, check_rows, opt.h, opt.eps, y_ref);
    }

    // ---- 计时：每 tile 一组（Event 对 + 末尾一次同步，同 OperatorTimer 口径）----
    for (int tile : opt.tiles) {
        std::vector<double> samples_ms;
        std::vector<cudaEvent_t> starts(opt.repeats), stops(opt.repeats);
        for (int i = 0; i < opt.repeats; ++i) {
            BENCH_CUDA_CHECK(cudaEventCreate(&starts[i]));
            BENCH_CUDA_CHECK(cudaEventCreate(&stops[i]));
        }
        auto launch_once = [&]() {
            if (is_fp16) {
                launch_rmsnorm_fp16(d_x, d_w, d_y, opt.rows, opt.h, opt.eps, tile, 0);
            } else {
                launch_rmsnorm_fp32(static_cast<const float*>(d_x), static_cast<const float*>(d_w),
                                    static_cast<float*>(d_y), opt.rows, opt.h, opt.eps, tile, 0);
            }
        };
        for (int i = 0; i < opt.warmup; ++i) launch_once();
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < opt.repeats; ++i) {
            BENCH_CUDA_CHECK(cudaEventRecord(starts[i]));
            launch_once();
            BENCH_CUDA_CHECK(cudaEventRecord(stops[i]));
        }
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < opt.repeats; ++i) {
            float ms = 0.0f;
            BENCH_CUDA_CHECK(cudaEventElapsedTime(&ms, starts[i], stops[i]));
            samples_ms.push_back(static_cast<double>(ms));
            BENCH_CUDA_CHECK(cudaEventDestroy(starts[i]));
            BENCH_CUDA_CHECK(cudaEventDestroy(stops[i]));
        }

        // ---- 自检：D2H 拉回前 check_rows 行，与 host 参考对照 ----
        std::vector<float> max_err, mean_err, cos_sim;
        if (check_rows > 0) {
            std::vector<float> y_host(static_cast<size_t>(check_rows) * opt.h);
            if (is_fp16) {
                std::vector<__half> yh(static_cast<size_t>(check_rows) * opt.h);
                BENCH_CUDA_CHECK(cudaMemcpy(yh.data(), d_y, yh.size() * sizeof(__half),
                                            cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < y_host.size(); ++i) y_host[i] = __half2float(yh[i]);
            } else {
                BENCH_CUDA_CHECK(cudaMemcpy(y_host.data(), d_y, y_host.size() * sizeof(float),
                                            cudaMemcpyDeviceToHost));
            }
            for (int r = 0; r < check_rows; ++r) {
                const size_t off = static_cast<size_t>(r) * opt.h;
                double local_max = 0.0, local_sum = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
                for (int j = 0; j < opt.h; ++j) {
                    const double a = y_host[off + j];
                    const double b = y_ref[off + j];
                    const double e = std::fabs(a - b);
                    local_max = std::max(local_max, e);
                    local_sum += e;
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                max_err.push_back(static_cast<float>(local_max));
                mean_err.push_back(static_cast<float>(local_sum / opt.h));
                cos_sim.push_back(static_cast<float>(
                    dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30)));
            }
        }
        const bool vec = (opt.h % (is_fp16 ? 8 : 4) == 0);
        print_json_line(device_name, opt, tile, vec, samples_ms, theoretical_bytes,
                        max_err, mean_err, cos_sim);
    }

    // ---- 文件 IO 输出（独立可执行程序对比路径；输出统一 FP32 形态）----
    if (!opt.out_y.empty()) {
        std::vector<float> y_full(static_cast<size_t>(opt.rows) * opt.h);
        if (is_fp16) {
            std::vector<__half> yh(y_full.size());
            BENCH_CUDA_CHECK(cudaMemcpy(yh.data(), d_y, yh.size() * sizeof(__half),
                                        cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < y_full.size(); ++i) y_full[i] = __half2float(yh[i]);
        } else {
            BENCH_CUDA_CHECK(cudaMemcpy(y_full.data(), d_y, y_full.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost));
        }
        std::ofstream fy(opt.out_y, std::ios::binary);
        fy.write(reinterpret_cast<const char*>(y_full.data()),
                 static_cast<std::streamsize>(y_full.size() * sizeof(float)));
        std::printf("[bench] 已写出 %s（%zu float）\n", opt.out_y.c_str(), y_full.size());
    }

    BENCH_CUDA_CHECK(cudaFree(d_x));
    BENCH_CUDA_CHECK(cudaFree(d_w));
    BENCH_CUDA_CHECK(cudaFree(d_y));
    return 0;
}

}  // namespace bench
}  // namespace mini_infra

int main(int argc, char** argv) { return mini_infra::bench::run(argc, argv); }
