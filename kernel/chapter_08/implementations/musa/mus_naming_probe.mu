// mus_naming_probe.cu —— mus* 前缀命名反证探针（预期编译失败）。
// 大纲与协议 §6 提示的 musMemcpy/musMallocHost 命名是否存在，以本文件能否
// 编译为准：编译失败 + 头文件 grep 双证据 => mus* 族不存在的实机结论。
// 本文件不需要能运行 —— 它的目标就是编译失败并留下错误全文。
#include <musa_runtime.h>

int main() {
    void* p = nullptr;
    musMemcpy(&p, &p, 4, musMemcpyHostToDevice);
    musMallocHost(&p, 4);
    musMemcpyAsync(&p, &p, 4, musMemcpyHostToDevice, nullptr);
    musStreamCreate((musStream_t*)nullptr);
    return 0;
}
