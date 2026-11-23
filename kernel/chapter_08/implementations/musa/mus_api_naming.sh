#!/usr/bin/env bash
# ============================================================================
# mus_api_naming.sh —— MUSA 传输 API 命名 grep 证据（本期最重要的迁移素材之一）。
#
# 大纲第 08 期标注 "musMemcpy/musMallocHost 对应迁移（API 命名待实机核对）"。
# 本脚本对 /usr/local/musa/include 逐符号 grep，mus* 与 musa* 两族全列，
# 输出计数表。静态证据；编译验证（build.sh 符号探测）为动态证据，两者互证。
# 输出被 build.sh tee 进 results/server-logs/musa-api-naming-grep.txt。
# ============================================================================
set -uo pipefail
INC="${MUSA_INC:-/usr/local/musa/include}"

echo "grep root: $INC ($(date -u +%FT%TZ))"
if [ ! -d "$INC" ]; then
    echo "ERROR: include 目录不存在"
    exit 1
fi

echo
echo "== musa* 前缀（预期全存在，第 02 期 M-06 结论的传输族扩展） =="
printf "%-38s %s\n" "symbol" "header_hits"
for name in \
    musaMemcpy musaMemcpyAsync musaMemcpy2DAsync \
    musaMalloc musaMallocHost musaFree musaFreeHost \
    musaHostAlloc musaHostRegister musaHostUnregister musaHostGetDevicePointer \
    musaSetDeviceFlags musaDeviceMapHost \
    musaStreamCreate musaStreamCreateWithFlags musaStreamCreateWithPriority \
    musaStreamDestroy musaStreamSynchronize musaStreamQuery musaStreamWaitEvent \
    musaEventCreate musaEventCreateWithFlags musaEventRecord musaEventSynchronize \
    musaEventElapsedTime musaEventDestroy \
    musaDeviceSynchronize musaDeviceGetPCIBusId musaGetDeviceProperties \
    musaGetDevice musaGetDeviceCount musaGetLastError musaGetErrorString \
    musaMemGetInfo musaMemPoolCreate \
    musaMemcpyHostToDevice musaMemcpyDeviceToHost musaMemcpyDeviceToDevice \
    musaHostAllocDefault musaHostAllocPortable musaHostAllocMapped \
    musaHostRegisterDefault musaStreamNonBlocking musaErrorNotReady \
    musaDeviceProp musaStream_t musaError_t musaEvent_t; do
    hits=$(grep -r "\b$name\b" "$INC" 2>/dev/null | wc -l)
    printf "%-38s %s\n" "$name" "$hits"
done

echo
echo "== mus* 前缀（大纲标注待核对项；预期全 0 —— M-06 结论的传输族扩展） =="
printf "%-38s %s\n" "symbol" "header_hits"
for name in \
    musMemcpy musMemcpyAsync musMemcpy2DAsync \
    musMalloc musMallocHost musFree musFreeHost \
    musHostAlloc musHostRegister musHostGetDevicePointer \
    musSetDeviceFlags musDeviceMapHost \
    musStreamCreate musStreamDestroy musStreamSynchronize musStreamWaitEvent \
    musEventCreate musEventRecord musEventElapsedTime \
    musDeviceSynchronize musGetDeviceProperties musStream_t musError_t; do
    hits=$(grep -r "\b$name\b" "$INC" 2>/dev/null | wc -l)
    printf "%-38s %s\n" "$name" "$hits"
done

echo
echo "== 关键声明所在头文件 =="
for name in musaMemcpyAsync musaMallocHost musaHostAlloc musaHostRegister \
            musaStreamCreate musaStreamWaitEvent musaEventElapsedTime musaDeviceGetPCIBusId; do
    f=$(grep -rl "\b$name\b" "$INC" 2>/dev/null | head -1)
    printf "%-30s %s\n" "$name" "${f#"$INC"/}"
done
echo
echo "（grep 为静态证据；符号能否链接/运行由 build.sh 编译探测 + api_probe 运行验证）"
