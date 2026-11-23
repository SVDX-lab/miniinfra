/* ===========================================================================
 * race_counter_bad.c —— 第 10 期 CPU 列：含数据竞争的 pthread 计数器。
 *
 * 角色：ThreadSanitizer（-fsanitize=thread）的思想参照——
 *   CPU 上"最简单的竞态"= 多线程对同一变量做非原子 read-modify-write
 *   （counter++ 实际是 load/add/store 三步，线程间交错丢更新）。
 *   症状（无 TSAN 时）：程序正常退出，最终值 < 期望值（丢更新），无崩溃；
 *   TSAN 下：首次竞态访存对被即时报告（data race），报错行即本文件两处。
 *
 * 这正是 GPU racecheck 检测的同一类错误的 CPU 原型：访问无同步约束、
 * 结果依赖调度顺序。修复版见 race_counter_fixed.c（互斥锁）。
 * =========================================================================== */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define N_THREADS 8
#define N_INCR 1000000

static long long g_counter = 0;  /* 被竞态共享的计数器 */

static void* worker(void* arg) {
    (void)arg;
    for (int i = 0; i < N_INCR; ++i) {
        g_counter = g_counter + 1;  /* 非原子 RMW：TSAN 报 data race 的写点 */
    }
    return NULL;
}

int main(void) {
    pthread_t tid[N_THREADS];
    for (int t = 0; t < N_THREADS; ++t) {
        if (pthread_create(&tid[t], NULL, worker, NULL) != 0) {
            perror("pthread_create");
            return 1;
        }
    }
    for (int t = 0; t < N_THREADS; ++t) pthread_join(tid[t], NULL);

    const long long expected = (long long)N_THREADS * N_INCR;
    printf("final=%lld expected=%lld lost=%lld (%.3f%%)\n",
           g_counter, expected, expected - g_counter,
           100.0 * (double)(expected - g_counter) / (double)expected);
    return (g_counter == expected) ? 0 : 2;  /* 退出码 2 = 结果损坏证据 */
}
