/* ===========================================================================
 * race_counter_fixed.c —— 第 10 期 CPU 列：互斥锁修复版计数器。
 *
 * 与 race_counter_bad.c 唯一差异：counter++ 的 load/add/store 三步被
 * pthread_mutex 包裹为临界区（TSAN 把 lock/unlock 记为 happens-before 边，
 * 竞态消失）。作为 racecheck 语境里"加同步原语后报告清零"的 CPU 对照。
 * =========================================================================== */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define N_THREADS 8
#define N_INCR 1000000

static long long g_counter = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void* worker(void* arg) {
    (void)arg;
    for (int i = 0; i < N_INCR; ++i) {
        pthread_mutex_lock(&g_lock);      /* happens-before 边的建立点 */
        g_counter = g_counter + 1;
        pthread_mutex_unlock(&g_lock);
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
    printf("final=%lld expected=%lld lost=%lld\n",
           g_counter, expected, expected - g_counter);
    return (g_counter == expected) ? 0 : 2;
}
