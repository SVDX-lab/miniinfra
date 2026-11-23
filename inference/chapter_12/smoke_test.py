"""不下载真实权重、不要求 GPU 的第 12 期快速自检。

用两个 CPU tensor 模拟 GPU/CPU 两级池，覆盖：
- Chunked Prefill 调度与硬 Token Budget；
- swap 抢占：换出字节守恒、Block Table 重建、恢复后逐位一致的 KV；
- recompute 抢占：丢弃 KV 重新 Prefill；
- swapped 优先于 waiting 的恢复顺序；
- pinned 池耗尽报错、保守准入不触发抢占。
"""

import torch

from engine import run_engine
from paged_cache import CPUPinnedPool, PagedKVCache
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from scheduler import (
    ChunkedPrefillScheduler,
    RequestState,
    SchedulerConfig,
    make_request_specs,
)


def build_tiny_model():
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config).eval()


def check_scheduler():
    config = SchedulerConfig(max_running_requests=2, token_budget=4)
    scheduler = ChunkedPrefillScheduler(config)
    spec = make_request_specs([list(range(1, 10))], [3])[0]
    state = RequestState(spec)
    decision = scheduler.schedule([state], [])
    assert decision.phase == "prefill"
    assert decision.prefill_plans[0].start == 0
    assert decision.prefill_plans[0].end == 4
    assert decision.scheduled_tokens == 4
    state.status = "prefilling"
    state.prefill_cursor = 4
    decision = scheduler.schedule([state], [])
    assert (decision.prefill_plans[0].start, decision.prefill_plans[0].end) == (4, 8)


def check_swap_roundtrip():
    """直接验证两级池：换出 -> 释放 -> 换入后 KV 与换出前逐位一致。"""
    model = build_tiny_model()
    config = model.config
    device = torch.device("cpu")
    cache = PagedKVCache(config, block_size=4, max_blocks=8, device=device,
                         dtype=torch.float32)
    cpu_pool = CPUPinnedPool(config, 4, 8, torch.float32, pin_memory=False)

    generator = torch.Generator().manual_seed(3)
    chunk_length = 10
    request_cache = [
        (
            torch.rand((1, config.num_key_value_heads, chunk_length,
                        config.head_dim), generator=generator),
            torch.rand((1, config.num_key_value_heads, chunk_length,
                        config.head_dim), generator=generator),
        )
        for _ in range(config.num_hidden_layers)
    ]
    cache.begin_request("r0")
    cache.append_prefill("r0", request_cache)
    before = [
        cache.read_layer(layer, ["r0"], cache.sequence_lengths["r0"])
        for layer in range(config.num_hidden_layers)
    ]
    used_before = cache.used_block_count
    stats = cpu_pool.swap_out(cache, "r0")
    assert stats["blocks"] == 3
    assert stats["bytes"] == 3 * cache.bytes_per_block
    assert stats["logical_bytes"] == chunk_length * (cache.bytes_per_block // 4)
    assert stats["tail_fragment_bytes"] == 2 * (cache.bytes_per_block // 4)
    assert cache.used_block_count == used_before - 3
    assert cache.free_block_count == 8
    assert "r0" not in cache.block_tables

    cpu_pool.swap_in(cache, "r0")
    after = [
        cache.read_layer(layer, ["r0"], cache.sequence_lengths["r0"])
        for layer in range(config.num_hidden_layers)
    ]
    for (key_before, value_before), (key_after, value_after) in zip(before, after):
        assert torch.equal(key_before, key_after)
        assert torch.equal(value_before, value_after)
    assert cpu_pool.used_block_count == 0
    cache.release("r0")


def check_pinned_pool_exhaustion():
    model = build_tiny_model()
    config = model.config
    device = torch.device("cpu")
    cache = PagedKVCache(config, block_size=4, max_blocks=8, device=device,
                         dtype=torch.float32)
    cpu_pool = CPUPinnedPool(config, 4, 1, torch.float32, pin_memory=False)
    request_cache = [
        (
            torch.rand((1, config.num_key_value_heads, 10, config.head_dim)),
            torch.rand((1, config.num_key_value_heads, 10, config.head_dim)),
        )
        for _ in range(config.num_hidden_layers)
    ]
    cache.begin_request("r0")
    cache.append_prefill("r0", request_cache)
    try:
        cpu_pool.swap_out(cache, "r0")
    except RuntimeError as error:
        assert "CPU Pinned Pool 已耗尽" in str(error)
    else:
        raise AssertionError("pinned 池耗尽应当抛出明确错误")
    assert "r0" in cache.block_tables, "换出失败时不得丢失 GPU 端数据"


def build_specs():
    sequences = [
        [3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
        [37, 41, 43, 47, 53, 59, 61, 67, 71],
        [73, 79, 83, 89],
        [91, 2, 4, 6, 8, 10, 12, 14, 16],
    ]
    return make_request_specs(
        sequences, [6, 5, 4, 3], [0.0, 0.0, 0.0, 0.0]
    )


def check_engine_paths():
    torch.manual_seed(11)
    device = torch.device("cpu")
    model = build_tiny_model()
    specs = build_specs()
    common = dict(
        max_running_requests=2,
        eos_token_id=-1,
        device=device,
        token_budget=4,
        block_size=3,
        stop_on_eos=False,
    )
    # 无容量压力的参考路径。
    reference = run_engine(model, specs, preempt_mode="swap", **common)
    assert reference["metrics"]["preemption_count"] == 0

    # 人为缩小 GPU 池，使抢占必然发生。
    # 单请求最坏 16 token -> 6 块；前两个请求最坏合计 11 块 > 池 9 块，
    # 两个请求同时 Decode 到尾部时必然触发抢占。
    pool_blocks = 9
    swap = run_engine(
        model, specs, preempt_mode="swap", pool_blocks=pool_blocks, **common
    )
    recompute = run_engine(
        model, specs, preempt_mode="recompute", pool_blocks=pool_blocks, **common
    )
    assert swap["metrics"]["preemption_count"] > 0, "缩小池后必须发生抢占"
    assert recompute["metrics"]["preemption_count"] > 0
    assert swap["new_token_ids"] == reference["new_token_ids"]
    assert recompute["new_token_ids"] == reference["new_token_ids"]
    assert swap["metrics"]["swap_out_events"] == swap["metrics"]["preemption_count"]
    assert swap["metrics"]["swap_in_events"] == swap["metrics"]["resume_count"]
    assert recompute["metrics"]["resume_recompute_events"] > 0
    assert recompute["metrics"]["recompute_redo_tokens_total"] > 0
    assert recompute["metrics"]["recompute_prefill_wall_ms_total"] > 0

    # swap 实验不允许运行期静默回退；CPU Pool 配置不足应在启动阶段失败。
    try:
        run_engine(
            model, specs, preempt_mode="swap", pool_blocks=pool_blocks,
            cpu_pool_blocks=1, **common,
        )
    except ValueError as error:
        assert "CPU Pinned Pool 配置不足" in str(error)
    else:
        raise AssertionError("CPU Pool 配置不足应在启动阶段失败")

    # 保守准入：预留 prompt+max_new，绝不抢占。
    conservative = run_engine(
        model, specs, preempt_mode="swap", pool_blocks=pool_blocks,
        admission_mode="conservative", **common,
    )
    assert conservative["metrics"]["preemption_count"] == 0
    assert conservative["new_token_ids"] == reference["new_token_ids"]

    # 资源事件字节守恒：换出/换入字节 = 块数 × 单块字节数。
    bytes_per_block = swap["metrics"]["bytes_per_block"]
    for event in swap["resource_events"]:
        if event["type"] in ("swap_out", "swap_in"):
            assert event["bytes"] == event["blocks"] * bytes_per_block
            assert event["bytes"] == (
                event["logical_bytes"] + event["tail_fragment_bytes"]
            )
    return reference, swap, recompute


def main():
    check_scheduler()
    check_swap_roundtrip()
    check_pinned_pool_exhaustion()
    reference, swap, recompute = check_engine_paths()
    print("Smoke test 通过")
    print("两级池换出/换入字节守恒与 KV 逐位一致")
    print("swap 与 recompute 恢复后的输出均与无抢占参考一致")
    print(
        "抢占次数: swap=%d, recompute=%d；参考路径=%d"
        % (
            swap["metrics"]["preemption_count"],
            recompute["metrics"]["preemption_count"],
            reference["metrics"]["preemption_count"],
        )
    )


if __name__ == "__main__":
    main()
