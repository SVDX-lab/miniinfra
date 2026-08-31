"""不下载权重、不要求 GPU 的第 13 期快速自检。"""

import torch

from engine import run_engine
from paged_cache import CPUPinnedPool, PagedKVCache
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from scheduler import make_request_specs
from transfer import KVTransferManager


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


def check_deferred_ownership():
    model = build_tiny_model()
    device = torch.device("cpu")
    cache = PagedKVCache(
        model.config, block_size=4, max_blocks=8, device=device,
        dtype=torch.float32,
    )
    cpu_pool = CPUPinnedPool(
        model.config, 4, 8, torch.float32, pin_memory=False
    )
    request_cache = [
        (
            torch.rand((1, model.config.num_key_value_heads, 10, 8)),
            torch.rand((1, model.config.num_key_value_heads, 10, 8)),
        )
        for _ in range(model.config.num_hidden_layers)
    ]
    cache.begin_request("r0")
    cache.append_prefill("r0", request_cache)
    before = [cache.read_layer(layer, ["r0"], 10) for layer in range(2)]
    manager = KVTransferManager(device, "async")

    out_task = manager.submit_swap_out(cache, cpu_pool, "r0", 0.0)
    assert "r0" in cache.block_tables, "完成回调前不能释放 GPU 源 Block"
    manager.finish(out_task, cache, cpu_pool, 0.0)
    assert "r0" not in cache.block_tables

    in_task = manager.submit_swap_in(cache, cpu_pool, "r0", 0.0)
    assert "r0" in cache.block_tables, "H2D 提交时必须预留 GPU 目标 Block"
    assert cpu_pool.used_block_count == 3, "H2D 完成前不能释放 CPU 源 Block"
    manager.finish(in_task, cache, cpu_pool, 0.0)
    after = [cache.read_layer(layer, ["r0"], 10) for layer in range(2)]
    for left, right in zip(before, after):
        assert torch.equal(left[0], right[0])
        assert torch.equal(left[1], right[1])
    assert cpu_pool.used_block_count == 0
    cache.release("r0")


def check_engine_modes():
    torch.manual_seed(11)
    model = build_tiny_model()
    specs = make_request_specs(
        [
            [3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
            [37, 41, 43, 47, 53, 59, 61, 67, 71],
            [73, 79, 83, 89],
            [91, 2, 4, 6, 8, 10, 12, 14, 16],
        ],
        [6, 5, 4, 3],
    )
    common = dict(
        max_running_requests=2,
        eos_token_id=-1,
        device=torch.device("cpu"),
        token_budget=4,
        block_size=3,
        stop_on_eos=False,
    )
    reference = run_engine(model, specs, transfer_mode="sync", **common)
    sync = run_engine(
        model, specs, transfer_mode="sync", pool_blocks=9, **common
    )
    asynchronous = run_engine(
        model, specs, transfer_mode="async", pool_blocks=9, **common
    )
    assert sync["metrics"]["preemption_count"] > 0
    assert asynchronous["metrics"]["preemption_count"] > 0
    assert sync["new_token_ids"] == reference["new_token_ids"]
    assert asynchronous["new_token_ids"] == reference["new_token_ids"]
    assert sync["new_token_ids"] == asynchronous["new_token_ids"]
    for result, mode in ((sync, "sync"), (asynchronous, "async")):
        assert result["metrics"]["transfer_mode"] == mode
        assert result["metrics"]["swap_out_events"] == result["metrics"][
            "preemption_count"
        ]
        assert result["metrics"]["swap_in_events"] == result["metrics"][
            "resume_count"
        ]
        for event in result["resource_events"]:
            if event["type"] in ("swap_out", "swap_in"):
                assert event["bytes"] == (
                    event["logical_bytes"] + event["tail_fragment_bytes"]
                )
    return reference, sync, asynchronous


def main():
    check_deferred_ownership()
    reference, sync, asynchronous = check_engine_modes()
    print("Smoke test 通过")
    print("异步传输完成前的 GPU/CPU Block 所有权正确")
    print("sync/async 输出均与无抢占参考一致")
    print(
        "抢占次数: sync=%d async=%d reference=%d"
        % (
            sync["metrics"]["preemption_count"],
            asynchronous["metrics"]["preemption_count"],
            reference["metrics"]["preemption_count"],
        )
    )


if __name__ == "__main__":
    main()
