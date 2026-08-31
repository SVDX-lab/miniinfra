"""不下载真实权重、不要求 GPU 的第 08 期快速自检。"""

import torch

from engine import run_engine
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from scheduler import make_request_specs


def build_tiny_model():
    config = Qwen3Config(
        vocab_size=127,
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


def run_pair(model, sequences, outputs, block_size=4, capacity=8):
    specs = make_request_specs(sequences, outputs)
    common = dict(
        model=model,
        request_specs=specs,
        max_running_requests=1,
        eos_token_id=-1,
        device=torch.device("cpu"),
        token_budget=4,
        block_size=block_size,
        stop_on_eos=False,
        capture_logits=True,
        prefix_cache_capacity_blocks=capacity,
        model_namespace="tiny-prefix-cache-smoke",
    )
    return (
        run_engine(prefix_cache_enabled=False, **common),
        run_engine(prefix_cache_enabled=True, **common),
    )


def assert_same_outputs(disabled, enabled, tolerance=1e-4):
    assert disabled["new_token_ids"] == enabled["new_token_ids"]
    for request_id, baseline in disabled["first_token_logits"].items():
        error = torch.max(torch.abs(
            baseline - enabled["first_token_logits"][request_id]
        )).item()
        assert error < tolerance, (request_id, error)


def check_shared_prefix(model):
    common = [3, 5, 7, 11, 13, 17, 19, 23]
    sequences = [common + [29, 31, 37], common + [41, 43, 47]]
    disabled, enabled = run_pair(model, sequences, [3, 3])
    assert_same_outputs(disabled, enabled)
    hits = [row["prefix_hit_tokens"] for row in enabled["request_metrics"]]
    assert hits == [0, 8], hits
    assert enabled["metrics"]["executed_prompt_tokens"] == 14
    assert disabled["metrics"]["executed_prompt_tokens"] == 22
    final = enabled["final_cache_snapshot"]
    assert sum(final["block_ref_counts"]) == 0
    assert final["cached_blocks"] == 2
    return disabled, enabled


def check_context_sensitive_hash(model):
    same_second_block = [53, 59, 61, 67]
    sequences = [
        [2, 3, 5, 7] + same_second_block + [71],
        [11, 13, 17, 19] + same_second_block + [73],
    ]
    disabled, enabled = run_pair(model, sequences, [1, 1])
    assert_same_outputs(disabled, enabled)
    hits = [row["prefix_hit_tokens"] for row in enabled["request_metrics"]]
    assert hits == [0, 0], hits


def check_capacity_eviction(model):
    sequences = [
        [2, 3, 5, 7, 11],
        [13, 17, 19, 23, 29],
        [31, 37, 41, 43, 47],
    ]
    disabled, enabled = run_pair(model, sequences, [1, 1, 1], capacity=1)
    assert_same_outputs(disabled, enabled)
    assert enabled["metrics"]["prefix_eviction_count"] >= 2
    final = enabled["final_cache_snapshot"]
    assert sum(final["block_ref_counts"]) == 0
    assert final["cached_blocks"] == 1


def main():
    torch.manual_seed(8)
    model = build_tiny_model()
    disabled, enabled = check_shared_prefix(model)
    check_context_sensitive_hash(model)
    check_capacity_eviction(model)
    print("Smoke test 通过")
    print("链式哈希、完整块命中、不可变共享、引用计数和淘汰检查通过")
    print("disabled 与 enabled 逐请求输出一致")
    print(
        "执行 Prompt Token: disabled=%d, enabled=%d; 命中=%d"
        % (
            disabled["metrics"]["executed_prompt_tokens"],
            enabled["metrics"]["executed_prompt_tokens"],
            enabled["metrics"]["prefix_hit_tokens"],
        )
    )


if __name__ == "__main__":
    main()
