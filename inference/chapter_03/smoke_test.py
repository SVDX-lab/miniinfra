"""不下载权重、不要求 GPU 的固定批处理快速自检。"""

import torch

from batch_generation import generate_fixed_batch, left_pad_sequences
from qwen3_model import Qwen3Config, Qwen3ForCausalLM, cache_size_bytes


def assert_close(left, right, message, atol=1e-5):
    error = (left - right).abs().max().item()
    if error > atol:
        raise AssertionError("%s，最大误差 %.8f" % (message, error))


def main():
    torch.manual_seed(7)
    device = torch.device("cpu")
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
    model = Qwen3ForCausalLM(config).eval()
    request_a = [3, 5, 7, 9]
    request_b = [11, 13]
    request_c = [17, 19]

    padded, mask, positions, lengths = left_pad_sequences(
        [request_a, request_b], pad_token_id=0, device=device
    )
    assert padded.tolist() == [[3, 5, 7, 9], [0, 0, 11, 13]]
    assert mask.tolist() == [[True] * 4, [False, False, True, True]]
    assert positions.tolist() == [[0, 1, 2, 3], [0, 0, 0, 1]]
    assert lengths.tolist() == [4, 2]

    single = generate_fixed_batch(
        model, [request_a], 4, eos_token_id=-1, device=device,
        stop_on_eos=False, capture_logits=True,
    )
    paired = generate_fixed_batch(
        model, [request_b, request_a], [4, 4], eos_token_id=-1,
        device=device, stop_on_eos=False, capture_logits=True,
    )
    replaced = generate_fixed_batch(
        model, [request_c, request_a], [4, 4], eos_token_id=-1,
        device=device, stop_on_eos=False, capture_logits=True,
    )

    assert single["new_token_ids"][0] == paired["new_token_ids"][1]
    assert paired["new_token_ids"][1] == replaced["new_token_ids"][1]
    for step in range(4):
        assert_close(
            single["logits_trace"][step][0],
            paired["logits_trace"][step][1],
            "单请求与 Batch 路径不一致",
        )
        assert_close(
            paired["logits_trace"][step][1],
            replaced["logits_trace"][step][1],
            "同批其他请求内容影响了目标请求",
        )

    slots = generate_fixed_batch(
        model,
        [request_a, request_b],
        [2, 4],
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
    )
    metrics = slots["metrics"]
    assert metrics["active_counts"] == [2, 2, 1, 1]
    assert metrics["completion_steps"] == [2, 4]
    assert abs(metrics["slot_utilization"] - 0.75) < 1e-12
    assert metrics["cache_length"] == 7  # max prompt 4 + 3 次 Decode 输入
    expected_cache_bytes = (
        2 * config.num_hidden_layers * 2 * config.num_key_value_heads
        * config.head_dim * metrics["cache_length"] * 4
    )
    assert metrics["cache_bytes"] == expected_cache_bytes

    # 非法 Cache batch 维和错误长度的 Attention Mask 必须被拒绝。
    prefill_ids = torch.tensor([request_a], dtype=torch.long)
    _, cache = model(prefill_ids, use_cache=True)
    assert cache_size_bytes(cache) > 0
    try:
        model(
            prefill_ids[:, -1:],
            attention_mask=torch.ones((1, 1), dtype=torch.bool),
            past_key_values=cache,
            use_cache=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("错误长度的 Attention Mask 未被拒绝")

    print("Smoke test 通过")
    print("左 Padding、Position IDs、请求隔离、排列不变性检查通过")
    print("Batched KV Cache、完成槽位和利用率检查通过")


if __name__ == "__main__":
    main()
