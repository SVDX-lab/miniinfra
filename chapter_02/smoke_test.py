"""使用随机小模型验证 no-cache 与 KV Cache 路径，不下载真实权重。"""

import torch

from qwen3_model import Qwen3Config, Qwen3ForCausalLM, cache_size_bytes


def main():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )
    config.validate()
    model = Qwen3ForCausalLM(config).eval()
    prompt = torch.tensor([[1, 2, 3]])

    with torch.inference_mode():
        baseline_logits, no_cache = model(prompt, use_cache=False)
        prefill_logits, past_key_values = model(prompt, use_cache=True)

    assert no_cache is None
    assert torch.allclose(baseline_logits, prefill_logits, atol=1e-6)
    assert len(past_key_values) == config.num_hidden_layers
    expected_shape = (1, config.num_key_value_heads, 3, config.head_dim)
    for key, value in past_key_values:
        assert tuple(key.shape) == expected_shape
        assert tuple(value.shape) == expected_shape

    expected_bytes = (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * config.head_dim
        * 3
        * torch.tensor([], dtype=torch.float32).element_size()
    )
    assert cache_size_bytes(past_key_values) == expected_bytes

    next_input = torch.tensor([[4]])
    full_context = torch.cat((prompt, next_input), dim=1)
    with torch.inference_mode():
        full_logits, _ = model(full_context, use_cache=False)
        cached_logits, updated_cache = model(
            next_input,
            past_key_values=past_key_values,
            use_cache=True,
        )

    assert torch.allclose(full_logits[:, -1, :], cached_logits[:, -1, :], atol=1e-5)
    assert updated_cache[0][0].shape[2] == 4

    try:
        model(next_input, past_key_values=past_key_values, use_cache=False)
    except ValueError:
        pass
    else:
        raise AssertionError("传入 Cache 却关闭 use_cache 时应拒绝运行")

    print("smoke test passed")


if __name__ == "__main__":
    main()
