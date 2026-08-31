"""使用一个很小的随机模型快速检查核心代码，不需要下载真实权重。"""

import torch

from qwen3_model import Qwen3Config, Qwen3ForCausalLM


def main():
    torch.manual_seed(0)

    # 缩小所有维度，使测试可以在 CPU 上几秒内完成。
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

    first_input = torch.tensor([[1, 2, 3]])
    second_input = torch.tensor([[1, 2, 4]])

    with torch.inference_mode():
        first_logits = model(first_input)
        second_logits = model(second_input)

    assert first_logits.shape == (1, 3, 32)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    # 两个输入只有最后一个 Token 不同。若 causal mask 正确，前两个位置
    # 的输出不应该受最后一个 Token 影响。
    assert torch.allclose(
        first_logits[:, :2, :], second_logits[:, :2, :], atol=1e-6
    )
    print("smoke test passed")


if __name__ == "__main__":
    main()

