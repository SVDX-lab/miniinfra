"""不下载权重、不要求 CUDA 的第 10 期快速自检。"""

import torch

from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from static_decode import StaticDecodeRunner


def tiny_model():
    torch.manual_seed(7)
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
    model = Qwen3ForCausalLM(config, attention_backend="eager")
    model.eval()
    return model


def run_path(runner, prompts, steps):
    first = runner.prepare_prompts(prompts)
    generated = [[token] for token in first]
    for _ in range(steps):
        tokens = runner.step("static_eager").clone()
        for slot in range(len(prompts)):
            generated[slot].append(int(tokens[slot].item()))
    return generated, runner.last_logits.clone()


def main():
    model = tiny_model()
    runner = StaticDecodeRunner(
        model,
        capacity=3,
        context_bucket=16,
        block_size=4,
        pad_token_id=0,
    )
    prompts = [[4, 8, 15, 16, 23], [42, 7, 9]]
    output_a, logits_a = run_path(runner, prompts, steps=3)
    lengths_a = runner.sequence_lengths.clone()
    output_b, logits_b = run_path(runner, prompts, steps=3)

    assert output_a == output_b
    torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)
    assert lengths_a.tolist() == [8, 6, 0]
    assert runner.sequence_lengths.tolist() == [8, 6, 0]
    assert int(runner.output_tokens[2]) == 0
    assert runner.cache.blocks_per_slot == 4
    assert runner.cache.block_table.shape == (3, 4)
    assert len(set(runner.cache.block_table.flatten().tolist())) == 12

    dense = runner.cache.dense_slot_cache(0, 8)
    assert len(dense) == model.config.num_hidden_layers
    assert dense[0][0].shape == (1, 2, 8, 8)
    print("Smoke test 通过")
    print("静态 Buffer、Block Table、KV 写入、Slot Mask 和确定性检查通过")


if __name__ == "__main__":
    main()
