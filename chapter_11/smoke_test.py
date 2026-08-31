"""不下载权重、不要求 CUDA 的第 11 期快速自检。"""

import copy

import torch

from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from speculative_decode import (
    longest_matching_prefix,
    speculative_greedy_generate,
    target_greedy_generate,
    truncate_at_eos,
)


def tiny_model(seed):
    torch.manual_seed(seed)
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
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def main():
    assert longest_matching_prefix([1, 2, 3], [9, 2, 3]) == 0
    assert longest_matching_prefix([1, 2, 3], [1, 9, 3]) == 1
    assert longest_matching_prefix([1, 2, 3], [1, 2, 3]) == 3
    assert truncate_at_eos([4, 5, 6], 5) == ([4, 5], True)
    assert truncate_at_eos([4, 5, 6], None) == ([4, 5, 6], False)

    prompt = [4, 8, 15, 16, 23]
    target = tiny_model(seed=7)
    different_draft = tiny_model(seed=19)
    baseline = target_greedy_generate(target, prompt, max_new_tokens=13)
    speculative = speculative_greedy_generate(
        target,
        different_draft,
        prompt,
        max_new_tokens=13,
        draft_length=4,
    )
    assert speculative.token_ids == baseline.token_ids
    assert speculative.target_cache_length == len(prompt) + 12
    assert speculative.draft_cache_length == speculative.target_cache_length
    assert speculative.stats.target_decode_calls < baseline.stats.target_decode_calls
    assert speculative.stats.proposed_tokens >= speculative.stats.accepted_tokens

    target_same = tiny_model(seed=31)
    identical_draft = copy.deepcopy(target_same)
    same_baseline = target_greedy_generate(
        target_same, prompt, max_new_tokens=13
    )
    same_speculative = speculative_greedy_generate(
        target_same,
        identical_draft,
        prompt,
        max_new_tokens=13,
        draft_length=4,
    )
    assert same_speculative.token_ids == same_baseline.token_ids
    assert same_speculative.stats.acceptance_rate == 1.0
    assert same_speculative.stats.target_decode_calls == 3

    eos_target = tiny_model(seed=41)
    eos_draft = tiny_model(seed=43)
    eos_id = target_greedy_generate(
        eos_target, prompt, max_new_tokens=1
    ).token_ids[0]
    eos_baseline = target_greedy_generate(
        eos_target, prompt, max_new_tokens=8, eos_token_id=eos_id
    )
    eos_speculative = speculative_greedy_generate(
        eos_target,
        eos_draft,
        prompt,
        max_new_tokens=8,
        draft_length=3,
        eos_token_id=eos_id,
    )
    assert eos_baseline.token_ids == [eos_id]
    assert eos_speculative.token_ids == [eos_id]

    print("Smoke test 通过")
    print("候选匹配、Target 等价性、全接受、拒绝回滚和 EOS 检查通过")


if __name__ == "__main__":
    main()
