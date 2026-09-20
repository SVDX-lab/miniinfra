#!/usr/bin/env python3
"""CPU-only Chapter 20 checks: packing plan, isolation, equivalence, negative control."""

from __future__ import annotations

import copy

import torch

from assistant_loss import EncodedConversation, IGNORE_INDEX
from mini_batch import batch_loss
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from sequence_packing import (
    build_block_diagonal_mask,
    build_pack,
    greedy_packing_plan,
)


def synthetic_sample(ids: list[int], supervised_from: int, source_line: int) -> EncodedConversation:
    labels = [IGNORE_INDEX if index < supervised_from else token for index, token in enumerate(ids)]
    return EncodedConversation(
        input_ids=torch.tensor([ids]),
        attention_mask=torch.tensor([[1] * len(ids)]),
        labels=torch.tensor([labels]),
        assistant_loss_mask=torch.tensor([[label != IGNORE_INDEX for label in labels]]),
        source_line=source_line,
        rendered_text="",
        token_spans=(),
        loss_mode="assistant_only",
    )


def tiny_model(seed: int = 17) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=41, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        rms_norm_eps=1e-6, rope_theta=10_000.0, torch_dtype="float32", tie_word_embeddings=True,
    ))


SAMPLES = [
    synthetic_sample(list(range(1, 12)), 4, 201),          # length 11
    synthetic_sample(list(range(20, 32)), 6, 202),         # length 12
    synthetic_sample(list(range(30, 35)), 2, 203),         # length 5
    synthetic_sample(list(range(24, 30)), 3, 204),         # length 6
]


def check_greedy_plan() -> None:
    lengths = [11, 12, 5, 6, 9, 10, 3]
    plan = greedy_packing_plan(lengths, capacity=16, seed=5, epoch=1)
    covered = sorted(index for pack in plan.packs for index in pack)
    assert covered == list(range(len(lengths))), "packs must cover each sample exactly once"
    for pack in plan.packs:
        assert sum(lengths[index] for index in pack) <= 16, "pack must respect capacity"
    replay = greedy_packing_plan(lengths, capacity=16, seed=5, epoch=1)
    assert replay.packs == plan.packs, "same seed must replay exactly"
    other = greedy_packing_plan(lengths, capacity=16, seed=6, epoch=1)
    assert other.packs != plan.packs, "different seeds must differ"
    assert 0.0 < plan.utilization(lengths) <= 1.0
    try:
        greedy_packing_plan([20], capacity=16, seed=1, epoch=1)
    except ValueError:
        pass
    else:
        raise AssertionError("a sample beyond capacity must raise")
    # 严格小于容量的长度组合：全排序语料存在无浪费的装包
    tight = greedy_packing_plan([8, 8, 8, 8], capacity=16, seed=1, epoch=1)
    assert tight.utilization([8, 8, 8, 8]) == 1.0


def check_pack_tensors_and_mask() -> None:
    lengths = [sample.non_padding_tokens for sample in SAMPLES]
    plan = greedy_packing_plan(lengths, capacity=18, seed=7, epoch=1)
    pack_index = max(range(plan.num_packs), key=lambda k: len(plan.packs[k]))
    members = plan.packs[pack_index]
    pack = build_pack(SAMPLES, members, capacity=18, pad_token_id=0)
    assert pack.input_ids.shape == (1, 18) and pack.labels.shape == (1, 18)
    assert pack.sample_spans[0] == (0, lengths[members[0]])
    total = sum(lengths[index] for index in members)
    assert pack.non_padding_tokens == total
    # Position ID 在每个样本内从 0 重置
    for (start, end), index in zip(pack.sample_spans, members):
        expected = torch.arange(end - start)
        assert torch.equal(pack.position_ids[0, start:end], expected)
        first_label = pack.labels[0, start]
        if start > 0:
            assert first_label == IGNORE_INDEX, "boundary label must be cut"
        assert torch.equal(
            pack.labels[0, start + 1 : end], SAMPLES[index].labels[0, 1 : end - start]
        ), "in-sample labels must equal the independent encoding"
    assert pack.supervised_tokens == sum(
        SAMPLES[index].supervised_tokens for index in members
    )
    # Mask：样本内下三角，跨样本与右侧 Padding 全部屏蔽
    mask = build_block_diagonal_mask(pack.sample_spans, 18, torch.float32, torch.device("cpu"), total_length=total)
    assert mask.shape == (1, 1, 18, 18)
    minimum = torch.finfo(torch.float32).min
    dense = mask[0, 0]
    for (start, end) in pack.sample_spans:
        for query in range(start, end):
            for key in range(18):
                allowed = (start <= key < end) and key <= query
                assert (dense[query, key].item() == 0.0) == allowed, (query, key)
    for query in range(total, 18):
        assert (dense[query, :total] == minimum).all(), "pad rows must not see real keys"
    assert (dense[:total, total:] == minimum).all(), "pad keys must be masked"


def run_pack_and_independent(model: Qwen3ForCausalLM, members, capacity: int, isolation: bool = True):
    pack = build_pack(SAMPLES, members, capacity=capacity, pad_token_id=0)
    if isolation:
        mask = build_block_diagonal_mask(pack.sample_spans, capacity, torch.float32, torch.device("cpu"))
        logits = model(
            pack.input_ids,
            attention_mask=pack.attention_mask,
            position_ids=pack.position_ids,
            attention_mask_4d=mask,
        )
    else:
        logits = model(pack.input_ids, attention_mask=pack.attention_mask)
    return pack, logits


def check_equivalence_isolation_and_negative_control() -> None:
    model = tiny_model()
    model.eval()
    lengths = [sample.non_padding_tokens for sample in SAMPLES]
    plan = greedy_packing_plan(lengths, capacity=18, seed=7, epoch=1)
    members = max(plan.packs, key=len)
    assert len(members) >= 2, "test needs a multi-sample pack"

    with torch.no_grad():
        pack, pack_logits = run_pack_and_independent(model, members, 18, isolation=True)
        independent = {}
        for index in members:
            sample = SAMPLES[index]
            independent[index] = model(sample.input_ids[:, : sample.non_padding_tokens])
        for (start, end), index in zip(pack.sample_spans, members):
            error = (pack_logits[0, start:end, :] - independent[index][0]).abs().max().item()
            assert error < 2e-5, f"pack sample {index} logits diverge: {error}"

        # 隔离证明：整体替换更早样本的内容，后续样本 Logits 逐位不变
        swapped_input = pack.input_ids.clone()
        first_start, first_end = pack.sample_spans[0]
        swapped_input[0, first_start:first_end] = torch.randint(
            1, 41, (1, first_end - first_start)
        )
        mask = build_block_diagonal_mask(pack.sample_spans, 18, torch.float32, torch.device("cpu"))
        swapped_logits = model(
            swapped_input,
            attention_mask=pack.attention_mask,
            position_ids=pack.position_ids,
            attention_mask_4d=mask,
        )
        for (start, end) in pack.sample_spans[1:]:
            assert torch.equal(swapped_logits[0, start:end, :], pack_logits[0, start:end, :]), \
                "later samples must not see earlier content"

        # 负对照：连续 Position + 普通因果 Mask，第一个样本仍一致，后续样本显著偏离
        _, naive_logits = run_pack_and_independent(model, members, 18, isolation=False)
        first_start, first_end = pack.sample_spans[0]
        first_error = (naive_logits[0, first_start:first_end, :] - independent[members[0]][0]).abs().max().item()
        assert first_error < 2e-5, f"negative control must leave the first sample intact: {first_error}"
        for (start, end), index in list(zip(pack.sample_spans, members))[1:]:
            divergence = (naive_logits[0, start:end, :] - independent[index][0]).abs().max().item()
            assert divergence > 1e-2, f"negative control failed to diverge at sample {index}: {divergence}"


def check_loss_and_gradient_equivalence() -> None:
    lengths = [sample.non_padding_tokens for sample in SAMPLES]
    plan = greedy_packing_plan(lengths, capacity=18, seed=7, epoch=1)
    members = max(plan.packs, key=len)

    serial = tiny_model()
    packed = copy.deepcopy(serial)
    packed.load_state_dict(serial.state_dict())

    pack = build_pack(SAMPLES, members, capacity=18, pad_token_id=0)
    mask = build_block_diagonal_mask(pack.sample_spans, 18, torch.float32, torch.device("cpu"))
    pack_logits = packed(pack.input_ids, attention_mask=pack.attention_mask, position_ids=pack.position_ids, attention_mask_4d=mask)
    pack_loss = batch_loss(pack_logits, pack.labels, "token_mean")
    pack_loss.backward()

    total_supervised = sum(SAMPLES[index].supervised_tokens for index in members)
    independent_loss = 0.0
    for index in members:
        sample = SAMPLES[index]
        length = sample.non_padding_tokens
        logits = serial(sample.input_ids[:, :length])
        loss = batch_loss(logits, sample.labels[:, :length], "token_mean")
        weight = sample.supervised_tokens / total_supervised
        (loss * weight).backward()
        independent_loss += float(loss.detach().item()) * weight

    assert abs(float(pack_loss.detach().item()) - independent_loss) < 2e-6, "pack loss must equal the weighted sum"
    for name, parameter in packed.named_parameters():
        reference = dict(serial.named_parameters())[name]
        assert torch.allclose(parameter.grad, reference.grad, atol=3e-6, rtol=3e-5), f"gradient mismatch at {name}"


def main() -> None:
    check_greedy_plan()
    check_pack_tensors_and_mask()
    check_equivalence_isolation_and_negative_control()
    check_loss_and_gradient_equivalence()
    print("chapter_20 smoke tests passed")


if __name__ == "__main__":
    main()
