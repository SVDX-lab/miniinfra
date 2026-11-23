"""不下载权重、不要求 GPU 的 Continuous Batching 快速自检。"""

import torch

from continuous_batching import (
    left_pad_sequences,
    make_request_specs,
    run_continuous_batching,
    run_fixed_batching,
)
from qwen3_model import Qwen3Config, Qwen3ForCausalLM


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


def main():
    torch.manual_seed(7)
    device = torch.device("cpu")
    model = build_tiny_model()
    sequences = [
        [3, 5, 7, 9],
        [11, 13],
        [17, 19, 23],
        [29, 31, 37, 41, 43],
    ]
    budgets = [2, 4, 3, 2]

    padded, mask, positions, lengths = left_pad_sequences(
        sequences[:2], pad_token_id=0, device=device
    )
    assert padded.tolist() == [[3, 5, 7, 9], [0, 0, 11, 13]]
    assert mask.tolist() == [[True] * 4, [False, False, True, True]]
    assert positions.tolist() == [[0, 1, 2, 3], [0, 0, 0, 1]]
    assert lengths == [4, 2]

    specs = make_request_specs(sequences, budgets)
    fixed = run_fixed_batching(
        model, specs, 2, -1, device, stop_on_eos=False
    )
    continuous = run_continuous_batching(
        model, specs, 2, -1, device, stop_on_eos=False
    )

    isolated = []
    for sequence, budget in zip(sequences, budgets):
        result = run_fixed_batching(
            model,
            make_request_specs([sequence], budget),
            1,
            -1,
            device,
            stop_on_eos=False,
        )
        isolated.append(result["new_token_ids"][0])
    assert fixed["new_token_ids"] == isolated
    assert continuous["new_token_ids"] == isolated

    prefill_admissions = [
        event["admitted"]
        for event in continuous["events"]
        if event["phase"] == "prefill"
    ]
    assert prefill_admissions[0] == ["0", "1"]
    assert prefill_admissions[1] == ["2"]
    assert prefill_admissions[2] == ["3"]
    assert (
        continuous["metrics"]["execution_slot_utilization"]
        > fixed["metrics"]["execution_slot_utilization"]
    )
    assert all(
        item["queue_ms"] >= 0 and item["ttft_ms"] >= item["queue_ms"]
        for item in continuous["request_metrics"]
    )

    staggered_specs = make_request_specs(
        sequences[:2], [4, 2], arrival_times_ms=[0.0, 5.0]
    )
    staggered = run_continuous_batching(
        model, staggered_specs, 2, -1, device, stop_on_eos=False
    )
    assert [len(tokens) for tokens in staggered["new_token_ids"]] == [4, 2]
    assert staggered["request_metrics"][1]["admitted_ms"] >= 5.0

    same_arrival_specs = make_request_specs([sequences[0]] * 12, 1)
    same_arrival = run_continuous_batching(
        model, same_arrival_specs, 3, -1, device, stop_on_eos=False
    )
    flattened_admissions = [
        request_id
        for event in same_arrival["events"]
        for request_id in event["admitted"]
    ]
    assert flattened_admissions == [str(index) for index in range(12)]

    print("Smoke test 通过")
    print("左 Padding、动态加入退出、FCFS、Cache 重组检查通过")
    print("固定批次、Continuous Batching 与单请求输出一致")


if __name__ == "__main__":
    main()
