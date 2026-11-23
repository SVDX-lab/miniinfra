"""不下载权重、不要求 GPU 的第 06 期快速自检。"""

import torch

from engine import run_scheduler
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from scheduler import (
    IterationScheduler,
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


def check_scheduler_decisions():
    specs = make_request_specs(
        [[1, 2, 3], [4, 5, 6, 7, 8], list(range(9))],
        [3, 3, 3],
    )
    waiting = [RequestState(spec) for spec in specs]
    running_spec = make_request_specs([[11, 12]], [4])[0]
    running_state = RequestState(running_spec, status="running")

    baseline = IterationScheduler(SchedulerConfig("baseline", 4))
    decision = baseline.schedule(waiting, [running_state], 0.0)
    assert decision.phase == "prefill"
    assert decision.request_ids == ["0", "1", "2"]

    budgeted = IterationScheduler(SchedulerConfig("budgeted", 4, 6))
    decision = budgeted.schedule(waiting, [running_state], 0.0)
    assert decision.phase == "prefill"
    assert decision.request_ids == ["0"]
    assert decision.scheduled_tokens == 3
    decision = budgeted.schedule(waiting[1:], [running_state], 0.0)
    assert decision.phase == "decode"

    oversize = IterationScheduler(SchedulerConfig("budgeted", 4, 6))
    decision = oversize.schedule(waiting[2:], [], 0.0)
    assert decision.request_ids == ["2"]
    assert decision.scheduled_tokens == 9
    assert decision.oversize_singleton


def check_engine():
    torch.manual_seed(7)
    device = torch.device("cpu")
    model = build_tiny_model()
    sequences = [
        [3, 5, 7],
        [11, 13, 17, 19],
        [23, 29, 31, 37, 41, 43],
        [47, 53],
        [59, 61, 67, 71, 73],
    ]
    output_budgets = [5, 5, 1, 1, 3]
    arrivals = [0.0, 0.0, 0.1, 0.1, 0.1]
    specs = make_request_specs(sequences, output_budgets, arrivals)
    baseline = run_scheduler(
        model, specs, 4, -1, device,
        policy="baseline", block_size=4, stop_on_eos=False,
    )
    budgeted = run_scheduler(
        model, specs, 4, -1, device,
        policy="budgeted", token_budget=6,
        block_size=4, stop_on_eos=False,
    )
    assert baseline["new_token_ids"] == budgeted["new_token_ids"]
    assert baseline["metrics"]["max_consecutive_prefill_iterations"] > 1
    assert budgeted["metrics"]["max_consecutive_prefill_iterations"] == 1
    assert budgeted["metrics"]["oversize_prefill_iterations"] == 0
    assert budgeted["metrics"]["prefill_padded_tokens"] <= baseline["metrics"][
        "prefill_padded_tokens"
    ]
    assert budgeted["metrics"]["block_reuse_count"] > 0
    return baseline, budgeted


def main():
    check_scheduler_decisions()
    baseline, budgeted = check_engine()
    print("Smoke test 通过")
    print("FCFS、Token Budget、Decode 保护和超预算单请求检查通过")
    print("baseline 与 budgeted 逐请求输出一致")
    print(
        "最大连续 Prefill iteration: baseline=%d, budgeted=%d"
        % (
            baseline["metrics"]["max_consecutive_prefill_iterations"],
            budgeted["metrics"]["max_consecutive_prefill_iterations"],
        )
    )


if __name__ == "__main__":
    main()
