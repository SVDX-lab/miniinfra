"""不下载真实权重、不要求 GPU 的第 07 期快速自检。"""

import torch

from engine import run_engine
from qwen3_model import Qwen3Config, Qwen3ForCausalLM
from scheduler import (
    ChunkedPrefillScheduler,
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


def check_scheduler():
    spec = make_request_specs([list(range(1, 10))], [3])[0]
    state = RequestState(spec)
    full = ChunkedPrefillScheduler(SchedulerConfig("full", 2, 4))
    decision = full.schedule([state], [], [], 0.0)
    assert decision.prefill_plans[0].start == 0
    assert decision.prefill_plans[0].end == 9
    assert decision.scheduled_tokens == 9
    assert decision.oversize_singleton

    chunked = ChunkedPrefillScheduler(SchedulerConfig("chunked", 2, 4))
    decision = chunked.schedule([state], [], [], 0.0)
    assert decision.prefill_plans[0].start == 0
    assert decision.prefill_plans[0].end == 4
    assert decision.scheduled_tokens == 4
    assert not decision.oversize_singleton
    state.status = "prefilling"
    state.prefill_cursor = 4
    decision = chunked.schedule([], [state], [], 1.0)
    assert (decision.prefill_plans[0].start, decision.prefill_plans[0].end) == (4, 8)


def check_engine():
    torch.manual_seed(7)
    device = torch.device("cpu")
    model = build_tiny_model()
    sequences = [
        [3, 5, 7],
        [11, 13, 17, 19, 23, 29, 31, 37, 41],
        [43, 47, 53, 59, 61],
        [67, 71],
    ]
    specs = make_request_specs(sequences, [5, 4, 3, 1], [0.0, 0.1, 0.1, 0.2])
    full = run_engine(
        model, specs, 3, -1, device,
        mode="full", token_budget=4, block_size=3,
        stop_on_eos=False, capture_logits=True,
    )
    chunked = run_engine(
        model, specs, 3, -1, device,
        mode="chunked", token_budget=4, block_size=3,
        stop_on_eos=False, capture_logits=True,
    )
    assert full["new_token_ids"] == chunked["new_token_ids"]
    for request_id in full["first_token_logits"]:
        error = torch.max(torch.abs(
            full["first_token_logits"][request_id]
            - chunked["first_token_logits"][request_id]
        )).item()
        assert error < 1e-4, (request_id, error)
    assert full["metrics"]["oversize_prefill_iterations"] > 0
    assert chunked["metrics"]["oversize_prefill_iterations"] == 0
    assert chunked["metrics"]["hard_budget_violations"] == 0
    assert chunked["metrics"]["prefill_iterations"] > full["metrics"][
        "prefill_iterations"
    ]
    assert chunked["metrics"]["block_reuse_count"] > 0
    return full, chunked


def main():
    check_scheduler()
    full, chunked = check_engine()
    print("Smoke test 通过")
    print("Prefill cursor、跨 Block 追加、硬 Token Budget 检查通过")
    print("full 与 chunked 逐请求输出一致")
    print(
        "Prefill iteration: full=%d, chunked=%d"
        % (
            full["metrics"]["prefill_iterations"],
            chunked["metrics"]["prefill_iterations"],
        )
    )


if __name__ == "__main__":
    main()
