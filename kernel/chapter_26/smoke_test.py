#!/usr/bin/env python3
"""第 26 期 smoke test：draft-verify 循环的边界行为验证。

两层：
1. Mock/小模型单测（无 0.6B 权重依赖，秒级）：
   - 接受游走的全接受/中途拒绝/奖励 token 路径（mock 目标模型控制 argmax）；
   - 统计计数器（proposed/evaluated/accepted）与游标语义一致；
   - k<1 拒绝；copy draft 语义；
   - 小型真 Qwen3（随机权重 2 层）上：fast_greedy_generate 与 ch01
     greedy_generate 逐 token 一致；投机输出与贪心输出逐 token 一致
     （分布不变性的小模型实证）；random draft 同种子确定性；
   - MTP 头 W_proj 两种初始化的数值行为。
2. 真模型检查（--with-real-model，加载 0.6B FP32 约 1-2 分钟）：
   - 首 Token logits SHA256 == ch01 真值（权重字节完整性）；
   - fast_greedy 与 ch01 greedy_generate 一致；投机(k=3, reuse) 32 Token
     与 ch01 真值逐 token 一致。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ep27_common  # noqa: E402

ep27_common.bootstrap()

from mtp_reference import (  # noqa: E402
    CopyLastDraft,
    DraftConfig,
    MTPDraftModel,
    SpecStats,
    SpeculativeDecoder,
    build_draft,
    fast_greedy_generate,
)
from qwen3_model import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 第一层：mock 接受游走
# ---------------------------------------------------------------------------


class MockTarget:
    """可编程 argmax 的假目标模型（只实现 SpeculativeDecoder 用到的接口）。

    argmax_table[j] = verify 的第 j 行 logits（位置 n-1..n+k-1）的 argmax。
    backbone 返回零隐藏（内容无关），lm_head 按行号查表返回 one-hot logits。
    """

    def __init__(self, argmax_table: list[int], vocab_size: int = 50):
        self.argmax_table = argmax_table
        self.vocab_size = vocab_size
        self.model = self  # backbone_hidden 调用 model.model(input_ids)

        def lm_head(hidden):
            rows = hidden.shape[1]
            logits = torch.zeros(1, rows, self.vocab_size)  # [1, rows, V]，与真 LM Head 一致
            for row in range(rows):
                source = self.argmax_table[row] if row < len(self.argmax_table) else 0
                logits[0, row, source] = 10.0
            return logits

        self.lm_head = lm_head  # logits_at_rows 调用 model.lm_head(切片)

    def __call__(self, input_ids):
        batch, seq = input_ids.shape
        return torch.zeros(batch, seq, 8)


def make_decoder_with_table(table: list[int]) -> SpeculativeDecoder:
    return SpeculativeDecoder(MockTarget(table), CopyLastDraft(), eos_token_id=None)


def test_mock_accept_walk() -> None:
    # 表：目标 argmax 依次为 [7, 8, 9, 10]，k=3。
    decoder = make_decoder_with_table([7, 8, 9, 10])
    # draft == [7, 8, 9]：全接受 → emitted=[7,8,9,10]（奖励来自第 k 行）。
    emitted, accepted, info = decoder.verify_and_accept([1, 2, 3], [7, 8, 9])
    check("mock 全接受路径", emitted == [7, 8, 9, 10] and accepted == 3 and info["outcome"] == "accept_all", str(emitted))
    # draft == [7, 99, 9]：位置 1 拒绝 → emitted=[7, 8]（纠正 = 表第 1 行）。
    emitted, accepted, info = decoder.verify_and_accept([1, 2, 3], [7, 99, 9])
    check("mock 中途拒绝路径", emitted == [7, 8] and accepted == 1 and info["outcome"] == "reject_at_1", str(emitted))
    # draft == [99, ...]：位置 0 拒绝 → emitted=[7]。
    emitted, accepted, info = decoder.verify_and_accept([1, 2, 3], [99, 8, 9])
    check("mock 位置 0 拒绝", emitted == [7] and accepted == 0 and info["outcome"] == "reject_at_0", str(emitted))

    # 统计计数器游标语义：accepts=1, k=3 → evaluated=[1,1,0]（位置 0、1）
    stats = SpecStats()
    stats.note_iteration(3, 1)
    ok = (
        stats.proposed == [1, 1, 1]
        and stats.evaluated == [1, 1, 0]
        and stats.accepted == [1, 0, 0]
    )
    check("stats 游标语义 accepts=1", ok, f"proposed={stats.proposed} evaluated={stats.evaluated} accepted={stats.accepted}")
    stats.note_iteration(3, 3)
    # iter1 accepts=1 → evaluated [1,1,0]；iter2 全接受（k=3）→ 三个位置都被游标到达。
    ok = (
        stats.proposed == [2, 2, 2]
        and stats.evaluated == [2, 2, 1]
        and stats.accepted == [2, 1, 1]
    )
    check("stats 游标语义 accepts=3", ok, f"evaluated={stats.evaluated} accepted={stats.accepted}")
    stats.note_iteration(2, 0)  # 截断轮（step_k=2）：位置 0 即拒绝 → 只有位置 0 被评估
    ok = stats.proposed == [3, 3, 2] and stats.evaluated == [3, 2, 1] and stats.accepted == [2, 1, 1]
    check("stats 游标语义 accepts=0/k=2", ok, f"proposed={stats.proposed} evaluated={stats.evaluated}")


def test_k_guard() -> None:
    decoder = make_decoder_with_table([1, 2, 3])
    try:
        decoder.verify_and_accept([1, 2], [])
        check("k=0 拒绝", False, "未抛出")
    except ValueError:
        check("k=0 拒绝", True)
    try:
        list(decoder.generate([1, 2], max_new_tokens=4, k=0))
        check("generate k=0 拒绝", False, "未抛出")
    except ValueError:
        check("generate k=0 拒绝", True)
    copy_draft = CopyLastDraft()
    check("copy draft 语义", copy_draft.draft([5, 6, 7], 3) == [7, 7, 7])

    # EOS 截断：draft=[EOS,8,9] 与目标 argmax 表 [EOS,8,9,10] 逐位相等 →
    # 全接受本应发射 4 个 token，但输出必须截断在 [EOS] 并停止。
    eos_id = 42
    eos_decoder = SpeculativeDecoder(MockTarget([eos_id, 8, 9, 10]), CopyLastDraft(), eos_token_id=eos_id)

    class FixedDraft:
        needs_seed_hidden = False

        def draft(self, token_ids, k, seed_hidden=None):
            return [eos_id, 8, 9][:k]

    eos_decoder.draft = FixedDraft()
    generated, finished, stats = eos_decoder.generate([1, 2], max_new_tokens=8, k=3)
    check(
        "EOS 截断于本轮发射序列",
        generated == [1, 2, eos_id] and finished and stats.emitted_tokens == 1,
        f"generated={generated}",
    )


# ---------------------------------------------------------------------------
# 第二层：小型真 Qwen3（随机权重，2 层）——结构级不变量
# ---------------------------------------------------------------------------


def make_tiny_model(seed: int = 1234) -> Qwen3ForCausalLM:
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        config = Qwen3Config(
            vocab_size=256,
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
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
        model.tie_weights()
    finally:
        torch.random.set_rng_state(state)
    model.eval()
    return model


def test_tiny_model_invariants() -> None:
    model = make_tiny_model()
    prompt = [3, 14, 15, 9, 26, 5, 35, 9]
    input_ids = torch.tensor([prompt], dtype=torch.long)
    eos = None

    # fast_greedy vs ch01 greedy_generate（全序列 logits 口径 vs 仅末位口径）。
    reference_ids, reference_finished = model.greedy_generate(
        input_ids, max_new_tokens=12, eos_token_id=eos
    )
    fast_ids, fast_finished = fast_greedy_generate(
        model, prompt, max_new_tokens=12, eos_token_id=eos
    )
    check(
        "tiny: fast_greedy == ch01 greedy_generate",
        reference_ids[0].tolist() == fast_ids and reference_finished == fast_finished,
        f"{len(fast_ids) - len(prompt)} tokens",
    )

    # 三种 draft ×投机 vs 贪心逐 token 一致（分布不变性）。
    greedy_tokens = fast_ids[len(prompt):]
    for mode in ("reuse", "reuse_hidden", "random", "copy", "lookup", "lookup2"):
        draft_model = build_draft(model, mode, hidden_path="identity" if mode == "reuse_hidden" else "zero", seed=77)
        decoder = SpeculativeDecoder(model, draft_model, eos_token_id=eos)
        generated, finished, stats = decoder.generate(prompt, max_new_tokens=12, k=3)
        spec_tokens = generated[len(prompt):]
        check(
            f"tiny: 投机({mode}) == 贪心",
            spec_tokens == greedy_tokens,
            f"emitted={len(spec_tokens)} accepts_hist={stats.iteration_accepts}",
        )

    # random draft 同种子确定性（两个实例同 seed 权重一致）。
    draft_a = MTPDraftModel(model, DraftConfig(mode="random", seed=99))
    draft_b = MTPDraftModel(model, DraftConfig(mode="random", seed=99))
    same = all(
        torch.equal(a, b)
        for a, b in zip(draft_a.state_dict().values(), draft_b.state_dict().values())
    )
    check("tiny: random draft 同种子确定性", same)
    draft_c = MTPDraftModel(model, DraftConfig(mode="random", seed=100))
    different = any(
        not torch.equal(a, b)
        for a, b in zip(draft_a.state_dict().values(), draft_c.state_dict().values())
    )
    check("tiny: random draft 异种子不同", different)

    # W_proj 两种初始化的数值行为。
    ids = torch.tensor([prompt], dtype=torch.long)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(ids)
        zero_draft = MTPDraftModel(model, DraftConfig("reuse", "zero"))
        hidden_seed = torch.randn(1, 1, 32)
        combined_zero = zero_draft.proj(
            torch.cat(
                (
                    zero_draft.eh_norm(embeddings),
                    zero_draft.hh_norm(hidden_seed).expand(1, ids.shape[1], -1),
                ),
                dim=-1,
            )
        )
        expected_zero = zero_draft.eh_norm(embeddings)
        check(
            "tiny: W_proj [I|0] → 输出 == EHNorm(Emb)",
            torch.allclose(combined_zero, expected_zero, atol=1e-6),
        )
        identity_draft = MTPDraftModel(model, DraftConfig("reuse", "identity"))
        combined_identity = identity_draft.proj(
            torch.cat(
                (
                    identity_draft.eh_norm(embeddings),
                    identity_draft.hh_norm(hidden_seed).expand(1, ids.shape[1], -1),
                ),
                dim=-1,
            )
        )
        expected_identity = identity_draft.hh_norm(hidden_seed).expand(1, ids.shape[1], -1)
        check(
            "tiny: W_proj [0|I] → 输出 == HHNorm(h)",
            torch.allclose(combined_identity, expected_identity, atol=1e-6),
        )

    # 恒等 draft 在投机循环中的截断行为：max_new_tokens=5, k=3 → 发射 5 个。
    draft_model = build_draft(model, "copy")
    decoder = SpeculativeDecoder(model, draft_model, eos_token_id=eos)
    generated, _, stats = decoder.generate(prompt, max_new_tokens=5, k=3)
    check(
        "tiny: max_new_tokens 截断",
        len(generated) - len(prompt) == 5 and stats.emitted_tokens == 5,
        f"emitted={stats.emitted_tokens}",
    )


def test_real_model() -> None:
    import json

    model = ep27_common.load_verified_model(ep27_common.default_model_dir())
    truth = ep27_common.load_truth_manifest()
    fixtures = ep27_common.load_fixtures()
    prompt_entry = next(entry for entry in fixtures["prompts"] if entry["id"] == "P01")
    check("real: prompt_token_ids == ch01 真值", prompt_entry["prompt_token_ids"] == truth["prompt_token_ids"])

    prompt = truth["prompt_token_ids"]
    eos = truth["eos_token_id"]

    started = time.perf_counter()
    ch01_ids, ch01_finished = model.greedy_generate(
        torch.tensor([prompt], dtype=torch.long), max_new_tokens=32, eos_token_id=eos
    )
    print(f"[timing] ch01 greedy_generate {time.perf_counter() - started:.1f}s")
    fast_ids, fast_finished = fast_greedy_generate(model, prompt, 32, eos)
    check("real: fast_greedy == ch01 greedy", ch01_ids[0].tolist() == fast_ids and ch01_finished == fast_finished)

    draft_model = build_draft(model, "reuse")
    decoder = SpeculativeDecoder(model, draft_model, eos_token_id=eos)
    generated, finished, stats = decoder.generate(prompt, max_new_tokens=32, k=3)
    spec_tokens = generated[len(prompt):]
    check(
        "real: 投机(reuse,k=3) == ch01 真值逐 token",
        spec_tokens == truth["generated_token_ids"] and finished == truth["finished_by_eos"],
        f"{len(spec_tokens)} tokens, accepts={stats.iteration_accepts}",
    )

    # lookup draft（EOS 截断回归路径：模板上下文含 EOS+后继，可被整段抄中）。
    lookup_model = build_draft(model, "lookup")
    lookup_decoder = SpeculativeDecoder(model, lookup_model, eos_token_id=eos)
    generated, finished, stats = lookup_decoder.generate(prompt, max_new_tokens=32, k=8)
    spec_tokens = generated[len(prompt):]
    check(
        "real: 投机(lookup,k=8) == ch01 真值逐 token（含 EOS 截断）",
        spec_tokens == truth["generated_token_ids"],
        f"{len(spec_tokens)} tokens, finished={finished}, accepts={stats.iteration_accepts}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-real-model", action="store_true", help="附加 0.6B 真模型检查")
    arguments = parser.parse_args()

    test_mock_accept_walk()
    test_k_guard()
    test_tiny_model_invariants()
    if arguments.with_real_model:
        test_real_model()

    failures = [name for name, ok, _ in CHECKS if not ok]
    print(f"\n[smoke] {len(CHECKS) - len(failures)}/{len(CHECKS)} PASS")
    if failures:
        print("[smoke] FAILED:", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
