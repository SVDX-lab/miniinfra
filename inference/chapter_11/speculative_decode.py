"""Target-only 与 Greedy Speculative Decode 的独立实现。"""

from dataclasses import asdict, dataclass, field

import torch


def token_argmax(logits):
    return int(torch.argmax(logits, dim=-1).item())


def token_argmax_tensor(logits):
    return torch.argmax(logits, dim=-1).reshape(1, 1)


def longest_matching_prefix(candidates, target_tokens):
    if len(candidates) != len(target_tokens):
        raise ValueError("候选和 Target 验证结果长度必须相同")
    accepted = 0
    for candidate, target in zip(candidates, target_tokens):
        if candidate != target:
            break
        accepted += 1
    return accepted


def truncate_at_eos(tokens, eos_token_id):
    if eos_token_id is None:
        return list(tokens), False
    result = []
    for token in tokens:
        result.append(token)
        if token == eos_token_id:
            return result, True
    return result, False


@dataclass
class RoundStats:
    draft_length: int
    candidates: list
    target_tokens: list
    accepted: int
    emitted: list
    all_accepted: bool


@dataclass
class GenerationStats:
    mode: str
    prompt_tokens: int
    generated_tokens: int = 0
    target_prefill_calls: int = 0
    target_decode_calls: int = 0
    draft_prefill_calls: int = 0
    draft_decode_calls: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    rounds: list = field(default_factory=list)

    @property
    def acceptance_rate(self):
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens

    @property
    def mean_accepted_tokens_per_round(self):
        if not self.rounds:
            return 0.0
        return self.accepted_tokens / len(self.rounds)

    def to_dict(self, include_rounds=True):
        result = asdict(self)
        result["acceptance_rate"] = self.acceptance_rate
        result["mean_accepted_tokens_per_round"] = (
            self.mean_accepted_tokens_per_round
        )
        if not include_rounds:
            result.pop("rounds")
        return result


@dataclass
class GenerationResult:
    token_ids: list
    stats: GenerationStats
    target_cache_length: int
    draft_cache_length: int = None


def prefill(model, prompt_ids, max_length):
    if not prompt_ids:
        raise ValueError("prompt_ids 不能为空")
    cache = model.new_cache(max_length)
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits = model(input_ids, cache)
    return cache, logits[:, -1, :]


@torch.inference_mode()
def target_greedy_generate(
    model,
    prompt_ids,
    max_new_tokens,
    eos_token_id=None,
    max_cache_length=None,
):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须大于 0")
    capacity = (
        len(prompt_ids) + max_new_tokens
        if max_cache_length is None
        else max_cache_length
    )
    cache, next_logits = prefill(model, prompt_ids, capacity)
    stats = GenerationStats(
        mode="target_only",
        prompt_tokens=len(prompt_ids),
        target_prefill_calls=1,
    )
    pending = token_argmax_tensor(next_logits)
    generated = [pending]
    stopped = eos_token_id is not None and int(pending.item()) == eos_token_id
    while len(generated) < max_new_tokens and not stopped:
        device = next(model.parameters()).device
        logits = model(
            pending.to(device=device), cache
        )
        stats.target_decode_calls += 1
        pending = token_argmax_tensor(logits[:, -1, :])
        generated.append(pending)
        stopped = eos_token_id is not None and int(pending.item()) == eos_token_id
    generated_ids = torch.cat(generated, dim=1).flatten().tolist()
    stats.generated_tokens = len(generated_ids)
    return GenerationResult(
        token_ids=generated_ids,
        stats=stats,
        target_cache_length=cache.length,
    )


@torch.inference_mode()
def speculative_greedy_generate(
    target_model,
    draft_model,
    prompt_ids,
    max_new_tokens,
    draft_length,
    eos_token_id=None,
    max_cache_length=None,
    include_round_details=True,
):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须大于 0")
    if draft_length < 1:
        raise ValueError("draft_length 必须大于 0")
    if target_model.config.vocab_size != draft_model.config.vocab_size:
        raise ValueError("Target 与 Draft 的词表大小必须相同")
    capacity = (
        len(prompt_ids) + max_new_tokens + draft_length
        if max_cache_length is None
        else max_cache_length
    )
    target_cache, target_next_logits = prefill(
        target_model, prompt_ids, capacity
    )
    draft_cache, _ = prefill(draft_model, prompt_ids, capacity)
    stats = GenerationStats(
        mode="speculative",
        prompt_tokens=len(prompt_ids),
        target_prefill_calls=1,
        draft_prefill_calls=1,
    )

    pending = token_argmax_tensor(target_next_logits)
    generated = [pending]
    target_device = next(target_model.parameters()).device
    draft_device = next(draft_model.parameters()).device

    stopped = eos_token_id is not None and int(pending.item()) == eos_token_id
    while len(generated) < max_new_tokens and not stopped:
        common_length = target_cache.length
        if draft_cache.length != common_length:
            raise RuntimeError("Target 与 Draft KV 长度失去同步")

        draft_logits = draft_model(
            pending.to(device=draft_device),
            draft_cache,
        )[:, -1, :]
        stats.draft_decode_calls += 1
        candidates = []
        for _ in range(draft_length):
            candidate = token_argmax_tensor(draft_logits)
            candidates.append(candidate)
            draft_logits = draft_model(
                candidate,
                draft_cache,
            )[:, -1, :]
            stats.draft_decode_calls += 1
            if eos_token_id is not None and int(candidate.item()) == eos_token_id:
                break

        candidate_tensor = torch.cat(candidates, dim=1).to(device=target_device)
        verify_ids = torch.cat(
            (pending.to(device=target_device), candidate_tensor), dim=1
        )
        verify_logits = target_model(
            verify_ids,
            target_cache,
        )
        stats.target_decode_calls += 1
        target_token_tensor = torch.argmax(
            verify_logits[:, : len(candidates), :], dim=-1
        )
        bonus_token = token_argmax_tensor(verify_logits[:, len(candidates), :])
        matches = (candidate_tensor == target_token_tensor).flatten()
        accepted = int(torch.cumprod(matches.to(torch.int32), dim=0).sum().item())
        if accepted == len(candidates):
            round_output = torch.cat((candidate_tensor, bonus_token), dim=1)
        else:
            correction = target_token_tensor[:, accepted : accepted + 1]
            round_output = torch.cat(
                (candidate_tensor[:, :accepted], correction), dim=1
            )
        reached_eos = False
        if eos_token_id is not None:
            eos_positions = (round_output == eos_token_id).flatten().nonzero()
            if eos_positions.numel() > 0:
                round_output = round_output[:, : int(eos_positions[0].item()) + 1]
                reached_eos = True
        remaining = max_new_tokens - len(generated)
        emitted = round_output[:, :remaining]
        if emitted.numel() == 0:
            raise RuntimeError("推测解码一轮没有产生 Token")

        # 两套 Cache 都只保留本轮最后一个输出 Token 之前的公共历史。
        # 最后一个 Token 作为下一轮 pending，与下一批候选一起送入 Target。
        emitted_count = emitted.shape[1]
        committed_this_round = 1 + emitted_count - 1
        synchronized_length = common_length + committed_this_round
        target_cache.rollback(synchronized_length)
        draft_cache.rollback(synchronized_length)
        pending = emitted[:, -1:]
        generated.extend(emitted[:, index : index + 1] for index in range(emitted_count))

        stats.proposed_tokens += len(candidates)
        stats.accepted_tokens += accepted
        if include_round_details:
            candidate_ids = candidate_tensor.flatten().tolist()
            target_token_ids = target_token_tensor.flatten().tolist()
            emitted_ids = emitted.flatten().tolist()
            stats.rounds.append(
                RoundStats(
                    draft_length=len(candidate_ids),
                    candidates=candidate_ids,
                    target_tokens=target_token_ids,
                    accepted=accepted,
                    emitted=emitted_ids,
                    all_accepted=accepted == len(candidate_ids),
                )
            )
        else:
            # 保留轻量占位，使平均每轮接受数仍可计算。
            stats.rounds.append(None)
        if reached_eos:
            break

    generated_ids = torch.cat(generated, dim=1).flatten().tolist()
    stats.generated_tokens = len(generated_ids)
    return GenerationResult(
        token_ids=generated_ids,
        stats=stats,
        target_cache_length=target_cache.length,
        draft_cache_length=draft_cache.length,
    )
