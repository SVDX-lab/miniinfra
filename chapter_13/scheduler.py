"""第 13 期独立的 Chunked Prefill 调度器。

本期完整复制所需底座，不 import 其他期代码。调度器负责在迭代内选择一次
Chunked Prefill 或一次 Decode，并维持
硬 Token Budget 与 Decode 保护规则。Block 级准入、抢占触发、受害者选择和
swapped 恢复由引擎的资源管理完成；调度器看到的 prefilling/running 均已是
持有 GPU Block 与运行名额的请求。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    token_ids: tuple
    max_new_tokens: int
    arrival_ms: float = 0.0

    def validate(self):
        if not self.request_id:
            raise ValueError("request_id 不能为空")
        if not self.token_ids:
            raise ValueError("Prompt Token 不能为空")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须大于 0")
        if self.arrival_ms < 0:
            raise ValueError("arrival_ms 不能小于 0")


@dataclass
class RequestState:
    spec: RequestSpec
    status: str = "waiting"
    generated: list = field(default_factory=list)
    slot_index: int | None = None
    cache_length: int = 0
    prefill_cursor: int = 0
    admitted_ms: float | None = None
    prefill_started_ms: float | None = None
    prefill_completed_ms: float | None = None
    first_token_ms: float | None = None
    completion_ms: float | None = None
    token_times_ms: list = field(default_factory=list)
    # 抢占与换出相关状态
    admission_seq: int = 0
    resuming: bool = False
    preempt_count: int = 0
    preempted_clock_ms: float | None = None
    paused_ms_total: float = 0.0
    resume_count: int = 0
    swap_out_count: int = 0
    swap_in_count: int = 0
    recompute_started_ms: float | None = None
    recompute_completed_ms: float | None = None
    recompute_prefill_ms_total: float = 0.0

    @property
    def prefill_source_tokens(self):
        """当前 Prefill 输入：新请求是 Prompt，recompute 恢复是 Prompt+已生成。"""
        if self.resuming:
            return tuple(self.spec.token_ids) + tuple(self.generated)
        return self.spec.token_ids


def make_request_specs(token_sequences, max_new_tokens, arrival_times_ms=None):
    sequences = [tuple(int(token) for token in sequence) for sequence in token_sequences]
    if isinstance(max_new_tokens, int):
        budgets = [max_new_tokens] * len(sequences)
    else:
        budgets = list(max_new_tokens)
    arrivals = (
        [0.0] * len(sequences)
        if arrival_times_ms is None else list(arrival_times_ms)
    )
    if len(budgets) != len(sequences) or len(arrivals) != len(sequences):
        raise ValueError("Prompt、输出预算和到达时间的数量必须一致")
    specs = [
        RequestSpec(str(index), sequence, int(budget), float(arrival))
        for index, (sequence, budget, arrival) in enumerate(
            zip(sequences, budgets, arrivals)
        )
    ]
    for spec in specs:
        spec.validate()
    if len({spec.request_id for spec in specs}) != len(specs):
        raise ValueError("request_id 必须唯一")
    return specs


@dataclass(frozen=True)
class SchedulerConfig:
    max_running_requests: int
    token_budget: int

    def validate(self):
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests 必须大于 0")
        if self.token_budget < self.max_running_requests:
            raise ValueError(
                "token_budget 不能小于 max_running_requests，"
                "否则无法为每个运行名额分配至少一个 Token"
            )


@dataclass(frozen=True)
class PrefillPlan:
    state: RequestState
    start: int
    end: int

    @property
    def token_count(self):
        return self.end - self.start


@dataclass(frozen=True)
class SchedulerOutput:
    phase: str
    selected: tuple
    prefill_plans: tuple = ()
    scheduled_tokens: int = 0
    logical_tokens: int = 0
    token_budget: int | None = None

    @property
    def request_ids(self):
        return [state.spec.request_id for state in self.selected]


class ChunkedPrefillScheduler:
    """每个 iteration 选择一次 Chunked Prefill 或一次 Decode。"""

    def __init__(self, config):
        config.validate()
        self.config = config
        self.last_phase = None
        self.iteration_count = 0

    @staticmethod
    def _padded_cost(lengths):
        return len(lengths) * max(lengths) if lengths else 0

    def _chunked_plans(self, candidates):
        if not candidates:
            return []
        # token_budget >= max_running_requests，因此所有已占用名额至少能推进
        # 一个 Token。每个候选使用相同上限，实际尾 Chunk 可以更短。
        chunk_cap = self.config.token_budget // len(candidates)
        plans = []
        for state in candidates:
            start = state.prefill_cursor
            remaining = len(state.prefill_source_tokens) - start
            if remaining < 1:
                raise RuntimeError("prefilling 请求没有剩余 Prompt Token")
            plans.append(PrefillPlan(state, start, start + min(remaining, chunk_cap)))
        cost = self._padded_cost([plan.token_count for plan in plans])
        if cost > self.config.token_budget:
            raise RuntimeError("Chunked Prefill 生成了超出硬预算的计划")
        return plans

    def schedule(self, prefilling, running):
        # 与第 06/07 期相同：已有 Decode 请求时，Prefill 后至少插入一次 Decode。
        if running and self.last_phase == "prefill":
            return self._decode_output(running)
        if prefilling:
            return self._prefill_output(self._chunked_plans(list(prefilling)))
        if running:
            return self._decode_output(running)
        return None

    def _prefill_output(self, plans):
        plans = tuple(plans)
        lengths = [plan.token_count for plan in plans]
        self.last_phase = "prefill"
        self.iteration_count += 1
        return SchedulerOutput(
            phase="prefill",
            selected=tuple(plan.state for plan in plans),
            prefill_plans=plans,
            scheduled_tokens=self._padded_cost(lengths),
            logical_tokens=sum(lengths),
            token_budget=self.config.token_budget,
        )

    def _decode_output(self, running):
        selected = tuple(running)
        self.last_phase = "decode"
        self.iteration_count += 1
        return SchedulerOutput(
            phase="decode",
            selected=selected,
            scheduled_tokens=len(selected),
            logical_tokens=len(selected),
            token_budget=self.config.token_budget,
        )
