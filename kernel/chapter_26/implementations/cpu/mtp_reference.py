"""第 26 期：MTP——多 Token 预测与投机解码，最小 draft-verify 循环参考实现（CPU）。

组成（全部以第 01 期手写 Qwen3 为目标模型，FP32，无 KV Cache、全序列重算，
与 ch01 "不引入优化" 边界一致——本期引入的唯一样式差异是"只在需要的位置
计算 logits"，该差异对 greedy 与投机两侧同口径生效，不改变 token 结果）：

1. MTPDraftModel：draft = 单个 Qwen3DecoderLayer + MTP 头。
   - 结构对齐 DeepSeek-V3 MTP 模块：EHNorm(Emb) ⊕ HHNorm(h_prev) → concat →
     W_proj 线性组合 → 单层 Transformer（复用目标模型第 0 层或同形状随机
     初始化）→ FinalNorm → 共享 LM Head（tied Embedding 线性投影）。
   - k 个候选 token 由该模块自回归迭代 k 步产生（DeepSeek MTP 的实际用法：
     同一模块逐步展开，而不是一个 k×vocab 的一次性投影头——后者等价于
     k 个独立分类器，无训练时各槽位无法互相条件化）。
   - 两种权重模式（实验 3 的对照）：
     * reuse：单层权重 = 目标模型第 0 层 + FinalNorm/LM Head/EHNorm 复用
       目标模型权重（EHNorm 借用 final norm，作为"未训练 MTP 归一化"的
       代替品，显式声明）；
     * random：同形状独立随机初始化（std=0.02，fixed seed）——draft 质量
       下界对照。
   - W_proj 的 hidden 半边支持 "zero"（只用 embedding 路径）与 "identity"
     （条件于目标模型 hidden）两种未训练初始化，显式声明。
2. CopyLastDraft：恒等复制上一个 token 的无模型 draft（启发式下界）。
3. SpeculativeDecoder：greedy 接受/拒绝循环。
   - greedy 下的接受判定 = 逐位置相等测试（draft token == 目标模型该位置
     argmax 即接受；遇到拒绝位置，插入该位置目标 argmax 作为纠正 token 并
     终止本轮）；全部接受时追加 verify 最后一行的 argmax 作为奖励 token。
   - 与随机拒绝采样（Leviathan/Chen 2023：按 min(1, p/q) 接受 + 残差分布
     重归一化补采样）的区别写在 EXPERIMENT_LOG 验收问答：greedy 是温度→0
     的退化情形，接受判定塌缩为相等测试，残差分布塌缩为 argmax 点质量，
     因此"不改分布"的证明退化为"每个输出 token 都等于目标模型在相应前缀
     下的 argmax"（归纳证明见实验 1）。
4. fast_greedy_generate：与 ch01 greedy_generate 同 token 结果的"仅末位
   logits"贪心解码（速度基线，LM Head 只算最后一行；与投机侧同口径）。

所有接口不修改 chapter_01 的任何文件；权重加载复用本期 copy 的
qwen3_model.load_handwritten_model。
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

# 复用本期目录内 ch01 copy 的模型定义（同目录 import）。
from qwen3_model import (  # noqa: F401  (re-export 供驱动脚本使用)
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    RMSNorm,
    load_handwritten_model,
    resolve_model_directory,
)


# ---------------------------------------------------------------------------
# 工具：只在需要的位置计算 logits（不改动 ch01 模型文件）
# ---------------------------------------------------------------------------


def _model_device(module: nn.Module) -> torch.device:
    """从模型参数推断设备（musa/cuda 列的 tensor 创建口径；CPU 透明）。"""
    parameters = getattr(module, "parameters", None)
    if parameters is not None:
        try:
            return next(module.parameters()).device
        except (StopIteration, AttributeError):
            pass
    return torch.device("cpu")


@torch.no_grad()
def backbone_hidden(model: Qwen3ForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """目标模型 backbone（28 层 + final norm），返回 [1, seq, hidden]。

    注意：Qwen3Model.forward 末尾已施加 final RMSNorm（ch01 行为保持），
    因此返回的是 final-norm 后的隐藏状态。verify 的 logits 与 draft 的
    条件 hidden 都从它切片。
    """
    return model.model(input_ids)


@torch.no_grad()
def logits_at_rows(model: Qwen3ForCausalLM, hidden: torch.Tensor, rows: slice) -> torch.Tensor:
    """对隐藏状态的指定位置行计算 LM Head logits（tied Embedding 线性投影）。"""
    return model.lm_head(hidden[:, rows, :])


@torch.no_grad()
def fast_greedy_generate(
    model: Qwen3ForCausalLM,
    input_ids,
    max_new_tokens: int,
    eos_token_id: int | None = None,
):
    """仅末位 logits 的贪心解码（速度基线）。

    与 ch01 greedy_generate 的唯一差异是 LM Head 只算最后一个位置
    （ch01 对全序列算 logits）。token 结果必须逐位一致——由 smoke_test.py
    用 ch01 原版 greedy_generate 逐 token 断言。
    返回 (generated_ids_list, finished)。
    """
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=_model_device(model))
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0).to(_model_device(model))
    generated = input_ids
    finished = False
    for _ in range(max_new_tokens):
        hidden = backbone_hidden(model, generated)
        logits = logits_at_rows(model, hidden, slice(-1, None))
        next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token_id), dim=1)
        if eos_token_id is not None and bool((next_token_id == eos_token_id).all()):
            finished = True
            break
    return generated[0].tolist(), finished


# ---------------------------------------------------------------------------
# Draft 模型
# ---------------------------------------------------------------------------


@dataclass
class DraftConfig:
    """draft 模型配置。

    mode: "reuse" = 目标模型第 0 层 + 目标 norm/head 权重复用；
          "random" = 同形状独立随机初始化（draft 质量下界）。
    hidden_path: W_proj 的 hidden 半边初始化——
          "zero"     = [I | 0]，draft 只用 embedding 路径（默认，无 hidden
                       条件，也就无需维护滞后 seed）；
          "identity" = [0 | I]，draft 条件于目标模型 backbone hidden
                       （最接近训练后 MTP 的条件形态；离线无训练时 seed
                       hidden 滞后一个位置，见 SpeculativeDecoder 声明）。
    seed: random 模式的初始化种子。
    """

    mode: str = "reuse"  # "reuse" | "random"
    hidden_path: str = "zero"  # "zero" | "identity"
    seed: int = 20271121


class MTPDraftModel(nn.Module):
    """单层 Qwen3 draft + MTP 头（DeepSeek-V3 MTP 结构的未训练等价物）。

    未训练代替品清单（全部显式声明，EXPERIMENT_LOG 同步记录）：
    - EHNorm：复用目标模型 final norm 权重（训练后 MTP 有独立 EHNorm）；
    - HHNorm：weight=1 的 RMSNorm（未训练）；
    - W_proj：[I|0] 或 [0|I] 块初始化（训练后 MTP 是可学习的 concat 投影）；
    - FinalNorm：reuse 模式复用目标 final norm；random 模式 weight=1；
    - 共享 LM Head：tied Embedding（这一点与 DeepSeek MTP 的"共享 head"一致）。
    """

    def __init__(self, target_model: Qwen3ForCausalLM, config: DraftConfig):
        super().__init__()
        if config.mode not in ("reuse", "random"):
            raise ValueError("DraftConfig.mode 必须是 reuse 或 random")
        if config.hidden_path not in ("zero", "identity"):
            raise ValueError("DraftConfig.hidden_path 必须是 zero 或 identity")
        self.config = config
        target_config: Qwen3Config = target_model.config

        # 单层 Transformer：reuse = 目标第 0 层权重；random = 同形状随机。
        # 两者都是同一个 Qwen3DecoderLayer 类，唯一差异是权重来源。
        if config.mode == "reuse":
            self.layer = copy.deepcopy(target_model.model.layers[0])
        else:
            generator_state = torch.random.get_rng_state()
            torch.manual_seed(config.seed)
            try:
                self.layer = Qwen3DecoderLayer(target_config)
            finally:
                torch.random.set_rng_state(generator_state)

        self.embed_tokens = target_model.model.embed_tokens  # 共享（只读）
        self.eh_norm = RMSNorm(target_config.hidden_size, target_config.rms_norm_eps)
        self.hh_norm = RMSNorm(target_config.hidden_size, target_config.rms_norm_eps)
        self.proj = nn.Linear(
            2 * target_config.hidden_size, target_config.hidden_size, bias=False
        )
        self.final_norm = RMSNorm(target_config.hidden_size, target_config.rms_norm_eps)
        # RoPE 与目标模型同款（inv_freq 是 buffer，引用目标对象即可）。
        self.rotary_emb = target_model.model.rotary_emb
        # 共享目标模型 tied LM Head 权重（多 Token 预测头的线性投影；不复制）。
        self.lm_head_weight = target_model.lm_head.weight

        with torch.no_grad():
            if config.mode == "reuse":
                self.eh_norm.weight.copy_(target_model.model.norm.weight)
                self.final_norm.weight.copy_(target_model.model.norm.weight)
            else:
                self.eh_norm.weight.fill_(1.0)
                self.final_norm.weight.fill_(1.0)
            eye = torch.eye(target_config.hidden_size)
            if config.hidden_path == "zero":
                # concat([emb, h]) @ [I | 0]^T = emb —— hidden 路径置零。
                self.proj.weight.copy_(torch.cat([eye, torch.zeros_like(eye)], dim=1))
            else:
                # concat([emb, h]) @ [0 | I]^T = h —— 条件于目标 hidden。
                self.proj.weight.copy_(torch.cat([torch.zeros_like(eye), eye], dim=1))

    @property
    def needs_seed_hidden(self) -> bool:
        return self.config.hidden_path == "identity"

    @torch.no_grad()
    def forward_hidden(self, token_ids: list[int], seed_hidden: torch.Tensor | None):
        """一步 draft：整段序列过单层，返回 (next_token, 最后位置 layer 输出)。

        seed_hidden：identity 模式下与最后一个 token 的 embedding 组合，
        形状 [1, 1, hidden]；zero 模式忽略（可为 None）。
        """
        device = next(self.layer.parameters()).device
        dtype = next(self.layer.parameters()).dtype
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        embeddings = self.embed_tokens(ids)  # [1, n, D]

        if self.config.hidden_path == "identity":
            hidden_part = self.hh_norm(seed_hidden.to(dtype)).expand(
                embeddings.shape[0], embeddings.shape[1], -1
            )
        else:
            hidden_part = torch.zeros_like(embeddings)
        combined = self.proj(
            torch.cat((self.eh_norm(embeddings).to(dtype), hidden_part), dim=-1)
        )
        sequence_length = ids.shape[1]
        position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
        cosine, sine = self.rotary_emb(position_ids, dtype)
        smallest = torch.finfo(dtype).min
        causal_mask = torch.triu(
            torch.full((sequence_length, sequence_length), smallest, dtype=dtype, device=device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        hidden = self.layer(combined, cosine, sine, causal_mask)  # [1, n, D]
        last_hidden = hidden[:, -1, :]
        logits = torch.nn.functional.linear(
            self.final_norm(last_hidden).to(dtype), self.lm_head_weight
        )
        next_token = int(logits[0].argmax().item())
        return next_token, last_hidden

    @torch.no_grad()
    def draft(
        self,
        token_ids: list[int],
        k: int,
        seed_hidden: torch.Tensor | None = None,
    ) -> list[int]:
        """自回归展开 k 个候选 token。

        链式条件（DeepSeek MTP 语义）：第 j 步输入 = 上一步产出的 token，
        hidden 条件 = 上一步 draft 层输出（j=0 时为目标模型的 seed hidden）。
        """
        if k < 1:
            raise ValueError("draft 长度 k 必须 >= 1")
        working = list(token_ids)
        drafted: list[int] = []
        step_hidden = seed_hidden  # identity 模式 j=0 的条件
        for _ in range(k):
            next_token, step_hidden = self.forward_hidden(working, step_hidden)
            drafted.append(next_token)
            working.append(next_token)
        return drafted

    def bind_lm_head(self, target_model: Qwen3ForCausalLM) -> None:
        """共享目标模型 tied LM Head 权重（不复制）。已由 __init__ 完成；
        保留该方法是给外部脚本显式声明共享关系的可读性入口。"""
        self.lm_head_weight = target_model.lm_head.weight


class CopyLastDraft:
    """恒等复制上一个 token 的 draft（无模型启发式基线）。"""

    needs_seed_hidden = False

    @torch.no_grad()
    def draft(self, token_ids: list[int], k: int, seed_hidden=None) -> list[int]:
        if k < 1:
            raise ValueError("draft 长度 k 必须 >= 1")
        return [token_ids[-1]] * k


class PromptLookupDraft:
    """prompt-lookup draft（无模型启发式的上界形态，Saxena 2023 / vLLM lookup）。

    机制：在上下文中查找"最后 n-gram 的历史出现位置"，把其后连续 k 个 token
    抄下来当候选（n=1 单 hit 兜底 n=2 双 hit）。文本重复性强（代码、模板句、
    重复短语）时接受率高；无重复时退化为 copy-last。它是"接受率曲线首位置
    高"形态的训练自由对照——MTP 训练想逼近的正是这种"从上下文抄"的能力。

    lookup_window=0 表示搜索全上下文；正数表示只搜最后 W 个 token
    （prompt-lookup 原版为 prompt 区间，这里全上下文更通用）。
    """

    needs_seed_hidden = False

    def __init__(self, ngram: int = 1):
        if ngram not in (1, 2):
            raise ValueError("ngram 仅支持 1 或 2")
        self.ngram = ngram

    @torch.no_grad()
    def draft(self, token_ids: list[int], k: int, seed_hidden=None) -> list[int]:
        if k < 1:
            raise ValueError("draft 长度 k 必须 >= 1")
        n = len(token_ids)
        if n <= self.ngram:
            return [token_ids[-1]] * k
        query = tuple(token_ids[-self.ngram :])
        best: tuple[int, ...] = ()
        for start in range(n - self.ngram):
            if tuple(token_ids[start : start + self.ngram]) == query:
                candidate = tuple(token_ids[start + self.ngram : start + self.ngram + k])
                if len(candidate) > len(best):
                    best = candidate
        if not best:
            return [token_ids[-1]] * k
        drafted = list(best)
        while len(drafted) < k:  # 候选不足 k 个时用已抄内容续抄
            drafted = drafted + drafted[: k - len(drafted)]
        return drafted[:k]


# ---------------------------------------------------------------------------
# 投机解码循环（greedy 接受/拒绝）
# ---------------------------------------------------------------------------


@dataclass
class SpecStats:
    """一轮投机解码的统计容器。

    proposed[j]  = 位置 j 上被起草的 draft token 数（每个 iteration 贡献
                   min(k, 剩余预算) 个）；
    evaluated[j] = 接受游标到达位置 j 的次数（即前缀 0..j-1 全部接受的
                   次数）——条件接受率的分母；
    accepted[j]  = 位置 j 被接受的次数；
    iteration_accepts = 每轮接受数的直方（{接受数: 轮数}）。
    """

    iterations: int = 0
    proposed: list[int] = field(default_factory=list)
    evaluated: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)
    iteration_accepts: dict[str, int] = field(default_factory=dict)
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    emitted_tokens: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0

    def note_iteration(self, k: int, accepts: int) -> None:
        """accepts = 本轮被接受的 draft 前缀长度。

        游标语义：位置 0 总被评估；前缀全接受（accepts=k）时游标走到 k，
        所有位置都被评估；中途拒绝于位置 accepts 时，位置 0..accepts 被评估。
        统一条件：position <= accepts（在 0..k-1 循环内）。
        """
        self.iterations += 1
        self.drafted_tokens += k
        self.accepted_tokens += accepts
        key = str(accepts)
        self.iteration_accepts[key] = self.iteration_accepts.get(key, 0) + 1
        for position in range(k):
            if len(self.proposed) <= position:
                self.proposed.append(0)
                self.evaluated.append(0)
                self.accepted.append(0)
            self.proposed[position] += 1
            if position <= accepts:
                self.evaluated[position] += 1
            if position < accepts:
                self.accepted[position] += 1


class SpeculativeDecoder:
    """目标模型 + draft 模型的 greedy 投机解码循环。

    分布不变性（greedy）归纳证明骨架（实验 1 用真值逐 token 实证）：
    设当前已发射前缀 S（与贪心解码相同），目标模型在位置 len(S)-1 的
    argmax 为 t。本轮 verify 的第 0 行 logits 恰是该位置的 logits：
    - 若 draft[0] == t，接受 → 发射 t；
    - 若不等，纠正 token = t → 发射 t。
    即无论 draft 提议什么，本轮发射的第一个 token 都等于 t = 贪心的下一个
    token。归纳下去，投机解码输出流 ≡ 贪心输出流；draft 只决定"一次 verify
    摊到几个 token"，从不决定 token 内容。
    """

    def __init__(
        self,
        target_model: Qwen3ForCausalLM,
        draft_model: MTPDraftModel | CopyLastDraft,
        eos_token_id: int | None = None,
    ):
        self.target = target_model
        self.draft = draft_model
        self.eos_token_id = eos_token_id

    @torch.no_grad()
    def verify_and_accept(
        self, token_ids: list[int], draft_ids: list[int]
    ) -> tuple[list[int], int, dict[str, Any]]:
        """一次并行 verify + 逐位置接受游走。

        verify = 一次目标模型前向，覆盖 len(token_ids)+len(draft_ids) 个位置
        （k 个候选位置被一批算完——"verify 必须并行"的结构来源：k+1 个位置的
        logits 来自同一次前向，而不是 k 次串行前向）。
        返回 (emitted_ids, accepted_count, info)。emitted 恒为 accepted_count+1
        个 token（全接受 → 奖励 token；中途拒绝 → 纠正 token）。
        """
        n = len(token_ids)
        k = len(draft_ids)
        if k < 1:
            raise ValueError("verify 需要 >= 1 个 draft token")
        device = _model_device(self.target)
        full_ids = torch.tensor([token_ids + draft_ids], dtype=torch.long, device=device)
        hidden = backbone_hidden(self.target, full_ids)  # [1, n+k, D]（final-norm 后）
        # 位置 n-1 .. n+k-1 的 logits：第 j 行预测 draft_ids[j] 该不该是
        # 下一个 token；第 k 行（位置 n+k-1）在全接受时给出奖励 token。
        logits = logits_at_rows(self.target, hidden, slice(n - 1, n + k))
        target_argmaxes = logits[0].argmax(dim=-1).tolist()  # 长度 k+1

        accepted = 0
        corrective: int | None = None
        for position in range(k):
            if draft_ids[position] == target_argmaxes[position]:
                accepted += 1
            else:
                corrective = target_argmaxes[position]
                break

        if corrective is not None:
            emitted = draft_ids[:accepted] + [corrective]
            outcome = "reject_at_" + str(accepted)
        else:
            emitted = draft_ids[:] + [target_argmaxes[k]]
            outcome = "accept_all"
        info = {
            "outcome": outcome,
            "accepted": accepted,
            "target_argmaxes": target_argmaxes,
            "draft_ids": list(draft_ids),
            # 本轮可复用的 backbone hidden（final-norm 后，[n+k, D]）。
            "verify_hidden": hidden[0],
            "verify_length": n + k,
        }
        return emitted, accepted, info

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        k: int,
    ) -> tuple[list[int], bool, SpecStats]:
        """投机解码主循环。返回 (generated_ids, finished_by_eos, stats)。"""
        if k < 1:
            raise ValueError("k 必须 >= 1")
        ids = list(prompt_ids)
        stats = SpecStats()
        finished = False
        emitted_total = 0
        seed_hidden: torch.Tensor | None = None

        if self.draft.needs_seed_hidden:
            # 首 iteration 的 seed：prompt 末位 backbone hidden（精确），
            # 形状 [1, 1, hidden]。
            prompt_tensor = torch.tensor(
                [ids], dtype=torch.long, device=_model_device(self.target)
            )
            seed_hidden = backbone_hidden(self.target, prompt_tensor)[0, -1, :].view(1, 1, -1)

        while emitted_total < max_new_tokens:
            remaining = max_new_tokens - emitted_total
            step_k = min(k, remaining)
            clock = time.perf_counter
            draft_start = clock()
            draft_ids = self.draft.draft(ids, step_k, seed_hidden)
            draft_ms = (clock() - draft_start) * 1000.0
            verify_start = clock()
            emitted, accepted, info = self.verify_and_accept(ids, draft_ids)
            verify_ms = (clock() - verify_start) * 1000.0
            stats.draft_time_ms += draft_ms
            stats.verify_time_ms += verify_ms

            # EOS 截断必须作用于"本轮发射序列"：贪心在 EOS 即停；投机一轮可能
            # 接受 EOS 之后仍继续接受更多 token（draft 从模板上下文抄到
            # EOS+后继时就会出现）——不改分布要求输出在第一个 EOS 处截断。
            # （smoke_test 的 real-model 检查覆盖此路径：lookup draft 曾实测
            # 在 EOS 后多发射 2 个 token，修复点即此处。）
            finished = False
            if self.eos_token_id is not None and self.eos_token_id in emitted:
                emitted = emitted[: emitted.index(self.eos_token_id) + 1]
                finished = True

            # 奖励/纠正 token 可能超出剩余预算：截断（标准做法，记入统计）。
            emitted = emitted[:remaining]

            # identity 模式的下一步 seed：复用本轮 verify 的 hidden，取
            # "最后一个 hidden 已被精确计算的位置"。最后发射 token 的
            # hidden 尚不存在（需额外一次前向），离线实现取倒数第二个
            # 位置——声明为未训练 MTP 的近似之一（EXPERIMENT_LOG §3）。
            if self.draft.needs_seed_hidden:
                verify_hidden = info["verify_hidden"]
                verify_length = info["verify_length"]
                seed_index = min(len(ids) + len(emitted) - 2, verify_length - 1)
                seed_hidden = verify_hidden[seed_index].unsqueeze(0).unsqueeze(0)

            ids.extend(emitted)
            emitted_total += len(emitted)
            stats.emitted_tokens += len(emitted)
            stats.note_iteration(len(draft_ids), accepted)

            if finished:
                break

        return ids, finished, stats


# ---------------------------------------------------------------------------
# 成本模型（理论加速比，实验 4 的标定对象）
# ---------------------------------------------------------------------------


def idealized_speedup(acceptance_alpha: float, cost_ratio_c: float, k: int) -> float:
    """(k+1)/(1+c·k) 型成本模型（含实测接受率的推广 (αk+1)/(1+ck)）。

    假设：verify 一次的成本 ≈ 目标模型单步成本 T_t（理想化：k+1 个位置与
    1 个位置同价——decode 是 memory-dominated，读一遍权重即可验证任意多
    候选，ep21 结论）；draft 一步成本 = c·T_t（c = draft/目标成本比）。
    每轮产出 αk+1 个 token（α = 每 draft 位置平均接受率）。
    α=1 时退化为任务给定的 (k+1)/(1+ck)；k→∞ 时上界 = min(α,1)/c。
    """
    if k < 1:
        raise ValueError("k 必须 >= 1")
    return (acceptance_alpha * k + 1) / (cost_ratio_c * k + 1)


def calibrated_speedup(
    acceptance_alpha: float,
    draft_step_ms: float,
    verify_ms: float,
    k: int,
) -> float:
    """用实测前向时间标定的成本模型（不做"verify=单步同价"的理想化）。

    每轮成本 = k·T_draft + T_verify(n+k)，产出 αk+1 token；
    基线 = T_target(n)（贪心单步，实测）。
    """
    if k < 1:
        raise ValueError("k 必须 >= 1")
    iteration_ms = k * draft_step_ms + verify_ms
    return acceptance_alpha * k / iteration_ms + 1.0 / iteration_ms


def summarize_spec_stats(stats: SpecStats) -> dict[str, Any]:
    """把 SpecStats 折算成可写入 results JSON 的字典。"""
    per_position = []
    for position, proposed in enumerate(stats.proposed):
        evaluated = stats.evaluated[position]
        accepted = stats.accepted[position]
        per_position.append(
            {
                "position": position + 1,  # 1-based draft 槽位
                "proposed": proposed,
                "evaluated": evaluated,
                "accepted": accepted,
                "accept_rate_over_proposed": round(accepted / proposed, 6) if proposed else None,
                "accept_rate_over_evaluated": round(accepted / evaluated, 6) if evaluated else None,
            }
        )
    alpha = (
        stats.accepted_tokens / stats.drafted_tokens if stats.drafted_tokens else 0.0
    )
    return {
        "iterations": stats.iterations,
        "drafted_tokens": stats.drafted_tokens,
        "accepted_tokens": stats.accepted_tokens,
        "emitted_tokens": stats.emitted_tokens,
        "mean_accepted_per_iteration": round(stats.accepted_tokens / stats.iterations, 6)
        if stats.iterations
        else 0.0,
        "alpha_over_drafted": round(alpha, 6),
        "iteration_accept_histogram": stats.iteration_accepts,
        "draft_time_ms": round(stats.draft_time_ms, 2),
        "verify_time_ms": round(stats.verify_time_ms, 2),
        "per_position": per_position,
    }


def build_draft(
    target_model: Qwen3ForCausalLM,
    mode: str,
    hidden_path: str = "zero",
    seed: int = 20271121,
) -> MTPDraftModel | CopyLastDraft | PromptLookupDraft:
    """按名字构造 draft：reuse / reuse_hidden / random / copy / lookup / lookup2。

    reuse_hidden 是 mode="reuse" + hidden_path="identity" 的别名（条件于
    目标模型 hidden 的未训练 MTP 形态）；lookup/lookup2 是 prompt-lookup
    启发式（1/2 gram），作为"接受率首位置高"形态的训练自由对照。
    """
    if mode == "copy":
        return CopyLastDraft()
    if mode == "lookup":
        return PromptLookupDraft(ngram=1)
    if mode == "lookup2":
        return PromptLookupDraft(ngram=2)
    if mode == "reuse_hidden":
        config = DraftConfig(mode="reuse", hidden_path="identity")
    elif mode == "reuse":
        config = DraftConfig(mode="reuse", hidden_path=hidden_path)
    elif mode == "random":
        config = DraftConfig(mode="random", hidden_path=hidden_path, seed=seed)
    else:
        raise ValueError("未知 draft 模式: " + mode)
    draft_model = MTPDraftModel(target_model, config)
    draft_model.bind_lm_head(target_model)
    return draft_model
