"""Paged KV Cache 的物理 Block Pool、Block Table 与参考 Attention 路径。

每个物理 Block 同时保存模型全部层的 K/V。请求只保存逻辑 Block 到物理
Block ID 的映射；请求结束后物理块回到空闲池，不移动其他请求的数据。
"""

import heapq
import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from qwen3_model import apply_rotary_embedding


class PagedKVCache:
    def __init__(self, config, block_size, max_blocks, device, dtype):
        if block_size < 1:
            raise ValueError("block_size 必须大于 0")
        if max_blocks < 1:
            raise ValueError("max_blocks 必须大于 0")
        self.config = config
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.max_blocks = max_blocks
        self.blocks = torch.empty(
            (config.num_hidden_layers, 2, max_blocks,
             config.num_key_value_heads, block_size, config.head_dim),
            dtype=dtype,
            device=device,
        )
        self.free_block_ids = list(range(max_blocks))
        self.block_tables = {}
        self.sequence_lengths = {}
        self.peak_used_blocks = 0
        self.allocation_count = 0
        self.reuse_count = 0
        self.release_count = 0
        self.ever_used_block_ids = set()

    @property
    def block_shape(self):
        return (
            self.config.num_hidden_layers,
            2,
            self.config.num_key_value_heads,
            self.block_size,
            self.config.head_dim,
        )

    @property
    def bytes_per_block(self):
        elements = math.prod(self.block_shape)
        return elements * torch.empty((), dtype=self.dtype).element_size()

    @property
    def used_block_count(self):
        return sum(len(table) for table in self.block_tables.values())

    @property
    def live_token_count(self):
        return sum(self.sequence_lengths.values())

    @property
    def allocated_token_slots(self):
        return self.used_block_count * self.block_size

    @property
    def live_cache_bytes(self):
        return self.used_block_count * self.bytes_per_block

    @property
    def pool_bytes(self):
        return self.max_blocks * self.bytes_per_block

    @property
    def utilization(self):
        slots = self.allocated_token_slots
        return self.live_token_count / slots if slots else 0.0

    def _new_block(self):
        if not self.free_block_ids:
            raise RuntimeError("Paged KV Cache 物理 Block 已耗尽")
        block_id = heapq.heappop(self.free_block_ids)
        if block_id in self.ever_used_block_ids:
            self.reuse_count += 1
        else:
            self.ever_used_block_ids.add(block_id)
            self.allocation_count += 1
        return block_id

    def _ensure_blocks(self, request_id, token_count):
        required = (token_count + self.block_size - 1) // self.block_size
        table = self.block_tables[request_id]
        while len(table) < required:
            table.append(self._new_block())
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_block_count)

    def store_prefill(self, request_id, request_cache):
        if request_id in self.block_tables:
            raise ValueError("请求已经存在于 Block Pool: " + request_id)
        if not request_cache:
            raise ValueError("request_cache 不能为空")
        length = request_cache[0][0].shape[2]
        if length < 1:
            raise ValueError("Prefill Cache 长度必须大于 0")
        self.block_tables[request_id] = []
        self.sequence_lengths[request_id] = length
        self._ensure_blocks(request_id, length)
        table = self.block_tables[request_id]
        for layer_index, (key, value) in enumerate(request_cache):
            if key.shape != value.shape or key.shape[0] != 1:
                raise ValueError("单请求 Prefill Cache 形状错误")
            for logical_block, block_id in enumerate(table):
                start = logical_block * self.block_size
                end = min(start + self.block_size, length)
                count = end - start
                self.blocks[layer_index, 0, block_id, :, :count, :].copy_(
                    key[0, :, start:end, :]
                )
                self.blocks[layer_index, 1, block_id, :, :count, :].copy_(
                    value[0, :, start:end, :]
                )

    def prepare_append(self, request_ids):
        positions = {}
        for request_id in request_ids:
            if request_id not in self.block_tables:
                raise KeyError("请求没有 Block Table: " + request_id)
            position = self.sequence_lengths[request_id]
            self._ensure_blocks(request_id, position + 1)
            positions[request_id] = position
        return positions

    def write_token(self, layer_index, request_id, key, value, position):
        if tuple(key.shape) != (
            self.config.num_key_value_heads, 1, self.config.head_dim
        ) or key.shape != value.shape:
            raise ValueError("Decode K/V 必须是 [num_kv_heads, 1, head_dim]")
        block_id = self.block_tables[request_id][position // self.block_size]
        offset = position % self.block_size
        self.blocks[layer_index, 0, block_id, :, offset, :].copy_(key[:, 0, :])
        self.blocks[layer_index, 1, block_id, :, offset, :].copy_(value[:, 0, :])

    def finish_append(self, request_ids):
        for request_id in request_ids:
            self.sequence_lengths[request_id] += 1

    def read_layer(self, layer_index, request_ids, logical_length):
        block_count = (logical_length + self.block_size - 1) // self.block_size
        block_ids = torch.tensor(
            [self.block_tables[request_id][:block_count] for request_id in request_ids],
            dtype=torch.long,
            device=self.device,
        )
        key = self.blocks[layer_index, 0, block_ids]
        value = self.blocks[layer_index, 1, block_ids]
        # [batch, blocks, kv_heads, block_size, head_dim]
        #   -> [batch, kv_heads, logical_sequence, head_dim]
        key = key.permute(0, 2, 1, 3, 4).reshape(
            len(request_ids), self.config.num_key_value_heads,
            block_count * self.block_size, self.config.head_dim,
        )[:, :, :logical_length, :]
        value = value.permute(0, 2, 1, 3, 4).reshape(
            len(request_ids), self.config.num_key_value_heads,
            block_count * self.block_size, self.config.head_dim,
        )[:, :, :logical_length, :]
        return key, value

    def block_id_tensor(self, request_ids, logical_length):
        block_count = (logical_length + self.block_size - 1) // self.block_size
        return torch.tensor(
            [self.block_tables[request_id][:block_count] for request_id in request_ids],
            dtype=torch.long,
            device=self.device,
        )

    def read_blocks(self, layer_index, block_ids):
        return (
            self.blocks[layer_index, 0, block_ids],
            self.blocks[layer_index, 1, block_ids],
        )

    def release(self, request_id):
        table = self.block_tables.pop(request_id)
        self.sequence_lengths.pop(request_id)
        for block_id in table:
            heapq.heappush(self.free_block_ids, block_id)
        self.release_count += len(table)

    def snapshot(self):
        return {
            "block_size": self.block_size,
            "pool_blocks": self.max_blocks,
            "used_blocks": self.used_block_count,
            "free_blocks": len(self.free_block_ids),
            "peak_used_blocks": self.peak_used_blocks,
            "live_tokens": self.live_token_count,
            "allocated_token_slots": self.allocated_token_slots,
            "cache_utilization": self.utilization,
            "live_cache_bytes": self.live_cache_bytes,
            "pool_bytes": self.pool_bytes,
            "allocation_count": self.allocation_count,
            "reuse_count": self.reuse_count,
            "release_count": self.release_count,
            "block_tables": {
                request_id: list(table)
                for request_id, table in self.block_tables.items()
            },
        }


def paged_decode_forward(model, input_ids, position_ids, slot_request_ids, cache):
    """只对相同有效上下文长度的请求分组，不补齐到全局最长 Cache。"""

    if input_ids.shape[1] != 1:
        raise ValueError("Paged Decode 每次只接受一个新 Token")
    active = [
        (slot, request_id)
        for slot, request_id in enumerate(slot_request_ids)
        if request_id is not None
    ]
    if not active:
        raise ValueError("Paged Decode 至少需要一个活跃请求")
    request_ids = [request_id for _, request_id in active]
    append_positions = cache.prepare_append(request_ids)

    groups = defaultdict(list)
    for slot, request_id in active:
        groups[append_positions[request_id] + 1].append((slot, request_id))
    group_plans = []
    for logical_length, members in groups.items():
        rows = torch.tensor(
            [slot for slot, _ in members], dtype=torch.long, device=input_ids.device
        )
        member_ids = [request_id for _, request_id in members]
        group_plans.append((
            logical_length,
            rows,
            cache.block_id_tensor(member_ids, logical_length),
        ))

    hidden_states = model.model.embed_tokens(input_ids)
    cosine, sine = model.model.rotary_emb(position_ids, hidden_states.dtype)
    visited_token_slots = 0

    for layer_index, layer in enumerate(model.model.layers):
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        attention = layer.self_attn
        batch_size = normalized.shape[0]
        query = attention.q_norm(
            attention.q_proj(normalized).view(
                batch_size, 1, attention.num_attention_heads,
                attention.head_dim,
            )
        ).transpose(1, 2)
        key = attention.k_norm(
            attention.k_proj(normalized).view(
                batch_size, 1, attention.num_key_value_heads,
                attention.head_dim,
            )
        ).transpose(1, 2)
        value = attention.v_proj(normalized).view(
            batch_size, 1, attention.num_key_value_heads,
            attention.head_dim,
        ).transpose(1, 2)
        query, key = apply_rotary_embedding(query, key, cosine, sine)

        for slot, request_id in active:
            position = append_positions[request_id]
            cache.write_token(
                layer_index, request_id, key[slot], value[slot], position
            )

        attention_output = torch.zeros_like(query)
        for logical_length, rows, block_ids in group_plans:
            group_key, group_value = cache.read_blocks(layer_index, block_ids)
            # [batch, blocks, kv_heads, block_size, head_dim]
            group_key = group_key.repeat_interleave(
                attention.num_key_value_groups, dim=2
            )
            group_value = group_value.repeat_interleave(
                attention.num_key_value_groups, dim=2
            )
            group_query = query[rows]
            scores = torch.einsum(
                "bhqd,bnhsd->bhqns", group_query, group_key
            )
            scores = scores / math.sqrt(attention.head_dim)
            tail = logical_length % cache.block_size
            if tail:
                scores[..., -1, tail:] = torch.finfo(scores.dtype).min
            flat_scores = scores.flatten(start_dim=-2)
            probabilities = F.softmax(
                flat_scores, dim=-1, dtype=torch.float32
            ).to(group_query.dtype).view_as(scores)
            attention_output[rows] = torch.einsum(
                "bhqns,bnhsd->bhqd", probabilities, group_value
            )
            if layer_index == 0:
                visited_token_slots += rows.numel() * logical_length

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, 1, -1
        )
        hidden_states = residual + attention.o_proj(attention_output)
        residual = hidden_states
        hidden_states = residual + layer.mlp(
            layer.post_attention_layernorm(hidden_states)
        )

    cache.finish_append(request_ids)
    hidden_states = model.model.norm(hidden_states)
    return model.lm_head(hidden_states), visited_token_slots
