"""第 09 期独立实现的 Paged KV Cache 与块级 Prefix Cache。

每个物理 Block 同时保存模型全部层的 K/V。请求只保存逻辑 Block 到物理
Block ID 的映射；请求结束后物理块回到空闲池，不移动其他请求的数据。
"""

import hashlib
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass

import torch

from qwen3_model import (
    apply_rotary_embedding,
    dense_attention_forward,
    repeat_key_value,
)


@dataclass
class PrefixEntry:
    digest: bytes
    parent_digest: bytes
    token_ids: tuple
    block_id: int
    last_access: int


class PagedKVCache:
    def __init__(
        self, config, block_size, max_blocks, device, dtype,
        prefix_cache_enabled=False, prefix_cache_capacity_blocks=0,
        model_namespace="Qwen/Qwen3-0.6B",
    ):
        if block_size < 1:
            raise ValueError("block_size 必须大于 0")
        if max_blocks < 1:
            raise ValueError("max_blocks 必须大于 0")
        self.config = config
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.max_blocks = max_blocks
        self.prefix_cache_enabled = bool(prefix_cache_enabled)
        self.prefix_cache_capacity_blocks = int(prefix_cache_capacity_blocks)
        if self.prefix_cache_capacity_blocks < 0:
            raise ValueError("prefix_cache_capacity_blocks 不能小于 0")
        self.model_namespace = (
            "%s|dtype=%s|block_size=%d" % (model_namespace, dtype, block_size)
        ).encode("utf-8")
        self.blocks = torch.empty(
            (config.num_hidden_layers, 2, max_blocks,
             config.num_key_value_heads, block_size, config.head_dim),
            dtype=dtype,
            device=device,
        )
        self.free_block_ids = list(range(max_blocks))
        self.block_tables = {}
        self.sequence_lengths = {}
        self.block_ref_counts = [0] * max_blocks
        self.prefix_entries = {}
        self.block_to_digest = {}
        self.access_clock = 0
        self.prefix_lookup_count = 0
        self.prefix_hit_tokens = 0
        self.prefix_publish_count = 0
        self.prefix_eviction_count = 0
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
        return self.max_blocks - len(self.free_block_ids)

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

    @property
    def cached_block_count(self):
        return len(self.prefix_entries)

    def _block_digest(self, parent_digest, token_ids):
        hasher = hashlib.sha256()
        hasher.update(self.model_namespace)
        hasher.update(parent_digest)
        for token_id in token_ids:
            hasher.update(int(token_id).to_bytes(8, "little", signed=True))
        return hasher.digest()

    def _free_if_unowned(self, block_id):
        if self.block_ref_counts[block_id] == 0 and block_id not in self.block_to_digest:
            heapq.heappush(self.free_block_ids, block_id)

    def _evict_one(self):
        if not self.prefix_entries:
            return False
        parents = {entry.parent_digest for entry in self.prefix_entries.values()}
        leaves = [
            entry for digest, entry in self.prefix_entries.items()
            if digest not in parents
        ]
        victim = min(leaves, key=lambda entry: entry.last_access)
        self.prefix_entries.pop(victim.digest)
        self.block_to_digest.pop(victim.block_id)
        self.prefix_eviction_count += 1
        self._free_if_unowned(victim.block_id)
        return True

    def _new_block(self):
        while not self.free_block_ids and self._evict_one():
            pass
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
            block_id = self._new_block()
            table.append(block_id)
            self.block_ref_counts[block_id] += 1
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_block_count)

    def attach_prefix(self, request_id, token_ids):
        """匹配从 Prompt 开头开始的完整块，并为请求增加引用。

        最多匹配 ``prompt_length - 1`` 个 Token，确保最后一个 Prompt Token
        仍会前向计算并产生首 Token Logits。
        """
        if request_id in self.block_tables:
            raise ValueError("请求已经存在于 Block Pool: " + request_id)
        self.block_tables[request_id] = []
        self.sequence_lengths[request_id] = 0
        self.prefix_lookup_count += 1
        if not self.prefix_cache_enabled:
            return 0
        reusable_blocks = max(0, (len(token_ids) - 1) // self.block_size)
        parent = b""
        for logical_block in range(reusable_blocks):
            start = logical_block * self.block_size
            block_tokens = tuple(int(token) for token in token_ids[start:start + self.block_size])
            digest = self._block_digest(parent, block_tokens)
            entry = self.prefix_entries.get(digest)
            if (
                entry is None or entry.parent_digest != parent
                or entry.token_ids != block_tokens
            ):
                break
            self.access_clock += 1
            entry.last_access = self.access_clock
            self.block_tables[request_id].append(entry.block_id)
            self.block_ref_counts[entry.block_id] += 1
            parent = digest
        hit_tokens = len(self.block_tables[request_id]) * self.block_size
        self.sequence_lengths[request_id] = hit_tokens
        self.prefix_hit_tokens += hit_tokens
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_block_count)
        return hit_tokens

    def publish_prompt(self, request_id, token_ids):
        """将已经写满的 Prompt Block 发布为不可变缓存条目。"""
        if not self.prefix_cache_enabled:
            return 0
        reusable_blocks = max(0, (len(token_ids) - 1) // self.block_size)
        if reusable_blocks > len(self.block_tables[request_id]):
            raise RuntimeError("Prompt Block Table 不完整，无法发布 Prefix Cache")
        published = 0
        parent = b""
        for logical_block in range(reusable_blocks):
            start = logical_block * self.block_size
            block_tokens = tuple(int(token) for token in token_ids[start:start + self.block_size])
            digest = self._block_digest(parent, block_tokens)
            block_id = self.block_tables[request_id][logical_block]
            existing = self.prefix_entries.get(digest)
            if existing is None:
                if self.prefix_cache_capacity_blocks == 0:
                    break
                # 当前链的父块还在索引中时，容量不足就停止发布更深的块，
                # 以免淘汰父块后留下无法从 Prompt 开头命中的孤儿条目。
                if (
                    len(self.prefix_entries) >= self.prefix_cache_capacity_blocks
                    and parent in self.prefix_entries
                ):
                    break
                while len(self.prefix_entries) >= self.prefix_cache_capacity_blocks:
                    if not self._evict_one():
                        break
                self.access_clock += 1
                entry = PrefixEntry(
                    digest, parent, block_tokens, block_id, self.access_clock
                )
                self.prefix_entries[digest] = entry
                self.block_to_digest[block_id] = digest
                self.prefix_publish_count += 1
                published += 1
            elif (
                existing.parent_digest != parent or existing.token_ids != block_tokens
            ):
                raise RuntimeError("Prefix Cache 哈希碰撞未通过确定性校验")
            parent = digest
        return published

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

    def dense_request_cache(self, request_id):
        """按逻辑顺序重建单请求 Cache，供教学版增量 Prefill 使用。"""
        if request_id not in self.block_tables:
            return None
        length = self.sequence_lengths[request_id]
        return [
            self.read_layer(layer_index, [request_id], length)
            for layer_index in range(self.config.num_hidden_layers)
        ]

    def append_prefill(self, request_id, request_cache):
        """把只包含当前 Prompt Chunk 的 K/V 追加到 Block Table。"""
        if not request_cache:
            raise ValueError("request_cache 不能为空")
        chunk_length = request_cache[0][0].shape[2]
        if chunk_length < 1:
            raise ValueError("Prefill Chunk 长度必须大于 0")
        if request_id not in self.block_tables:
            self.block_tables[request_id] = []
            self.sequence_lengths[request_id] = 0
        start = self.sequence_lengths[request_id]
        end = start + chunk_length
        self._ensure_blocks(request_id, end)
        table = self.block_tables[request_id]
        for layer_index, (key, value) in enumerate(request_cache):
            expected = (
                1, self.config.num_key_value_heads,
                chunk_length, self.config.head_dim,
            )
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError("Prefill Chunk Cache 形状错误")
            source = 0
            position = start
            while position < end:
                block_id = table[position // self.block_size]
                offset = position % self.block_size
                count = min(end - position, self.block_size - offset)
                if block_id in self.block_to_digest:
                    raise RuntimeError("不能写入已经发布的 Prefix Cache Block")
                self.blocks[
                    layer_index, 0, block_id, :, offset:offset + count, :
                ].copy_(key[0, :, source:source + count, :])
                self.blocks[
                    layer_index, 1, block_id, :, offset:offset + count, :
                ].copy_(value[0, :, source:source + count, :])
                position += count
                source += count
        self.sequence_lengths[request_id] = end

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
        if block_id in self.block_to_digest:
            raise RuntimeError("Decode 不能写入共享 Prefix Cache Block")
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
            self.block_ref_counts[block_id] -= 1
            if self.block_ref_counts[block_id] < 0:
                raise RuntimeError("物理 Block 引用计数小于 0")
            self._free_if_unowned(block_id)
        self.release_count += len(table)

    def snapshot(self):
        return {
            "block_size": self.block_size,
            "pool_blocks": self.max_blocks,
            "used_blocks": self.used_block_count,
            "free_blocks": len(self.free_block_ids),
            "cached_blocks": self.cached_block_count,
            "active_block_references": sum(self.block_ref_counts),
            "peak_used_blocks": self.peak_used_blocks,
            "live_tokens": self.live_token_count,
            "allocated_token_slots": self.allocated_token_slots,
            "cache_utilization": self.utilization,
            "live_cache_bytes": self.live_cache_bytes,
            "pool_bytes": self.pool_bytes,
            "allocation_count": self.allocation_count,
            "reuse_count": self.reuse_count,
            "release_count": self.release_count,
            "prefix_cache_enabled": self.prefix_cache_enabled,
            "prefix_cache_capacity_blocks": self.prefix_cache_capacity_blocks,
            "prefix_lookup_count": self.prefix_lookup_count,
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "prefix_publish_count": self.prefix_publish_count,
            "prefix_eviction_count": self.prefix_eviction_count,
            "block_ref_counts": list(self.block_ref_counts),
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
            group_key = group_key.permute(0, 2, 1, 3, 4).reshape(
                rows.numel(), attention.num_key_value_heads, -1,
                attention.head_dim,
            )
            group_value = group_value.permute(0, 2, 1, 3, 4).reshape(
                rows.numel(), attention.num_key_value_heads, -1,
                attention.head_dim,
            )
            group_key = repeat_key_value(
                group_key, attention.num_key_value_groups
            )
            group_value = repeat_key_value(
                group_value, attention.num_key_value_groups
            )
            group_query = query[rows]
            key_valid = torch.arange(
                group_key.shape[2], device=input_ids.device
            ).unsqueeze(0) < logical_length
            key_valid = key_valid.expand(rows.numel(), -1)
            query_valid = torch.ones(
                (rows.numel(), 1), dtype=torch.bool, device=input_ids.device
            )
            attention_output[rows] = dense_attention_forward(
                attention.attention_backend,
                group_query,
                group_key,
                group_value,
                key_valid,
                query_valid,
                logical_length - 1,
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
