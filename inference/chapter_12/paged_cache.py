"""第 12 期 Paged KV Cache：块主序 Block Pool、CPU Pinned Pool 与换出/换入。

与第 05/07 期的层主序布局不同，本期 GPU Pool 采用块主序：
[max_blocks, num_layers, 2, kv_heads, block_size, head_dim]。
每个物理 Block 的全部层 K/V 在物理上连续，因此一个物理块可以在 GPU Pool 与
Pinned CPU Pool 之间通过一次 contiguous 拷贝完成搬移。

CPU Pinned Pool 是暂停请求的私有换出存储：只为恢复原请求服务，不做命中、
共享或跨请求复用。换出不改变请求的逻辑序列，只改变 KV 的物理驻留位置。
"""

import heapq
import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from qwen3_model import apply_rotary_embedding


class PagedKVCache:
    """块主序 GPU 物理 Block Pool 与请求 Block Table。"""

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
        # 块主序：一个物理 Block 的所有层 K/V 连续存放。
        self.blocks = torch.empty(
            (max_blocks, config.num_hidden_layers, 2,
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
    def free_block_count(self):
        return len(self.free_block_ids)

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
        if len(table) < required and self.free_block_count < required - len(table):
            raise RuntimeError(
                "Block 不足：request=%s 需要 %d 块（现有 %d，空闲 %d，"
                "token_count=%d，pool=%d）"
                % (request_id, required, len(table), self.free_block_count,
                   token_count, self.max_blocks)
            )
        while len(table) < required:
            table.append(self._new_block())
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_block_count)

    def begin_request(self, request_id):
        """准入时创建空 Block Table，随后用 reserve_blocks 预留全部 Block。"""
        if request_id in self.block_tables:
            raise ValueError("请求已经存在于 Block Pool: " + request_id)
        self.block_tables[request_id] = []
        self.sequence_lengths[request_id] = 0

    def reserve_blocks(self, request_id, token_count):
        """按 token_count 预留物理 Block；内容仍由 Prefill Chunk 逐段写入。"""
        if request_id not in self.block_tables:
            raise KeyError("请求尚未创建 Block Table: " + request_id)
        self._ensure_blocks(request_id, token_count)

    def blocks_for_tokens(self, token_count):
        return (token_count + self.block_size - 1) // self.block_size

    def blocks_needed(self, request_id):
        table = self.block_tables.get(request_id)
        return 0 if table is None else len(table)

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
                self.blocks[
                    block_id, layer_index, 0, :, offset:offset + count, :
                ].copy_(key[0, :, source:source + count, :])
                self.blocks[
                    block_id, layer_index, 1, :, offset:offset + count, :
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
        offset = position % self.block_size
        self.blocks[block_id, layer_index, 0, :, offset, :].copy_(key[:, 0, :])
        self.blocks[block_id, layer_index, 1, :, offset, :].copy_(value[:, 0, :])

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
        key = self.blocks[block_ids, layer_index, 0]
        value = self.blocks[block_ids, layer_index, 1]
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
            self.blocks[block_ids, layer_index, 0],
            self.blocks[block_ids, layer_index, 1],
        )

    def release(self, request_id):
        table = self.block_tables.pop(request_id)
        self.sequence_lengths.pop(request_id)
        for block_id in table:
            heapq.heappush(self.free_block_ids, block_id)
        self.release_count += len(table)

    def snapshot(self):
        return {
            "layout": "block_major",
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


class CPUPinnedPool:
    """Pinned CPU Block Pool：暂停请求 KV 的换出目的地。

    布局与 GPU Pool 完全一致（块主序），换出/换入都是整请求、逐块的
    同步 blocking copy。CPU Block 只在请求换出期间被占用。
    """

    def __init__(self, config, block_size, max_blocks, dtype, pin_memory):
        if max_blocks < 1:
            raise ValueError("cpu max_blocks 必须大于 0")
        if pin_memory and not torch.cuda.is_available():
            raise ValueError("pin_memory=True 需要可用的 CUDA 设备")
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.pin_memory = pin_memory
        self.blocks = torch.empty(
            (max_blocks, config.num_hidden_layers, 2,
             config.num_key_value_heads, block_size, config.head_dim),
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.free_block_ids = list(range(max_blocks))
        # request_id -> 按 GPU Block Table 顺序记录的 CPU Block ID 列表
        self.swap_tables = {}
        self.swap_lengths = {}
        self.peak_used_blocks = 0
        self.swap_out_count = 0
        self.swap_in_count = 0
        self.peak_live_bytes = 0

    @property
    def bytes_per_block(self):
        return self.blocks[0].numel() * self.blocks.element_size()

    @property
    def bytes_per_token(self):
        return self.bytes_per_block // self.block_size

    def _movement_stats(self, block_count, logical_tokens):
        physical_bytes = block_count * self.bytes_per_block
        logical_bytes = logical_tokens * self.bytes_per_token
        if logical_bytes > physical_bytes:
            raise RuntimeError("逻辑 KV 字节数不能超过实际传输字节数")
        return {
            # bytes 保留为兼容字段，始终表示 PCIe 实际搬运的物理字节。
            "bytes": physical_bytes,
            "physical_bytes": physical_bytes,
            "logical_bytes": logical_bytes,
            "tail_fragment_bytes": physical_bytes - logical_bytes,
        }

    @property
    def used_block_count(self):
        return sum(len(table) for table in self.swap_tables.values())

    @property
    def free_block_count(self):
        return len(self.free_block_ids)

    @property
    def live_bytes(self):
        return self.used_block_count * self.bytes_per_block

    @property
    def pool_bytes(self):
        return self.max_blocks * self.bytes_per_block

    def blocks_needed(self, request_id):
        table = self.swap_tables.get(request_id)
        return 0 if table is None else len(table)

    def swap_out(self, gpu_cache, request_id):
        """整请求换出：逐块同步 D2H，随后归还全部 GPU Block。

        返回换出统计；计时由引擎在外层用 CUDA 同步包围。
        """
        table = list(gpu_cache.block_tables[request_id])
        length = gpu_cache.sequence_lengths[request_id]
        if request_id in self.swap_tables:
            raise ValueError("请求已在 CPU 池中: " + request_id)
        if self.free_block_count < len(table):
            raise RuntimeError(
                "CPU Pinned Pool 已耗尽：free=%d 需要=%d（%s）"
                % (self.free_block_count, len(table), request_id)
            )
        cpu_ids = [heapq.heappop(self.free_block_ids) for _ in table]
        for gpu_id, cpu_id in zip(table, cpu_ids):
            self.blocks[cpu_id].copy_(gpu_cache.blocks[gpu_id])
        # 换出按整块进行，尾块内部碎片也一并搬运。
        if len(table) != gpu_cache.blocks_for_tokens(length):
            raise RuntimeError("换出块数与逻辑长度不一致")
        self.swap_tables[request_id] = cpu_ids
        self.swap_lengths[request_id] = length
        self.swap_out_count += 1
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_block_count)
        self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
        gpu_cache.release(request_id)
        return {
            "request_id": request_id,
            "blocks": len(table),
            "logical_tokens": length,
            **self._movement_stats(len(table), length),
        }

    def swap_in(self, gpu_cache, request_id):
        """换入：申请新 GPU Block，逐块同步 H2D，重建 Block Table。

        不要求恢复原来的物理块编号；完成后归还 CPU Block。
        """
        if request_id not in self.swap_tables:
            raise KeyError("请求不在 CPU 池中: " + request_id)
        cpu_ids = self.swap_tables[request_id]
        length = self.swap_lengths[request_id]
        if gpu_cache.free_block_count < len(cpu_ids):
            raise RuntimeError(
                "GPU 空闲 Block 不足：free=%d 需要=%d（%s）"
                % (gpu_cache.free_block_count, len(cpu_ids), request_id)
            )
        gpu_ids = [gpu_cache._new_block() for _ in cpu_ids]
        for cpu_id, gpu_id in zip(cpu_ids, gpu_ids):
            gpu_cache.blocks[gpu_id].copy_(self.blocks[cpu_id])
        gpu_cache.block_tables[request_id] = gpu_ids
        gpu_cache.sequence_lengths[request_id] = length
        gpu_cache.peak_used_blocks = max(
            gpu_cache.peak_used_blocks, gpu_cache.used_block_count
        )
        for cpu_id in cpu_ids:
            heapq.heappush(self.free_block_ids, cpu_id)
        del self.swap_tables[request_id]
        del self.swap_lengths[request_id]
        self.swap_in_count += 1
        return {
            "request_id": request_id,
            "blocks": len(cpu_ids),
            "logical_tokens": length,
            **self._movement_stats(len(cpu_ids), length),
        }

    def snapshot(self):
        return {
            "pin_memory": self.pin_memory,
            "pool_blocks": self.max_blocks,
            "pool_bytes": self.pool_bytes,
            "used_blocks": self.used_block_count,
            "peak_used_blocks": self.peak_used_blocks,
            "peak_live_bytes": self.peak_live_bytes,
            "live_bytes": self.live_bytes,
            "swap_out_count": self.swap_out_count,
            "swap_in_count": self.swap_in_count,
            "swap_lengths": dict(self.swap_lengths),
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
