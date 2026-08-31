"""不下载权重的 CPU 自检：协议、LRU、Payload 和引擎 cold/warm 路径。"""

from types import SimpleNamespace

import torch
from torch import nn

from cache_protocol import TokenChunker
from engine import run_request
from experiment_utils import cache_service
from paged_cache import PagedKVCache


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=2,
            head_dim=4,
            rope_theta=10000.0,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        batch, count = input_ids.shape
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        new_values = (
            input_ids.float().view(batch, 1, count, 1)
            + position_ids.float().view(batch, 1, count, 1) / 100
        ).expand(batch, 2, count, 4)
        present = []
        for layer in range(2):
            key = new_values + layer
            value = new_values + layer + 0.5
            if past_key_values is not None:
                key = torch.cat((past_key_values[layer][0], key), dim=2)
                value = torch.cat((past_key_values[layer][1], value), dim=2)
            present.append((key, value))
        logits = torch.zeros((batch, count, 64), dtype=torch.float32)
        selected = (input_ids + position_ids) % 64
        logits.scatter_(2, selected.unsqueeze(-1), 1.0)
        return logits, present


class FailingClient:
    def lookup(self, identities):
        return 1

    def load(self, identities):
        raise RuntimeError("simulated corrupt payload")

    def store(self, *args, **kwargs):
        raise RuntimeError("simulated unavailable store")


def test_paged_roundtrip():
    model = DummyModel()
    cache = PagedKVCache(model.config, 4, 8, "cpu", torch.float32)
    cache.create_request("a")
    tokens = torch.arange(8).view(1, 1, 8, 1).expand(1, 2, 8, 4).float()
    values = [(tokens + layer, tokens + layer + 0.5) for layer in range(2)]
    cache.append("a", values)
    payload = cache.export_chunk("a", 0, 8)
    other = PagedKVCache(model.config, 4, 8, "cpu", torch.float32)
    other.create_request("b")
    other.import_chunk("b", payload, 8)
    for left, right in zip(cache.dense("a"), other.dense("b")):
        assert torch.equal(left[0], right[0])
        assert torch.equal(left[1], right[1])
    assert cache.block_tables["a"] != other.block_tables["b"] or cache is not other


def test_service_and_engine():
    model = DummyModel().eval()
    token_ids = list(range(25))
    with cache_service(capacity_mib=1) as (client, _):
        cold = run_request(
            model, token_ids, 3, 63, "cpu",
            mode="external", external_client=client,
            model_id="dummy", revision="v1", block_size=4,
            external_chunk_size=8, token_budget=5,
            capture_logits=True,
        )
        warm = run_request(
            model, token_ids, 3, 63, "cpu",
            mode="external", external_client=client,
            model_id="dummy", revision="v1", block_size=4,
            external_chunk_size=8, token_budget=5,
            capture_logits=True,
        )
        baseline = run_request(
            model, token_ids, 3, 63, "cpu",
            mode="recompute", model_id="dummy", revision="v1",
            block_size=4, external_chunk_size=8, token_budget=5,
            capture_logits=True,
        )
        assert cold["metrics"]["hit_tokens"] == 0
        assert warm["metrics"]["hit_tokens"] == 24
        assert warm["metrics"]["executed_prefill_tokens"] == 1
        assert warm["new_token_ids"] == baseline["new_token_ids"]
        assert torch.equal(warm["first_token_logits"], baseline["first_token_logits"])
        wrong_revision = TokenChunker({"model": "dummy", "revision": "v2"}, 8)
        assert client.lookup(wrong_revision.identities(token_ids)) == 0
        stats = client.stats()
        assert stats["entry_count"] == 3


def test_fixed_capacity_lru():
    namespace = {"model": "dummy", "revision": "v1", "dtype": "float32"}
    chunker = TokenChunker(namespace, 4)
    first = chunker.identities([1, 2, 3, 4, 9])[0]
    second = chunker.identities([5, 6, 7, 8, 9])[0]
    payload = bytes(range(128))
    with cache_service(capacity_mib=200 / (1024 * 1024)) as (client, _):
        client.store(first, payload, chunker.namespace_digest, "float32", (32,), [])
        client.store(second, payload, chunker.namespace_digest, "float32", (32,), [])
        assert client.lookup([first]) == 0
        assert client.lookup([second]) == 1
        stats = client.stats()
        assert stats["entry_count"] == 1
        assert stats["eviction_count"] == 1


def test_failure_falls_back_to_recompute():
    model = DummyModel().eval()
    token_ids = list(range(17))
    result = run_request(
        model, token_ids, 2, 63, "cpu",
        mode="external", external_client=FailingClient(),
        model_id="dummy", revision="v1", block_size=4,
        external_chunk_size=8, token_budget=5,
    )
    assert result["metrics"]["load_fallback"] is True
    assert result["metrics"]["hit_tokens"] == 0
    assert result["metrics"]["executed_prefill_tokens"] == len(token_ids)
    assert result["metrics"]["store_error"] is not None


def main():
    test_paged_roundtrip()
    test_service_and_engine()
    test_fixed_capacity_lru()
    test_failure_falls_back_to_recompute()
    print("chapter_14 smoke_test: PASS")


if __name__ == "__main__":
    main()
