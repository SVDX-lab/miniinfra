"""无需模型权重和 GPU 的第 15 期快速自检。"""

from types import SimpleNamespace

import torch

from engine import (
    build_namespace, decode_from_payload, export_handoff, prefill_request,
    run_monolithic,
)
from experiment_utils import handoff_service


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            num_hidden_layers=2, num_key_value_heads=2, head_dim=4,
            rope_theta=10000.0, vocab_size=64,
        )

    def forward(
        self, input_ids, attention_mask, position_ids,
        past_key_values=None, use_cache=True,
    ):
        batch, count = input_ids.shape
        current = (
            input_ids.float().view(batch, 1, count, 1)
            + position_ids.float().view(batch, 1, count, 1) / 100
        ).expand(batch, 2, count, 4)
        present = []
        for layer in range(self.config.num_hidden_layers):
            key, value = current + layer, current + layer + 0.5
            if past_key_values is not None:
                key = torch.cat((past_key_values[layer][0], key), dim=2)
                value = torch.cat((past_key_values[layer][1], value), dim=2)
            present.append((key, value))
        logits = torch.zeros((batch, count, self.config.vocab_size))
        targets = (input_ids + 1) % self.config.vocab_size
        logits.scatter_(2, targets.unsqueeze(-1), 1.0)
        return logits, present


def main():
    model = ToyModel()
    tokens = [3, 7, 11, 15, 19]
    baseline = run_monolithic(
        model, tokens, max_new_tokens=4, device="cpu",
        block_size=2, token_budget=3,
    )
    cache, prefill = prefill_request(
        model, tokens, "cpu", block_size=2, token_budget=3,
        max_new_tokens=4,
    )
    namespace = build_namespace("toy", "v1", model, torch.float32, 2)
    manifest, payload, _ = export_handoff(
        cache, "request-0", tokens, prefill["first_token"],
        prefill["first_logits_sha256"], namespace, "attempt-0",
    )
    with handoff_service() as (client, _):
        assert client.publish(manifest, payload)["state"] == "kv_ready"
        response, received = client.receive("request-0", "attempt-0")
        acknowledged = []

        def ack():
            client.acknowledge("request-0", "attempt-0", True)
            acknowledged.append(True)

        split = decode_from_payload(
            model, response["manifest"], received, "cpu", namespace,
            max_new_tokens=4, block_size=2, on_imported=ack,
        )
        assert acknowledged
        assert client.status("request-0", "attempt-0")["state"] == "acknowledged"
        cache.release("request-0")
        assert client.release("request-0", "attempt-0")["released"]
        assert client.stats()["entries"] == 0
        timeout_manifest = dict(manifest)
        timeout_manifest["request_id"] = "timeout-request"
        timeout_manifest["attempt_id"] = "timeout-attempt"
        client.publish(timeout_manifest, payload)
        assert client.abort(
            "timeout-request", "timeout-attempt", "模拟 ACK 超时"
        )["state"] == "fallback"
        assert client.release(
            "timeout-request", "timeout-attempt"
        )["released"]
        assert client.stats()["entries"] == 0
    assert baseline["new_token_ids"] == split["new_token_ids"]
    assert cache.snapshot()["used_blocks"] == 0
    assert split["metrics"]["final_cache_snapshot"]["used_blocks"] == 0
    broken = bytearray(payload)
    broken[-1] ^= 1
    try:
        decode_from_payload(
            model, manifest, bytes(broken), "cpu", namespace,
            max_new_tokens=2, block_size=2,
        )
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("损坏 Payload 未被拒绝")
    wrong_namespace = dict(namespace)
    wrong_namespace["format_version"] = 2
    try:
        decode_from_payload(
            model, manifest, payload, "cpu", wrong_namespace,
            max_new_tokens=2, block_size=2,
        )
    except ValueError as error:
        assert "Namespace" in str(error)
    else:
        raise AssertionError("错误 Namespace 未被拒绝")
    print("第 15 期 CPU smoke test 通过")


if __name__ == "__main__":
    main()
