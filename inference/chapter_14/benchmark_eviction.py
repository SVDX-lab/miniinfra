"""固定容量 leaf-LRU Trace；不下载模型权重。"""

import argparse

from cache_protocol import TokenChunker
from experiment_utils import cache_service, write_json


def main():
    parser = argparse.ArgumentParser(description="第 14 期外部 Cache LRU Trace")
    parser.add_argument("--entry-bytes", type=int, default=4096)
    parser.add_argument("--capacity-entries", type=int, default=2)
    parser.add_argument("--trace", default="A,B,A,C,A,B")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.entry_bytes < 1 or args.capacity_entries < 1:
        raise ValueError("entry-bytes 和 capacity-entries 必须大于 0")
    chunker = TokenChunker({"model": "trace", "revision": "v1"}, 4)
    identities = {
        name: chunker.identities(
            [100 + index * 10 + offset for offset in range(4)] + [999]
        )[0]
        for index, name in enumerate(("A", "B", "C"))
    }
    payloads = {
        name: bytes([index + 1]) * args.entry_bytes
        for index, name in enumerate(("A", "B", "C"))
    }
    capacity_mib = (
        args.entry_bytes * args.capacity_entries / (1024 * 1024)
    )
    events = []
    with cache_service(capacity_mib) as (client, _):
        for name in args.trace.split(","):
            name = name.strip()
            identity = identities[name]
            hit = client.lookup([identity]) == 1
            if not hit:
                client.store(
                    identity, payloads[name], chunker.namespace_digest,
                    "uint8", (args.entry_bytes,), [],
                )
            stats = client.stats()
            events.append({
                "name": name,
                "hit": hit,
                "resident": stats["digests"],
                "used_bytes": stats["used_bytes"],
                "eviction_count": stats["eviction_count"],
            })
        final_stats = client.stats()
    result = {
        "config": vars(args),
        "events": events,
        "hit_count": sum(event["hit"] for event in events),
        "miss_count": sum(not event["hit"] for event in events),
        "final_stats": final_stats,
    }
    write_json(args.output, result)
    print("hits=%d misses=%d evictions=%d" % (
        result["hit_count"], result["miss_count"],
        final_stats["eviction_count"],
    ))


if __name__ == "__main__":
    main()
