#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument("--expected-gpus", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=args.address, logging_level="ERROR")
    deadline = time.monotonic() + args.timeout
    alive: list[dict] = []
    while time.monotonic() < deadline:
        alive = [node for node in ray.nodes() if node.get("Alive")]
        gpu_total = sum(float(node.get("Resources", {}).get("GPU", 0)) for node in alive)
        if len(alive) == args.expected_nodes and gpu_total == args.expected_gpus:
            break
        time.sleep(2)

    gpu_total = sum(float(node.get("Resources", {}).get("GPU", 0)) for node in alive)

    @ray.remote(num_cpus=0)
    def identify() -> dict[str, str]:
        return {"hostname": socket.gethostname(), "node_id": ray.get_runtime_context().get_node_id()}

    identities = []
    for node in alive:
        node_id = str(node["NodeID"])
        identities.append(
            ray.get(
                identify.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
                ).remote(),
                timeout=30,
            )
        )
    report = {
        "ok": len(alive) == args.expected_nodes
        and gpu_total == args.expected_gpus
        and len({item["hostname"] for item in identities}) == args.expected_nodes,
        "address": args.address,
        "expected_nodes": args.expected_nodes,
        "alive_nodes": len(alive),
        "expected_gpus": args.expected_gpus,
        "declared_gpus": gpu_total,
        "workers": sorted(identities, key=lambda item: item["hostname"]),
    }
    print(json.dumps(report, sort_keys=True))
    ray.shutdown()
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
