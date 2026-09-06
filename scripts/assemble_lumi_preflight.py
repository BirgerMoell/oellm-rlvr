#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--expected-nodes", type=int, default=2)
    parser.add_argument("--expected-gpus-per-node", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.results_dir)
    node_paths = sorted(root.glob("node-*.json"))
    nodes = [json.loads(path.read_text()) for path in node_paths]
    ray = json.loads((root / "ray-cluster.json").read_text())
    generation = json.loads((root / "model-generation.json").read_text())
    transfer_log = root / "hierarchical-transfer.log"
    transfer_text = transfer_log.read_text()
    completed_nodes = transfer_text.count('"event": "hierarchical_weight_transfer_probe_node_complete"')
    received_ranks = transfer_text.count('"received": 9173')
    gates = {
        "node_count": len(nodes) == args.expected_nodes,
        "gpu_count_per_node": all(
            node.get("ok")
            and node.get("details", {}).get("visible_gpus") == args.expected_gpus_per_node
            for node in nodes
        ),
        "ray_cluster": bool(ray.get("ok")),
        "offline_bf16_generation": bool(generation.get("ok")),
        "hierarchical_transfer": completed_nodes == args.expected_nodes and received_ranks == 9,
    }
    report = {
        "ok": all(gates.values()),
        "gates": gates,
        "nodes": nodes,
        "ray": ray,
        "generation": generation,
        "hierarchical_transfer": {
            "node_completions": completed_nodes,
            "received_ranks": received_ranks,
            "log": str(transfer_log),
            "sha256": _hash(transfer_log),
        },
        "package_lock_sha256": _hash(root / "package-lock.txt"),
        "container_and_driver_digests_sha256": _hash(root / "container-and-driver-digests.txt"),
    }
    (root / "preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
