#!/usr/bin/env python3
"""Probe trainer→relay→seven-leaf RCCL transfer without loading a model."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-address", required=True)
    parser.add_argument("--relay-address", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--trainer-host", required=True)
    parser.add_argument("--leaf-count", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--role", choices=("trainer", "relay", "leaf"))
    parser.add_argument("--leaf-index", type=int)
    parser.add_argument("--local-device", type=int, default=0)
    return parser.parse_args()


def local_roles(hostname: str, trainer_host: str, leaf_count: int) -> list[tuple[str, int | None, int]]:
    """Return ``(role, leaf index, local device)`` assignments for a node."""
    if hostname == trainer_host:
        return [("trainer", None, 0)]
    return [("relay", None, 0), *(("leaf", index, index) for index in range(1, leaf_count + 1))]


def child_environment(local_device: int) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = str(local_device)
    env["CUDA_VISIBLE_DEVICES"] = str(local_device)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _group(address: str, port: int, rank: int):
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    return NCCLWeightTransferEngine._stateless_init_process_group(address, port, rank, 2, device=0)


def run_role(args: argparse.Namespace) -> int:
    import torch

    torch.cuda.set_device(args.local_device)
    started = time.perf_counter()
    if args.role == "trainer":
        groups = [_group(args.trainer_address, args.master_port, 0)]
        value = torch.tensor([9173], dtype=torch.int64, device="cuda")
    elif args.role == "relay":
        groups = [_group(args.trainer_address, args.master_port, 1)]
        groups.extend(
            _group(args.relay_address, args.master_port + leaf_index, 0)
            for leaf_index in range(1, args.leaf_count + 1)
        )
        value = torch.tensor([-1], dtype=torch.int64, device="cuda")
    else:
        if args.leaf_index is None:
            raise ValueError("leaf role requires --leaf-index")
        groups = [_group(args.relay_address, args.master_port + args.leaf_index, 1)]
        value = torch.tensor([-1], dtype=torch.int64, device="cuda")

    initialized = time.perf_counter()
    if args.role == "relay":
        groups[0].broadcast(value, src=0, stream=torch.cuda.current_stream())
        for group in groups[1:]:
            group.broadcast(value, src=0, stream=torch.cuda.current_stream())
    else:
        groups[0].broadcast(value, src=0, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    received = int(value.item())
    if received != 9173:
        raise RuntimeError(f"{args.role}: broadcast returned {received}, expected 9173")
    print(
        json.dumps(
            {
                "event": "hierarchical_weight_transfer_probe_rank",
                "role": args.role,
                "leaf_index": args.leaf_index,
                "hostname": socket.gethostname(),
                "device": torch.cuda.get_device_name(args.local_device),
                "init_seconds": round(initialized - started, 3),
                "total_seconds": round(time.perf_counter() - started, 3),
                "received": received,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def run_orchestrator(args: argparse.Namespace) -> int:
    hostname = socket.gethostname()
    assignments = local_roles(hostname, args.trainer_host, args.leaf_count)
    script = str(Path(__file__).resolve())
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    for role, leaf_index, local_device in assignments:
        command = [
            sys.executable,
            script,
            "--trainer-address",
            args.trainer_address,
            "--relay-address",
            args.relay_address,
            "--master-port",
            str(args.master_port),
            "--trainer-host",
            args.trainer_host,
            "--leaf-count",
            str(args.leaf_count),
            "--timeout",
            str(args.timeout),
            "--role",
            role,
            "--local-device",
            "0",
        ]
        if leaf_index is not None:
            command.extend(["--leaf-index", str(leaf_index)])
        label = role if leaf_index is None else f"leaf-{leaf_index}"
        processes.append(
            (label, subprocess.Popen(command, env=child_environment(local_device), text=True))
        )

    deadline = time.monotonic() + args.timeout
    pending = dict(processes)
    failures: dict[str, int] = {}
    while pending and time.monotonic() < deadline:
        for label, process in list(pending.items()):
            result = process.poll()
            if result is None:
                continue
            pending.pop(label)
            if result != 0:
                failures[label] = result
        if pending:
            time.sleep(0.1)

    for process in pending.values():
        process.terminate()
    for process in pending.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if pending:
        raise TimeoutError(f"{hostname}: roles {sorted(pending)} exceeded {args.timeout}s")
    if failures:
        raise RuntimeError(f"{hostname}: role failures {failures}")
    print(
        json.dumps(
            {
                "event": "hierarchical_weight_transfer_probe_node_complete",
                "hostname": hostname,
                "roles": [label for label, _ in processes],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.leaf_count < 1 or args.leaf_count > 7:
        raise ValueError("leaf count must be in [1, 7]")
    if args.master_port + args.leaf_count > 65535:
        raise ValueError("master port range exceeds 65535")
    return run_role(args) if args.role else run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
