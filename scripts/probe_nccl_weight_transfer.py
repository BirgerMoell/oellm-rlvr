#!/usr/bin/env python3
"""Exercise vLLM's native NCCL/RCCL transfer group without loading a model."""

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
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=9)
    parser.add_argument("--trainer-host", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--rank", type=int)
    return parser.parse_args()


def local_rank_assignments(hostname: str, trainer_host: str, world_size: int) -> list[tuple[int, int]]:
    """Return ``(global rank, local device)`` pairs for this probe node."""
    if hostname == trainer_host:
        return [(0, 0)]
    return [(rank, rank - 1) for rank in range(1, world_size)]


def child_environment(local_device: int) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = str(local_device)
    env["CUDA_VISIBLE_DEVICES"] = str(local_device)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_rank(args: argparse.Namespace) -> int:
    import torch
    from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

    assert args.rank is not None
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"rank {args.rank}: expected one visible device, found {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    started = time.perf_counter()
    group = NCCLWeightTransferEngine._stateless_init_process_group(
        args.master_address,
        args.master_port,
        args.rank,
        args.world_size,
        device=0,
    )
    initialized = time.perf_counter()
    value = torch.tensor([9173 if args.rank == 0 else -1], dtype=torch.int64, device="cuda")
    group.broadcast(value, src=0, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    received = int(value.item())
    if received != 9173:
        raise RuntimeError(f"rank {args.rank}: broadcast returned {received}, expected 9173")
    print(
        json.dumps(
            {
                "event": "weight_transfer_probe_rank",
                "rank": args.rank,
                "hostname": socket.gethostname(),
                "device": torch.cuda.get_device_name(0),
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
    assignments = local_rank_assignments(hostname, args.trainer_host, args.world_size)
    script = str(Path(__file__).resolve())
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for global_rank, local_device in assignments:
        command = [
            sys.executable,
            script,
            "--master-address",
            args.master_address,
            "--master-port",
            str(args.master_port),
            "--world-size",
            str(args.world_size),
            "--trainer-host",
            args.trainer_host,
            "--timeout",
            str(args.timeout),
            "--rank",
            str(global_rank),
        ]
        processes.append(
            (
                global_rank,
                subprocess.Popen(command, env=child_environment(local_device), text=True),
            )
        )

    deadline = time.monotonic() + args.timeout
    pending = dict(processes)
    failures: dict[int, int] = {}
    while pending and time.monotonic() < deadline:
        for rank, process in list(pending.items()):
            result = process.poll()
            if result is None:
                continue
            pending.pop(rank)
            if result != 0:
                failures[rank] = result
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
        raise TimeoutError(f"{hostname}: ranks {sorted(pending)} exceeded {args.timeout}s")
    if failures:
        raise RuntimeError(f"{hostname}: rank failures {failures}")
    print(
        json.dumps(
            {
                "event": "weight_transfer_probe_node_complete",
                "hostname": hostname,
                "ranks": [rank for rank, _ in assignments],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.world_size < 2:
        raise ValueError("world size must be at least two")
    if args.rank is not None:
        if not 0 <= args.rank < args.world_size:
            raise ValueError(f"rank {args.rank} is outside world size {args.world_size}")
        return run_rank(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
