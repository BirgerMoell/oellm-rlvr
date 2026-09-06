#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

EXPECTED_SKYRL = "f5bc3b78dfddfb352870d5d7430cd226e5785838"
EXPECTED_HARBOR = "4407eb5227a2ff4f0d3f16b2eb48849382fdf276"
EXPECTED_HARBOR_BOOTSTRAP_SHA256 = "1cb240109faf7caa4fd273fab6af035a19e14e679570987e500e4ff6d1ff124f"


def _commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _dirty_paths(path: Path) -> list[str]:
    value = subprocess.run(
        ["git", "-C", str(path), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(line[3:] for line in value.splitlines() if line.strip())


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def inspect_stack(skyrl_source: Path, harbor_source: Path, *, require_gpu: bool) -> dict[str, Any]:
    errors: list[str] = []
    commits = {"skyrl": _commit(skyrl_source), "harbor": _commit(harbor_source)}
    dirty_paths = {"skyrl": _dirty_paths(skyrl_source), "harbor": _dirty_paths(harbor_source)}
    if commits["skyrl"] != EXPECTED_SKYRL:
        errors.append(f"SkyRL commit mismatch: {commits['skyrl']}")
    if commits["harbor"] != EXPECTED_HARBOR:
        errors.append(f"Harbor commit mismatch: {commits['harbor']}")
    if dirty_paths["skyrl"]:
        errors.append(f"unexpected SkyRL source changes: {dirty_paths['skyrl']}")
    allowed_harbor_patch = ["src/harbor/environments/singularity/bootstrap.sh"]
    if dirty_paths["harbor"] not in ([], allowed_harbor_patch):
        errors.append(f"unexpected Harbor source changes: {dirty_paths['harbor']}")
    harbor_bootstrap = harbor_source / allowed_harbor_patch[0]
    harbor_bootstrap_sha256 = hashlib.sha256(harbor_bootstrap.read_bytes()).hexdigest()
    if (
        dirty_paths["harbor"] == allowed_harbor_patch
        and harbor_bootstrap_sha256 != EXPECTED_HARBOR_BOOTSTRAP_SHA256
    ):
        errors.append(f"unexpected Harbor bootstrap patch digest: {harbor_bootstrap_sha256}")

    modules = (
        "skyrl",
        "skyrl.train.config",
        "skyrl.train.entrypoints.main_base",
        "skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker",
        "skyrl.backends.skyrl_train.weight_sync.broadcast_strategy",
        "harbor",
        "harbor.environments.singularity",
        "examples.train_integrations.harbor.harbor_generator",
        "examples.train_integrations.harbor.entrypoints.main_harbor",
    )
    imported: dict[str, str] = {}
    for name in modules:
        try:
            importlib.import_module(name)
            imported[name] = "ok"
        except Exception as error:  # noqa: BLE001 - compatibility report records every import boundary
            imported[name] = f"{type(error).__name__}: {error}"
            errors.append(f"import {name}: {error}")

    import torch

    versions = {
        name: _version(name)
        for name in (
            "torch",
            "vllm",
            "transformers",
            "ray",
            "skyrl",
            "skyrl-gym",
            "harbor",
            "litellm",
        )
    }
    if not torch.version.hip:
        errors.append("torch.version.hip is empty; this is not a ROCm build")
    if not str(versions["vllm"]).startswith("0.22.1+lumi_aif_gfx90a"):
        errors.append(f"unexpected vLLM build: {versions['vllm']}")
    if versions["ray"] != "2.56.0":
        errors.append(f"SkyRL overlay requires Ray 2.56.0, got {versions['ray']}")
    if require_gpu and torch.cuda.device_count() != 8:
        errors.append(f"expected 8 visible MI250X GCDs, got {torch.cuda.device_count()}")

    loaded_maps = Path("/proc/self/maps").read_text(errors="replace")
    cuda_libraries = sorted(
        {
            token
            for line in loaded_maps.splitlines()
            for token in line.split()
            if "libcuda.so" in token or "libnccl.so" in token
        }
    )
    if cuda_libraries:
        errors.append(f"CUDA/NCCL libraries were loaded: {cuda_libraries}")
    return {
        "ok": not errors,
        "commits": commits,
        "source_changes": dirty_paths,
        "harbor_bootstrap_sha256": harbor_bootstrap_sha256,
        "versions": versions,
        "torch_hip": torch.version.hip,
        "visible_gpus": torch.cuda.device_count(),
        "required_gpu_check": require_gpu,
        "imports": imported,
        "loaded_cuda_or_nccl_libraries": cuda_libraries,
        "environment": {
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skyrl-source", required=True)
    parser.add_argument("--harbor-source", required=True)
    parser.add_argument("--output")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    report = inspect_stack(Path(args.skyrl_source), Path(args.harbor_source), require_gpu=args.require_gpu)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
