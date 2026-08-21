#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", choices=("rocm", "cuda"), required=True)
    parser.add_argument("--gpus", type=int, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    details: dict[str, object] = {"python": sys.version.split()[0]}
    try:
        import torch

        details.update(
            torch=torch.__version__,
            hip=torch.version.hip,
            cuda=torch.version.cuda,
            visible_gpus=torch.cuda.device_count(),
        )
        if torch.cuda.device_count() != args.gpus:
            errors.append(f"expected {args.gpus} visible GPUs, found {torch.cuda.device_count()}")
        if args.accelerator == "rocm" and not torch.version.hip:
            errors.append("ROCm profile selected but torch.version.hip is empty")
        if args.accelerator == "cuda" and not torch.version.cuda:
            errors.append("CUDA profile selected but torch.version.cuda is empty")
    except Exception as error:
        errors.append(f"torch: {error}")

    for module in ("ray", "vllm", "open_instruct"):
        try:
            loaded = importlib.import_module(module)
            details[module] = getattr(loaded, "__version__", "imported")
        except Exception as error:
            errors.append(f"{module}: {error}")
    try:
        importlib.import_module("vllm.distributed.weight_transfer.nccl_engine")
        details["weight_transfer"] = "native_nccl_engine"
    except Exception as error:
        errors.append(f"vLLM native weight transfer: {error}")

    print(json.dumps({"ok": not errors, "details": details, "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
