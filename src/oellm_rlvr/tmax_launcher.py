from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Any

MI250X_GCD_SPEC = {
    "flops": 191.5e12,
    "memory_size": 64e9,
    "memory_bandwidth": 1.6e12,
}


def _install_local_snapshot_alias(model_id: str, model_path: str) -> str:
    import huggingface_hub

    resolved = Path(model_path).resolve()
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"local model is missing config.json: {resolved}")
    original = huggingface_hub.snapshot_download

    def snapshot_download(repo_id: str, *args: Any, **kwargs: Any) -> str:
        if repo_id in {model_id, str(resolved)}:
            return str(resolved)
        return original(repo_id, *args, **kwargs)

    huggingface_hub.snapshot_download = snapshot_download
    return str(resolved)


def _rewrite_model_arg(args: list[str], model_id: str, model_path: str) -> list[str]:
    rewritten = list(args)
    for index, arg in enumerate(rewritten[:-1]):
        if arg == "--model_name_or_path" and rewritten[index + 1] == model_id:
            rewritten[index + 1] = model_path
    return rewritten


def _install_lumi_device_spec() -> None:
    from open_instruct import utils

    # PyTorch exposes each MI250X GCD as one 64 GB logical device on LUMI.
    utils.GPU_SPECS.setdefault("mi250x", MI250X_GCD_SPEC)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m oellm_rlvr.tmax_launcher BACKEND_SCRIPT [ARGS...]")
    script, backend_args = sys.argv[1], sys.argv[2:]
    model_id = os.environ.get("OELLM_MODEL_ID")
    model_path = os.environ.get("OELLM_MODEL_PATH")
    if not model_id or not model_path:
        raise RuntimeError("OELLM_MODEL_ID and OELLM_MODEL_PATH are required")
    resolved_model_path = _install_local_snapshot_alias(model_id, model_path)
    backend_args = _rewrite_model_arg(backend_args, model_id, resolved_model_path)
    _install_lumi_device_spec()
    sys.argv = [script, *backend_args]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
