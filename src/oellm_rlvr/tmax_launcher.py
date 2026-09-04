from __future__ import annotations

import importlib
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .compat import patch_open_instruct_grpo_module

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


def _run_grpo_fast(module: ModuleType) -> None:
    """Run pinned TMAX GRPO after its import-time compatibility hooks.

    ``runpy.run_path(..., run_name="__main__")`` never imports
    ``open_instruct.grpo_fast`` by its canonical name. That prevents a
    post-import hook from replacing methods on its already decorated Ray
    trainer class. Importing first gives the hook a stable module and class;
    this reproduces the pinned script's small ``if __name__`` entry point.
    """
    parser = module.ArgumentParserPlus(
        (
            module.grpo_utils.GRPOExperimentConfig,
            module.TokenizerConfig,
            module.ModelConfig,
            module.data_loader_lib.StreamingDataLoaderConfig,
            module.data_loader_lib.VLLMConfig,
            module.EnvsConfig,
        )
    )
    parser.set_defaults(
        exp_name="grpo",
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        per_device_train_batch_size=1,
    )
    parsed = parser.parse_args_into_dataclasses()
    if len(parsed) != 6:
        raise RuntimeError(f"pinned GRPO entry point returned {len(parsed)} configs; expected 6")
    # Match the pinned script footer exactly.  Some trainer actor methods still
    # resolve ``streaming_config`` as a module global instead of using
    # ``self.streaming_config``.  The actor wrapper repeats this assignment in
    # each Ray worker, where module globals are process-local.
    (
        module.args,
        module.tokenizer_config,
        module.model_config,
        module.streaming_config,
        module.vllm_config,
        module.tools_config,
    ) = parsed
    module.main(*parsed)


def _run_backend_entrypoint(script: str) -> None:
    if os.environ.get("OELLM_WEIGHT_TRANSFER") == "hierarchical" and Path(script).name == "grpo_fast.py":
        print(
            "oellm-rlvr: importing open_instruct.grpo_fast before hierarchical launch",
            file=sys.stderr,
        )
        module = importlib.import_module("open_instruct.grpo_fast")
        patch_open_instruct_grpo_module(module)
        _run_grpo_fast(module)
        return
    runpy.run_path(script, run_name="__main__")


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
    _run_backend_entrypoint(script)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
