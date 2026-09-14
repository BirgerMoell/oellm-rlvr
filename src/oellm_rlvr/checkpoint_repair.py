from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


QWEN35_VLM_TEXT_PREFIX = "model.language_model."
QWEN35_CAUSAL_TEXT_PREFIX = "model."


def qwen35_text_key(source_key: str) -> str:
    """Map a text-only Qwen3.5 export from VLM to causal-LM key layout."""
    if source_key.startswith(QWEN35_VLM_TEXT_PREFIX):
        return QWEN35_CAUSAL_TEXT_PREFIX + source_key[len(QWEN35_VLM_TEXT_PREFIX) :]
    if source_key == "lm_head.weight":
        return source_key
    raise ValueError(f"unexpected Qwen3.5 text checkpoint key: {source_key}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_weight_files(source: Path) -> list[Path]:
    single = source / "model.safetensors"
    if single.is_file():
        return [single]
    shards = sorted(source.glob("model-*-of-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors model weights under {source}")
    return shards


def repair_qwen35_text_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    max_shard_bytes: int = 4 * 1024**3,
) -> dict[str, Any]:
    """Create an atomic text-only Qwen3.5 checkpoint with corrected keys.

    Tensor values and dtypes are preserved. The output is sharded so the
    conversion never needs to retain the full checkpoint in host memory.
    """
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("checkpoint repair requires safetensors and torch") from error

    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    if max_shard_bytes < 1:
        raise ValueError("max_shard_bytes must be positive")

    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "qwen3_5_text":
        raise ValueError(f"expected model_type=qwen3_5_text, found {config.get('model_type')!r}")
    if config.get("architectures") != ["Qwen3_5ForCausalLM"]:
        raise ValueError(f"unexpected architectures: {config.get('architectures')!r}")

    temporary = output.parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    source_files = _source_weight_files(source)
    source_hashes = {path.name: _sha256(path) for path in source_files}
    weight_map: dict[str, str] = {}
    part_paths: list[Path] = []
    tensors: dict[str, Any] = {}
    tensor_bytes = 0
    total_bytes = 0
    dtype_counts: Counter[str] = Counter()
    source_key_count = 0

    def flush() -> None:
        nonlocal tensors, tensor_bytes
        if not tensors:
            return
        part = temporary / f"part-{len(part_paths) + 1:05d}.safetensors"
        save_file(tensors, str(part), metadata={"format": "pt"})
        for key in tensors:
            weight_map[key] = part.name
        part_paths.append(part)
        tensors = {}
        tensor_bytes = 0

    try:
        seen: set[str] = set()
        for source_file in source_files:
            with safe_open(source_file, framework="pt", device="cpu") as handle:
                for source_key in handle.keys():
                    destination_key = qwen35_text_key(source_key)
                    if destination_key in seen:
                        raise ValueError(f"duplicate destination tensor key: {destination_key}")
                    tensor = handle.get_tensor(source_key)
                    size = tensor.numel() * tensor.element_size()
                    if tensors and tensor_bytes + size > max_shard_bytes:
                        flush()
                    tensors[destination_key] = tensor
                    tensor_bytes += size
                    total_bytes += size
                    dtype_counts[str(tensor.dtype)] += 1
                    seen.add(destination_key)
                    source_key_count += 1
        flush()

        shard_count = len(part_paths)
        renamed: dict[str, str] = {}
        output_hashes: dict[str, str] = {}
        for index, part in enumerate(part_paths, start=1):
            name = (
                "model.safetensors"
                if shard_count == 1
                else f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            )
            destination = temporary / name
            part.rename(destination)
            renamed[part.name] = name
            output_hashes[name] = _sha256(destination)
        weight_map = {key: renamed[name] for key, name in weight_map.items()}
        if shard_count > 1:
            index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
            (temporary / "model.safetensors.index.json").write_text(
                json.dumps(index, indent=2, sort_keys=True) + "\n"
            )

        for child in source.iterdir():
            if child.name == "model.safetensors.index.json" or child.suffix == ".safetensors":
                continue
            destination = temporary / child.name
            if child.is_dir():
                shutil.copytree(child, destination)
            else:
                shutil.copy2(child, destination)
        config["dtype"] = "bfloat16"
        (temporary / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

        manifest = {
            "schema": "oellm-qwen35-text-checkpoint-repair-v1",
            "operation": f"{QWEN35_VLM_TEXT_PREFIX}* -> {QWEN35_CAUSAL_TEXT_PREFIX}*",
            "source": str(source),
            "source_files": {
                path.name: {"bytes": path.stat().st_size, "sha256": source_hashes[path.name]}
                for path in source_files
            },
            "output": str(output),
            "output_files": {
                name: {"bytes": (temporary / name).stat().st_size, "sha256": digest}
                for name, digest in sorted(output_hashes.items())
            },
            "source_tensor_keys": source_key_count,
            "output_tensor_keys": len(weight_map),
            "tensor_value_bytes": total_bytes,
            "tensor_dtypes": dict(sorted(dtype_counts.items())),
            "config_changes": {"dtype": [str(json.loads(config_path.read_text()).get("dtype")), "bfloat16"]},
        }
        (temporary / "repair-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
