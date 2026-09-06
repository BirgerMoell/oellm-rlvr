from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 8 * 1024 * 1024


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return sha256(payload).hexdigest()


def _is_within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _component_inventory(root: Path, files: list[dict[str, Any]]) -> dict[str, Any]:
    names = {str(item["path"]): item for item in files}
    config = json.loads((root / "config.json").read_text()) if "config.json" in names else {}
    tokenizer_config = (
        json.loads((root / "tokenizer_config.json").read_text()) if "tokenizer_config.json" in names else {}
    )

    weight_files = sorted(
        name
        for name in names
        if name.endswith((".safetensors", ".bin", ".pt")) and not name.startswith("optimizer")
    )
    tokenizer_files = sorted(
        name
        for name in names
        if Path(name).name.startswith("tokenizer")
        or Path(name).name in {"special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"}
    )
    chat_template_source: str | None = None
    chat_template_sha256: str | None = None
    if "chat_template.jinja" in names:
        chat_template_source = "chat_template.jinja"
        chat_template_sha256 = str(names["chat_template.jinja"]["sha256"])
    elif tokenizer_config.get("chat_template"):
        chat_template_source = "tokenizer_config.json#chat_template"
        chat_template_sha256 = sha256(str(tokenizer_config["chat_template"]).encode()).hexdigest()

    return {
        "model_config": {
            "path": "config.json" if "config.json" in names else None,
            "sha256": names.get("config.json", {}).get("sha256"),
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
        },
        "weights": {
            "files": weight_files,
            "aggregate_sha256": _canonical_digest({name: names[name]["sha256"] for name in weight_files}),
        },
        "tokenizer": {
            "files": tokenizer_files,
            "aggregate_sha256": _canonical_digest({name: names[name]["sha256"] for name in tokenizer_files}),
        },
        "chat_template": {"source": chat_template_source, "sha256": chat_template_sha256},
        "generation_config": {
            "path": "generation_config.json" if "generation_config.json" in names else None,
            "sha256": names.get("generation_config.json", {}).get("sha256"),
        },
    }


def build_checkpoint_manifest(
    root: str | Path,
    *,
    model_id: str,
    revision: str,
    allowed_symlink_roots: list[str | Path] | None = None,
    require_components: bool = True,
) -> dict[str, Any]:
    """Create a deterministic, content-addressed manifest for a local HF checkpoint.

    Directory symlinks are never traversed. File symlinks are only dereferenced when
    their resolved target is under the checkpoint root or an explicitly permitted
    cache root.
    """
    checkpoint_root = Path(root).expanduser().resolve(strict=True)
    if not checkpoint_root.is_dir():
        raise ValueError(f"checkpoint root is not a directory: {checkpoint_root}")
    allowed_roots = tuple(
        dict.fromkeys(
            [checkpoint_root]
            + [Path(path).expanduser().resolve(strict=True) for path in (allowed_symlink_roots or [])]
        )
    )

    files: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(checkpoint_root, followlinks=False):
        directory_path = Path(directory)
        symlink_directories = [name for name in directory_names if (directory_path / name).is_symlink()]
        if symlink_directories:
            names = ", ".join(sorted(str((directory_path / name).relative_to(checkpoint_root)) for name in symlink_directories))
            raise ValueError(f"directory symlinks are not allowed in checkpoint manifests: {names}")
        for name in sorted(file_names):
            path = directory_path / name
            relative = path.relative_to(checkpoint_root).as_posix()
            entry: dict[str, Any] = {"path": relative}
            if path.is_symlink():
                target = path.resolve(strict=True)
                if not _is_within(target, allowed_roots):
                    raise ValueError(
                        f"symlink {relative} resolves outside allowed roots: {target}; "
                        "pass --allow-symlink-root for an immutable Hugging Face cache root"
                    )
                if not target.is_file():
                    raise ValueError(f"checkpoint symlink does not resolve to a regular file: {relative}")
                entry.update(kind="symlink", link_target=os.readlink(path), resolved_target=str(target))
                content_path = target
            elif path.is_file():
                entry["kind"] = "file"
                content_path = path
            else:
                raise ValueError(f"unsupported checkpoint entry: {relative}")
            entry.update(size=content_path.stat().st_size, sha256=_hash_file(content_path))
            files.append(entry)

    files.sort(key=lambda item: str(item["path"]))
    components = _component_inventory(checkpoint_root, files)
    missing: list[str] = []
    if not components["model_config"]["path"]:
        missing.append("config.json")
    if not components["weights"]["files"]:
        missing.append("model weights")
    if not components["tokenizer"]["files"]:
        missing.append("tokenizer files")
    if not components["chat_template"]["source"]:
        missing.append("chat template")
    if not components["generation_config"]["path"]:
        missing.append("generation_config.json")
    if require_components and missing:
        raise ValueError(f"checkpoint is missing required components: {', '.join(missing)}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "resolved_root": str(checkpoint_root),
        "allowed_symlink_roots": [str(path) for path in allowed_roots],
        "file_count": len(files),
        "files": files,
        "components": components,
        "component_check": {"complete": not missing, "missing": missing},
    }
    manifest["manifest_sha256"] = _canonical_digest(manifest)
    return manifest


def write_checkpoint_manifest(
    manifest: dict[str, Any], output: str | Path, checksums_output: str | Path | None = None
) -> tuple[Path, Path]:
    output_path = Path(output)
    checksum_path = Path(checksums_output) if checksums_output else output_path.with_name("checkpoint-files.sha256")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum_path.write_text("".join(f"{entry['sha256']}  {entry['path']}\n" for entry in manifest["files"]))
    return output_path, checksum_path


def verify_checkpoint_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text())
    expected_manifest_digest = manifest.pop("manifest_sha256", None)
    actual_manifest_digest = _canonical_digest(manifest)
    manifest["manifest_sha256"] = expected_manifest_digest
    errors: list[str] = []
    if expected_manifest_digest != actual_manifest_digest:
        errors.append("manifest content digest does not match")

    root = Path(manifest["resolved_root"])
    allowed_roots = tuple(Path(value).resolve(strict=True) for value in manifest["allowed_symlink_roots"])
    for entry in manifest["files"]:
        relative = str(entry["path"])
        candidate = root / relative
        if not candidate.exists():
            errors.append(f"missing file: {relative}")
            continue
        if candidate.is_symlink():
            target = candidate.resolve(strict=True)
            if not _is_within(target, allowed_roots):
                errors.append(f"symlink escaped allowed roots: {relative}")
                continue
            content_path = target
        else:
            content_path = candidate
        if not content_path.is_file():
            errors.append(f"not a regular file: {relative}")
            continue
        if content_path.stat().st_size != int(entry["size"]):
            errors.append(f"size mismatch: {relative}")
        elif _hash_file(content_path) != entry["sha256"]:
            errors.append(f"SHA-256 mismatch: {relative}")

    return {
        "ok": not errors,
        "model_id": manifest.get("model_id"),
        "revision": manifest.get("revision"),
        "files_checked": len(manifest.get("files", [])),
        "manifest_sha256": expected_manifest_digest,
        "errors": errors,
    }
