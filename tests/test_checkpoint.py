from __future__ import annotations

import json
from pathlib import Path

import pytest

from oellm_rlvr.checkpoint import (
    build_checkpoint_manifest,
    verify_checkpoint_manifest,
    write_checkpoint_manifest,
)


def _checkpoint(root: Path) -> None:
    (root / "config.json").write_text(json.dumps({"model_type": "test", "architectures": ["TestModel"]}))
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (root / "generation_config.json").write_text("{}")


def test_manifest_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _checkpoint(model)
    first = build_checkpoint_manifest(model, model_id="org/model", revision="a" * 40)
    second = build_checkpoint_manifest(model, model_id="org/model", revision="a" * 40)

    assert first == second
    assert first["component_check"]["complete"] is True
    assert first["components"]["chat_template"]["source"] == "tokenizer_config.json#chat_template"

    output, checksums = write_checkpoint_manifest(first, tmp_path / "artifacts/checkpoint-manifest.json")
    report = verify_checkpoint_manifest(output)
    assert report["ok"] is True
    assert len(checksums.read_text().splitlines()) == 5


def test_manifest_detects_changed_content(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _checkpoint(model)
    manifest = build_checkpoint_manifest(model, model_id="org/model", revision="a" * 40)
    output, _ = write_checkpoint_manifest(manifest, tmp_path / "checkpoint-manifest.json")
    (model / "model.safetensors").write_bytes(b"different")

    report = verify_checkpoint_manifest(output)
    assert report["ok"] is False
    assert any("model.safetensors" in error for error in report["errors"])


def test_external_symlink_must_be_explicitly_allowed(tmp_path: Path) -> None:
    model = tmp_path / "snapshot"
    cache = tmp_path / "blobs"
    model.mkdir()
    cache.mkdir()
    _checkpoint(model)
    external = cache / "weights"
    external.write_bytes(b"cached")
    (model / "cached.safetensors").symlink_to(external)

    with pytest.raises(ValueError, match="outside allowed roots"):
        build_checkpoint_manifest(model, model_id="org/model", revision="a" * 40)

    manifest = build_checkpoint_manifest(
        model,
        model_id="org/model",
        revision="a" * 40,
        allowed_symlink_roots=[cache],
    )
    cached = next(item for item in manifest["files"] if item["path"] == "cached.safetensors")
    assert cached["kind"] == "symlink"


def test_missing_components_are_rejected_by_default(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="missing required components"):
        build_checkpoint_manifest(tmp_path, model_id="org/model", revision="a" * 40)

    manifest = build_checkpoint_manifest(
        tmp_path,
        model_id="org/model",
        revision="a" * 40,
        require_components=False,
    )
    assert manifest["component_check"]["complete"] is False
