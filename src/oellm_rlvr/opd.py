from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import RunConfig
from .datasets import write_rows


def _canonical_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_id_fingerprint(
    root: Path, manifest: dict[str, Any], label: str, errors: list[str]
) -> str | None:
    components = manifest.get("components", {})
    token_files = components.get("tokenizer", {}).get("files", [])
    entries = {str(item.get("path")): item for item in manifest.get("files", [])}
    fingerprint_parts: dict[str, str] = {}
    for relative in token_files:
        entry = entries.get(str(relative))
        candidate = root / str(relative)
        if entry is None or not candidate.is_file():
            errors.append(f"{label} tokenizer file is missing from checkpoint or manifest: {relative}")
            continue
        actual_hash = _hash_file(candidate)
        if actual_hash != entry.get("sha256"):
            errors.append(f"{label} tokenizer file does not match its manifest: {relative}")
        if Path(str(relative)).name == "tokenizer_config.json":
            try:
                tokenizer_config = json.loads(candidate.read_text())
            except json.JSONDecodeError as error:
                errors.append(f"{label} tokenizer_config.json is invalid: {error}")
                continue
            if not isinstance(tokenizer_config, dict):
                errors.append(f"{label} tokenizer_config.json must contain an object")
                continue
            # Prompt rendering does not alter token IDs. Compare it separately
            # while keeping special-token and added-token configuration strict.
            tokenizer_config.pop("chat_template", None)
            fingerprint_parts[str(relative)] = _canonical_digest(tokenizer_config)
        else:
            fingerprint_parts[str(relative)] = actual_hash
    return _canonical_digest(fingerprint_parts) if fingerprint_parts else None


def _read_manifest(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    manifest_path = Path(path).expanduser()
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"cannot read manifest {manifest_path}: {error}"]
    expected = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if expected != _canonical_digest(unsigned):
        errors.append(f"manifest content digest does not match: {manifest_path}")
    return manifest, errors


def _check_checkpoint(
    label: str,
    *,
    model_id: str,
    revision: str,
    model_path: str,
    manifest_path: str,
) -> tuple[dict[str, Any], list[str]]:
    manifest, errors = _read_manifest(manifest_path)
    if not manifest:
        return {}, errors
    try:
        configured_root = Path(model_path).expanduser().resolve(strict=True)
    except OSError as error:
        errors.append(f"{label} checkpoint is unavailable: {error}")
        configured_root = Path(model_path).expanduser().resolve()
    recorded_root = Path(str(manifest.get("resolved_root", ""))).expanduser().resolve()
    if configured_root != recorded_root:
        errors.append(f"{label} manifest root {recorded_root} != configured checkpoint {configured_root}")
    if manifest.get("model_id") != model_id:
        errors.append(f"{label} manifest model_id {manifest.get('model_id')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        errors.append(f"{label} manifest revision {manifest.get('revision')!r} != {revision!r}")
    components = manifest.get("components", {})
    tokenizer_sha = _token_id_fingerprint(configured_root, manifest, label, errors)
    chat_template_sha = components.get("chat_template", {}).get("sha256")
    if not tokenizer_sha:
        errors.append(f"{label} manifest has no tokenizer fingerprint")
    if not chat_template_sha:
        errors.append(f"{label} manifest has no chat-template fingerprint")
    return {
        "model_id": manifest.get("model_id"),
        "revision": manifest.get("revision"),
        "resolved_root": str(recorded_root),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "tokenizer_sha256": tokenizer_sha,
        "chat_template_sha256": chat_template_sha,
    }, errors


def preflight_opd(config: RunConfig) -> dict[str, Any]:
    """Validate immutable OPD checkpoint contracts without re-hashing weights.

    `checkpoint-manifest` performs the expensive content hashing once. This
    preflight authenticates those manifests, their configured roots/revisions,
    and the tokenizer contract needed for token-level teacher log-probabilities.
    """
    if config.backend.kind != "verl_opd" or config.distillation is None:
        raise ValueError("OPD preflight requires a verl_opd run config")
    assert config.model.local_path and config.model.revision and config.model.manifest_path
    student, errors = _check_checkpoint(
        "student",
        model_id=config.model.name_or_path,
        revision=config.model.revision,
        model_path=config.model.local_path,
        manifest_path=config.model.manifest_path,
    )
    teachers: list[dict[str, Any]] = []
    warnings: list[str] = []
    for teacher in config.distillation.teachers:
        summary, teacher_errors = _check_checkpoint(
            f"teacher {teacher.name}",
            model_id=teacher.name_or_path,
            revision=teacher.revision,
            model_path=teacher.model_path,
            manifest_path=teacher.manifest_path,
        )
        summary.update(name=teacher.name, key=teacher.key)
        teachers.append(summary)
        errors.extend(teacher_errors)
        student_tokenizer = student.get("tokenizer_sha256")
        teacher_tokenizer = summary.get("tokenizer_sha256")
        if student_tokenizer and teacher_tokenizer and student_tokenizer != teacher_tokenizer:
            errors.append(f"teacher {teacher.name} tokenizer does not match the student tokenizer")
        student_template = student.get("chat_template_sha256")
        teacher_template = summary.get("chat_template_sha256")
        if student_template and teacher_template and student_template != teacher_template:
            warnings.append(
                f"teacher {teacher.name} chat template differs; verl sends student token IDs directly, "
                "but the teacher may have been tuned for a different prompt convention"
            )
    return {"ok": not errors, "student": student, "teachers": teachers, "warnings": warnings, "errors": errors}


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Parquet input requires pyarrow (install oellm-rlvr[data])") from error
        return list(pq.read_table(source).to_pylist())
    if source.suffix == ".json":
        value = json.loads(source.read_text())
        if not isinstance(value, list):
            raise ValueError("JSON OPD input must contain a top-level list of rows")
        return value
    rows: list[dict[str, Any]] = []
    with source.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"OPD input line {line_number} is not an object")
                rows.append(value)
    return rows


def prepare_opd_dataset(
    source: str | Path,
    output: str | Path,
    *,
    data_source: str,
    ability: str = "general",
    require_ground_truth: bool = False,
) -> dict[str, Any]:
    """Convert project prompt rows to verl's prompt-only RL/OPD schema."""
    if not data_source.strip():
        raise ValueError("data_source must be non-empty (it is also the multi-teacher routing key)")
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(_read_rows(source)):
        messages = row.get("prompt", row.get("messages"))
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"row {index} must contain a non-empty messages or prompt list")
        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user"}:
                raise ValueError(f"row {index} prompts may contain only system and user messages")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"row {index} has an empty prompt message")
            normalized_messages.append({"role": str(message["role"]), "content": content})
        ground_truth = row.get("ground_truth", row.get("answer", ""))
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0] if ground_truth else ""
        if ground_truth is None:
            ground_truth = ""
        if require_ground_truth and not str(ground_truth).strip():
            raise ValueError(f"row {index} has no ground truth for hybrid OPD+RLVR")
        extra_info = dict(row.get("extra_info", {})) if isinstance(row.get("extra_info", {}), dict) else {}
        extra_info.setdefault("index", index)
        for key in ("id", "dataset", "verifier_kind", "semantic_group_id"):
            if key in row:
                extra_info.setdefault(key, row[key])
        converted.append(
            {
                "data_source": data_source,
                "prompt": normalized_messages,
                "ability": ability,
                "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
                "extra_info": extra_info,
            }
        )
    if not converted:
        raise ValueError("OPD input contains no rows")
    write_rows(converted, output)
    return {"ok": True, "source": str(source), "output": str(output), "rows": len(converted), "data_source": data_source}
