from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .datasets import write_rows


EU24_NON_ENGLISH = (
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "ga",
    "hr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
)


def _score(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_add(
    heap: list[tuple[int, str, dict[str, Any]]],
    *,
    limit: int,
    score: int,
    group: str,
    row: dict[str, Any],
) -> None:
    item = (-score, group, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _selected(heap: list[tuple[int, str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def _prepare(row: dict[str, Any], pool: str) -> dict[str, Any]:
    prepared = dict(row)
    ground_truth = prepared.get("ground_truth")
    if isinstance(ground_truth, list):
        if len(ground_truth) != 1:
            raise ValueError(f"row {prepared.get('id')} has {len(ground_truth)} ground truths")
        ground_truth = ground_truth[0]
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError(f"row {prepared.get('id')} has no scalar ground truth")
    if not prepared.get("messages") or not prepared.get("verifier_kind"):
        raise ValueError(f"row {prepared.get('id')} is missing the prompt or verifier contract")
    prepared["ground_truth"] = ground_truth
    prepared["oellm_source_dataset"] = str(prepared.get("dataset", "oellm-math-rlvr"))
    prepared["dataset"] = "math"
    prepared["oellm_curriculum_pool"] = pool
    return prepared


def _iter_rows(source: Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("building EU math pools requires pyarrow") from error
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "semantic_groups": len({str(row["semantic_group_id"]) for row in rows}),
        "by_language": dict(sorted(Counter(str(row["language"]) for row in rows).items())),
        "by_difficulty": dict(sorted(Counter(str(row["difficulty"]) for row in rows).items())),
        "by_subdomain": dict(sorted(Counter(str(row["subdomain"]) for row in rows).items())),
    }


def build_eu_math_pools(
    source: str | Path,
    output_dir: str | Path,
    *,
    eu_per_language: int = 128,
    english_count: int = 2048,
    min_difficulty: int = 3,
    max_difficulty: int = 5,
    seed: int = 20260914,
    languages: Iterable[str] = EU24_NON_ENGLISH,
) -> dict[str, Any]:
    """Build balanced EU and disjoint English RLVR pools.

    Each multilingual semantic group is deterministically assigned to exactly one
    language before sampling. This maximizes problem diversity and prevents a
    translation of the same latent problem from appearing in both pools.
    """
    source_path = Path(source)
    output = Path(output_dir)
    language_codes = tuple(languages)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not language_codes or len(language_codes) != len(set(language_codes)):
        raise ValueError("languages must be non-empty and unique")
    if "en" in language_codes:
        raise ValueError("the EU treatment language list must not contain English")
    if eu_per_language < 1 or english_count < 1:
        raise ValueError("pool sizes must be positive")
    if min_difficulty > max_difficulty:
        raise ValueError("min_difficulty cannot exceed max_difficulty")

    eu_heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {language: [] for language in language_codes}
    for row in _iter_rows(source_path):
        language = str(row.get("language", ""))
        if language not in eu_heaps:
            continue
        difficulty = int(row.get("difficulty", 0))
        if not min_difficulty <= difficulty <= max_difficulty:
            continue
        group = str(row.get("semantic_group_id") or row.get("id") or "")
        if not group:
            continue
        assigned = language_codes[_score(seed, group, "language") % len(language_codes)]
        if language != assigned:
            continue
        _bounded_add(
            eu_heaps[language],
            limit=eu_per_language,
            score=_score(seed, group, "eu"),
            group=group,
            row=row,
        )

    shortages = {language: len(rows) for language, rows in eu_heaps.items() if len(rows) < eu_per_language}
    if shortages:
        raise ValueError(f"not enough assigned EU rows: {shortages}")
    eu_rows = [_prepare(row, "eu") for language in language_codes for row in _selected(eu_heaps[language])]
    eu_groups = {str(row["semantic_group_id"]) for row in eu_rows}

    english_heap: list[tuple[int, str, dict[str, Any]]] = []
    for row in _iter_rows(source_path):
        if row.get("language") != "en":
            continue
        difficulty = int(row.get("difficulty", 0))
        if not min_difficulty <= difficulty <= max_difficulty:
            continue
        group = str(row.get("semantic_group_id") or row.get("id") or "")
        if not group or group in eu_groups:
            continue
        _bounded_add(
            english_heap,
            limit=english_count,
            score=_score(seed, group, "english"),
            group=group,
            row=row,
        )
    if len(english_heap) < english_count:
        raise ValueError(f"not enough disjoint English rows: found {len(english_heap)}, requested {english_count}")
    english_rows = [_prepare(row, "english_replay") for row in _selected(english_heap)]
    english_groups = {str(row["semantic_group_id"]) for row in english_rows}
    if eu_groups & english_groups:
        raise AssertionError("EU and English pools overlap by semantic group")

    output.mkdir(parents=True, exist_ok=True)
    eu_path = output / "eu-math.parquet"
    english_path = output / "english-replay.parquet"
    write_rows(eu_rows, eu_path)
    write_rows(english_rows, english_path)
    manifest = {
        "schema": "oellm-eu-math-pools-v1",
        "seed": seed,
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "difficulty_range": [min_difficulty, max_difficulty],
        "eu_languages": list(language_codes),
        "semantic_group_overlap": 0,
        "eu": {**_summary(eu_rows), "path": str(eu_path), "sha256": _sha256(eu_path)},
        "english_replay": {
            **_summary(english_rows),
            "path": str(english_path),
            "sha256": _sha256(english_path),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
