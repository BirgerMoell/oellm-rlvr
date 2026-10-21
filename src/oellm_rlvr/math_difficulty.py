"""Checkpoint-specific difficulty annotations from *all* frozen survey attempts.

Never derive these labels from an RL trainer's accepted/filtered rollouts: active
sampling selects for mixed rewards and makes those estimates unrepresentative.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def _band(correct: int, attempts: int) -> str:
    rate = correct / attempts
    if correct == 0:
        return "unsolved_at_k"
    if correct == attempts:
        return "solved_at_k"
    if rate < 0.375:
        return "hard"
    if rate <= 0.625:
        return "medium"
    return "easy"


def _wilson(correct: int, attempts: int) -> tuple[float, float]:
    z = 1.96
    rate = correct / attempts
    denominator = 1 + z * z / attempts
    centre = (rate + z * z / (2 * attempts)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / attempts + z * z / (4 * attempts * attempts)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def annotate_predictions(
    predictions: str | Path,
    output_csv: str | Path,
    *,
    expected_samples: int | None = None,
) -> dict[str, Any]:
    """Write one difficulty row per problem; fail on incomplete or ambiguous data."""
    source = Path(predictions)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    with source.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = str(record["id"])
            sample_index = int(record["sample_index"])
            key = (problem_id, sample_index)
            if key in seen:
                raise ValueError(f"duplicate survey attempt {key} at line {line_number}")
            seen.add(key)
            groups[problem_id].append(record)
    if not groups:
        raise ValueError("survey has no predictions")
    if expected_samples is not None and any(len(group) != expected_samples for group in groups.values()):
        raise ValueError("survey is incomplete: inconsistent attempts per problem")

    annotations: list[dict[str, Any]] = []
    semantic_ids: set[str] = set()
    for problem_id, attempts in groups.items():
        indices = {int(row["sample_index"]) for row in attempts}
        if indices != set(range(len(attempts))):
            raise ValueError(f"noncontiguous sample indices for {problem_id}")
        metadata = attempts[0].get("metadata") or {}
        if any((row.get("metadata") or {}) != metadata for row in attempts):
            raise ValueError(f"conflicting metadata for {problem_id}")
        semantic_id = str(metadata.get("semantic_group_id") or problem_id)
        if semantic_id in semantic_ids:
            raise ValueError(f"semantic group appears more than once: {semantic_id}")
        semantic_ids.add(semantic_id)
        analyses = [row["analysis"] for row in attempts]
        correct = sum(bool(item["correct"]) for item in analyses)
        count = len(analyses)
        low, high = _wilson(correct, count)
        annotations.append({
            "id": problem_id,
            "semantic_group_id": semantic_id,
            "language": metadata.get("canonical_language") or metadata.get("language") or "",
            "subdomain": metadata.get("subdomain") or "",
            "generator_family": metadata.get("generator_family") or "",
            "source_difficulty": metadata.get("difficulty") or "",
            "source_pool": metadata.get("oellm_curriculum_pool") or "",
            "attempts": count,
            "correct": correct,
            "sample_accuracy": correct / count,
            "pass_at_k": int(correct > 0),
            "difficulty_band": _band(correct, count),
            "wilson_95_low": low,
            "wilson_95_high": high,
            "length_stop_rate": fmean(bool(item.get("length_stopped")) for item in analyses),
            "high_repetition_rate": fmean(bool(item.get("high_repetition")) for item in analyses),
        })
    annotations.sort(key=lambda row: (row["sample_accuracy"], row["id"]))
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotations[0]))
        writer.writeheader()
        writer.writerows(annotations)
    return {
        "schema": "oellm-math-difficulty-v1",
        "predictions": str(source),
        "annotations": str(output),
        "problems": len(annotations),
        "attempts": len(seen),
        "difficulty_bands": dict(sorted(Counter(row["difficulty_band"] for row in annotations).items())),
        "by_language": {
            language: {"problems": len(rows), "sample_accuracy": fmean(row["sample_accuracy"] for row in rows)}
            for language, rows in sorted(_group_by(annotations, "language").items())
        },
    }


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped
