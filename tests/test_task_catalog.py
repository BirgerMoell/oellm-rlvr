from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from oellm_rlvr.task_catalog import load_task_catalog, profile_task_attempts


def _task(task_id: str, *, split: str = "train", cluster: str | None = None, language: str = "en") -> dict:
    return {
        "id": task_id,
        "split": split,
        "domain": "math",
        "language": language,
        "semantic_cluster_id": cluster or f"cluster-{task_id}",
        "prompt_sha256": ("a" if task_id == "one" else "b" if task_id == "two" else "c") * 64,
        "source": {
            "dataset": "org/data",
            "revision": "d" * 40,
            "split": split,
            "row_id": task_id,
            "license": "Apache-2.0",
        },
        "verifier": {"kind": "math", "revision": "v1", "config_sha256": "e" * 64},
    }


def _write_catalog(path: Path, tasks: list[dict]) -> None:
    path.write_text(yaml.safe_dump({"version": 1, "name": "test", "tasks": tasks}, sort_keys=False))


def test_catalog_rejects_semantic_leakage(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    _write_catalog(
        path,
        [_task("one", cluster="shared"), _task("eval", split="evaluation", cluster="shared")],
    )
    with pytest.raises(ValidationError, match="semantic-cluster overlap"):
        load_task_catalog(path)


def test_profile_builds_domain_language_and_error_artifacts(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.yaml"
    _write_catalog(catalog, [_task("one"), _task("two", language="sv")])
    attempts = tmp_path / "attempts.jsonl"
    rows = []
    for task_id, passes in (("one", 4), ("two", 2)):
        for sample_index in range(8):
            passed = sample_index < passes
            rows.append(
                {
                    "task_id": task_id,
                    "sample_index": sample_index,
                    "status": "pass" if passed else "model_failure",
                    "reward": float(passed),
                    "response_tokens": 20,
                    "policy_version": "parent",
                }
            )
    attempts.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = profile_task_attempts(catalog, attempts, tmp_path / "profile", samples_per_prompt=8)

    assert report["ok"] is True
    assert report["informative_prompt_fraction"] == 1.0
    assert Path(report["artifacts"]["task_profile"]).exists()
    histograms = json.loads((tmp_path / "profile/pass-rate-histograms.json").read_text())
    assert histograms["by_language"]["sv"]["[0.10,0.90]"] == 1


def test_profile_separates_infrastructure_failures(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.yaml"
    _write_catalog(catalog, [_task("one")])
    attempts = tmp_path / "attempts.jsonl"
    rows = [
        {
            "task_id": "one",
            "sample_index": sample_index,
            "status": "environment_error" if sample_index == 0 else "model_failure",
            "reward": 0,
            "policy_version": "parent",
            "verifier_message": "container failed" if sample_index == 0 else None,
        }
        for sample_index in range(8)
    ]
    attempts.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = profile_task_attempts(catalog, attempts, tmp_path / "profile", samples_per_prompt=8)

    assert report["ok"] is False
    assert report["verifier_environment_error_rate"] == 0.125
    errors = json.loads((tmp_path / "profile/verifier-error-report.json").read_text())
    assert errors["by_status"] == {"environment_error": 1}
