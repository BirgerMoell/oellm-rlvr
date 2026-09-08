from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .datasets import write_rows


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskSource(StrictModel):
    dataset: str
    revision: str
    split: str
    row_id: str
    license: str


class VerifierContract(StrictModel):
    kind: Literal["math", "code_tests", "json_schema", "function_call", "harbor"]
    revision: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CatalogTask(StrictModel):
    id: str
    split: Literal["train", "evaluation"]
    profile: bool = True
    domain: Literal["math", "code", "structured_output", "function_calling", "agentic"]
    language: str
    semantic_cluster_id: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: TaskSource
    verifier: VerifierContract
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskCatalog(StrictModel):
    version: Literal[1] = 1
    name: str
    tasks: list[CatalogTask] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicates_and_leakage(self) -> TaskCatalog:
        ids = [task.id for task in self.tasks]
        duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate task IDs: {', '.join(duplicates)}")
        train = [task for task in self.tasks if task.split == "train"]
        evaluation = [task for task in self.tasks if task.split == "evaluation"]
        train_clusters = {task.semantic_cluster_id for task in train}
        eval_clusters = {task.semantic_cluster_id for task in evaluation}
        cluster_overlap = sorted(train_clusters & eval_clusters)
        if cluster_overlap:
            raise ValueError(f"train/evaluation semantic-cluster overlap: {', '.join(cluster_overlap)}")
        train_prompts = {task.prompt_sha256 for task in train}
        eval_prompts = {task.prompt_sha256 for task in evaluation}
        if train_prompts & eval_prompts:
            raise ValueError("train/evaluation prompt SHA-256 overlap")
        return self

    def summary(self) -> dict[str, Any]:
        profiled = [task for task in self.tasks if task.split == "train" and task.profile]
        return {
            "valid": True,
            "name": self.name,
            "tasks": len(self.tasks),
            "profiled_training_tasks": len(profiled),
            "by_split": dict(sorted(Counter(task.split for task in self.tasks).items())),
            "by_domain": dict(sorted(Counter(task.domain for task in profiled).items())),
            "by_language": dict(sorted(Counter(task.language for task in profiled).items())),
            "train_evaluation_prompt_overlap": 0,
            "train_evaluation_semantic_cluster_overlap": 0,
        }


class ProfileAttempt(StrictModel):
    task_id: str
    sample_index: int = Field(ge=0)
    status: Literal["pass", "model_failure", "verifier_error", "environment_error"]
    reward: float = Field(ge=0.0, le=1.0)
    response_tokens: int | None = Field(default=None, ge=0)
    policy_version: str
    verifier_message: str | None = None


def load_task_catalog(path: str | Path) -> TaskCatalog:
    return TaskCatalog.model_validate(yaml.safe_load(Path(path).read_text()))


def _read_attempts(path: str | Path) -> list[ProfileAttempt]:
    source = Path(path)
    rows: list[dict[str, Any]]
    if source.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Parquet input requires pyarrow (install oellm-rlvr[data])") from error
        rows = pq.read_table(source).to_pylist()
    else:
        rows = []
        with source.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"attempt line {line_number} is not a JSON object")
                rows.append(value)
    return [ProfileAttempt.model_validate(row) for row in rows]


def _read_profile_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Parquet input requires pyarrow (install oellm-rlvr[data])") from error
        return pq.read_table(source).to_pylist()
    rows: list[dict[str, Any]] = []
    with source.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"profile line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def build_curriculum_pools(profile_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    rows = _read_profile_rows(profile_path)
    required = {"task_id", "domain", "pass_rate", "infrastructure_errors"}
    buckets: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    seen: set[str] = set()
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"profile row {index} is missing: {', '.join(missing)}")
        task_id = str(row["task_id"])
        if task_id in seen:
            raise ValueError(f"duplicate profiled task ID: {task_id}")
        seen.add(task_id)
        domain = str(row["domain"])
        pass_rate = float(row["pass_rate"])
        errors = int(row["infrastructure_errors"])
        if not 0 <= pass_rate <= 1:
            raise ValueError(f"profile task {task_id} has pass_rate outside [0, 1]")
        if errors:
            bucket = "infrastructure_reject"
        elif pass_rate == 0:
            bucket = "impossible"
        elif pass_rate < 0.25:
            bucket = "hard"
        elif pass_rate < 0.625:
            bucket = "medium"
        elif pass_rate < 1:
            bucket = "easy"
        else:
            bucket = "saturated"
        buckets[domain][bucket].append(task_id)

    bucket_order = ("easy", "medium", "hard", "saturated", "impossible", "infrastructure_reject")
    by_domain = {
        domain: {bucket: sorted(values.get(bucket, [])) for bucket in bucket_order}
        for domain, values in sorted(buckets.items())
    }
    counts = {
        domain: {bucket: len(values[bucket]) for bucket in bucket_order}
        for domain, values in by_domain.items()
    }
    report = {
        "ok": True,
        "source_profile": str(profile_path),
        "tasks": len(rows),
        "policy": {
            "easy": "0.625 <= pass_rate < 1.0",
            "medium": "0.25 <= pass_rate < 0.625",
            "hard": "0.0 < pass_rate < 0.25",
            "saturated": "pass_rate == 1.0; exclude from group-relative updates",
            "impossible": "pass_rate == 0.0; route to SFT/data repair/later checkpoint",
            "infrastructure_reject": "one or more verifier/environment errors; repair before reuse",
        },
        "by_domain": by_domain,
        "counts": counts,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _bin(pass_rate: float) -> str:
    if pass_rate == 0:
        return "0"
    if pass_rate < 0.10:
        return "(0,0.10)"
    if pass_rate <= 0.90:
        return "[0.10,0.90]"
    if pass_rate < 1:
        return "(0.90,1)"
    return "1"


def _histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    order = ("0", "(0,0.10)", "[0.10,0.90]", "(0.90,1)", "1")
    counts = Counter(str(row["pass_rate_bin"]) for row in rows)
    return {label: counts.get(label, 0) for label in order}


def profile_task_attempts(
    catalog_path: str | Path,
    attempts_path: str | Path,
    output_dir: str | Path,
    *,
    samples_per_prompt: int,
) -> dict[str, Any]:
    if samples_per_prompt < 2:
        raise ValueError("samples_per_prompt must be at least two to estimate reward variance")
    catalog = load_task_catalog(catalog_path)
    tasks = {task.id: task for task in catalog.tasks if task.split == "train" and task.profile}
    attempts = _read_attempts(attempts_path)
    unknown = sorted({attempt.task_id for attempt in attempts} - tasks.keys())
    if unknown:
        raise ValueError(f"attempts reference unknown or non-profile task IDs: {', '.join(unknown)}")
    grouped: dict[str, list[ProfileAttempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.task_id].append(attempt)

    missing_tasks = sorted(tasks.keys() - grouped.keys())
    if missing_tasks:
        raise ValueError(f"profile attempts are missing tasks: {', '.join(missing_tasks)}")

    profile_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for task_id, task in tasks.items():
        values = grouped[task_id]
        indexes = sorted(attempt.sample_index for attempt in values)
        if indexes != list(range(samples_per_prompt)):
            raise ValueError(
                f"task {task_id} needs sample indexes 0..{samples_per_prompt - 1}; found {indexes}"
            )
        for attempt in values:
            key = (attempt.task_id, attempt.sample_index)
            if key in seen:
                raise ValueError(f"duplicate profile attempt: {attempt.task_id}/{attempt.sample_index}")
            seen.add(key)
            if attempt.status in {"verifier_error", "environment_error"}:
                error_rows.append(
                    {
                        "task_id": task_id,
                        "sample_index": attempt.sample_index,
                        "status": attempt.status,
                        "message": attempt.verifier_message,
                    }
                )
        model_outcomes = [value for value in values if value.status in {"pass", "model_failure"}]
        rewards = [value.reward for value in model_outcomes]
        pass_rate = fmean(value.status == "pass" for value in model_outcomes) if model_outcomes else 0.0
        reward_mean = fmean(rewards) if rewards else 0.0
        reward_variance = fmean((reward - reward_mean) ** 2 for reward in rewards) if rewards else 0.0
        infrastructure_errors = len(values) - len(model_outcomes)
        profile_rows.append(
            {
                "task_id": task_id,
                "domain": task.domain,
                "language": task.language,
                "semantic_cluster_id": task.semantic_cluster_id,
                "policy_version": sorted({value.policy_version for value in values}),
                "samples": len(values),
                "valid_model_outcomes": len(model_outcomes),
                "passes": sum(value.status == "pass" for value in model_outcomes),
                "pass_rate": pass_rate,
                "pass_rate_bin": _bin(pass_rate),
                "reward_mean": reward_mean,
                "reward_variance": reward_variance,
                "infrastructure_errors": infrastructure_errors,
                "admitted": infrastructure_errors == 0 and 0.10 <= pass_rate <= 0.90,
            }
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    profile_path = destination / "task-profile.parquet"
    write_rows(profile_rows, profile_path)
    by_domain = {
        key: _histogram(values)
        for key, values in sorted(
            (key, [row for row in profile_rows if row["domain"] == key]) for key in {row["domain"] for row in profile_rows}
        )
    }
    by_language = {
        key: _histogram(values)
        for key, values in sorted(
            (key, [row for row in profile_rows if row["language"] == key]) for key in {row["language"] for row in profile_rows}
        )
    }
    histograms = {"all": _histogram(profile_rows), "by_domain": by_domain, "by_language": by_language}
    histogram_path = destination / "pass-rate-histograms.json"
    histogram_path.write_text(json.dumps(histograms, indent=2, sort_keys=True) + "\n")
    error_rate = len(error_rows) / len(attempts) if attempts else 0.0
    errors_report = {
        "attempts": len(attempts),
        "errors": len(error_rows),
        "error_rate": error_rate,
        "by_status": dict(sorted(Counter(row["status"] for row in error_rows).items())),
        "records": error_rows,
    }
    errors_path = destination / "verifier-error-report.json"
    errors_path.write_text(json.dumps(errors_report, indent=2, sort_keys=True) + "\n")
    informative_fraction = fmean(bool(row["admitted"]) for row in profile_rows)
    gates = {
        "informative_prompt_fraction_at_least_0_60": informative_fraction >= 0.60,
        "verifier_environment_error_rate_below_0_02": error_rate < 0.02,
        "train_evaluation_leakage_absent": True,
    }
    summary = {
        "ok": all(gates.values()),
        "catalog": catalog.name,
        "tasks": len(profile_rows),
        "attempts": len(attempts),
        "samples_per_prompt": samples_per_prompt,
        "informative_prompt_fraction": informative_fraction,
        "verifier_environment_error_rate": error_rate,
        "gates": gates,
        "artifacts": {
            "task_profile": str(profile_path),
            "pass_rate_histograms": str(histogram_path),
            "verifier_error_report": str(errors_path),
        },
    }
    (destination / "profile-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
