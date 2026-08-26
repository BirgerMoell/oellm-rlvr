from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CodeTaskManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    image: str
    seed_files: dict[str, str] = Field(default_factory=dict)
    test_files: dict[str, str]
    ground_truth: str | list[str] = ""
    max_steps: int = Field(default=32, ge=1)
    copies: int = Field(default=1, ge=1)

    @field_validator("test_files")
    @classmethod
    def test_entrypoint(cls, value: dict[str, str]) -> dict[str, str]:
        if "test.sh" not in value:
            raise ValueError("test_files must contain test.sh")
        return value


def _safe_write(root: Path, relative: str, content: str) -> None:
    target = (root / relative).resolve()
    resolved_root = root.resolve()
    if resolved_root not in target.parents:
        raise ValueError(f"path escapes task directory: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def write_rows(rows: list[dict[str, object]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Parquet output requires pyarrow (install oellm-rlvr[data])") from error
        pq.write_table(pa.Table.from_pylist(rows), destination)
        return
    with destination.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_math_smoke(path: str | Path, count: int = 64) -> None:
    rows = [
        {
            "messages": [{"role": "user", "content": f"Compute {n} + {n + 1}. Give the final answer in \\boxed{{}}."}],
            "ground_truth": str(2 * n + 1),
            "dataset": "math",
            "oellm_source_dataset": "oellm_math_smoke",
        }
        for n in range(1, count + 1)
    ]
    write_rows(rows, path)


def _sample_parquet_rows(
    path: str | Path,
    count: int,
    *,
    language: str | None = None,
    min_difficulty: int | None = None,
    max_difficulty: int | None = None,
    diverse_by: str | None = None,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    if min_difficulty is not None and max_difficulty is not None and min_difficulty > max_difficulty:
        raise ValueError("min_difficulty cannot exceed max_difficulty")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet input requires pyarrow (install oellm-rlvr[data])") from error
    parquet = pq.ParquetFile(path)
    selected: list[dict[str, Any]] = []
    semantic_groups: set[str] = set()
    diversity_values: set[str] = set()
    for batch in parquet.iter_batches(batch_size=max(64, count * 4)):
        for row in batch.to_pylist():
            row_language = row.get("language") or row.get("prompt_language")
            if language is not None and row_language != language:
                continue
            difficulty = row.get("difficulty")
            if min_difficulty is not None and (difficulty is None or int(difficulty) < min_difficulty):
                continue
            if max_difficulty is not None and (difficulty is None or int(difficulty) > max_difficulty):
                continue
            group = str(row.get("semantic_group_id", row.get("id", "")))
            if group in semantic_groups:
                continue
            diversity_value = str(row.get(diverse_by, "")) if diverse_by else ""
            if diverse_by and (not diversity_value or diversity_value in diversity_values):
                continue
            semantic_groups.add(group)
            if diverse_by:
                diversity_values.add(diversity_value)
            selected.append(row)
            if len(selected) == count:
                return selected
    filters = []
    if language is not None:
        filters.append(f"language={language}")
    if min_difficulty is not None:
        filters.append(f"difficulty>={min_difficulty}")
    if max_difficulty is not None:
        filters.append(f"difficulty<={max_difficulty}")
    if diverse_by:
        filters.append(f"distinct {diverse_by}")
    suffix = f" for {', '.join(filters)}" if filters else ""
    raise ValueError(f"dataset contains only {len(selected)} matching semantic groups{suffix}; requested {count}")


def sample_math_dataset(
    source: str | Path,
    output: str | Path,
    count: int = 4,
    language: str | None = None,
    min_difficulty: int | None = None,
    max_difficulty: int | None = None,
    diverse_by: str | None = None,
) -> None:
    required = {"messages", "ground_truth", "verifier_kind"}
    rows = _sample_parquet_rows(
        source,
        count,
        language=language,
        min_difficulty=min_difficulty,
        max_difficulty=max_difficulty,
        diverse_by=diverse_by,
    )
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"math row {index} is missing columns: {sorted(missing)}")
        ground_truth = row["ground_truth"]
        if isinstance(ground_truth, list):
            if len(ground_truth) != 1:
                raise ValueError(
                    f"math row {index} must contain exactly one ground truth for the math verifier; "
                    f"found {len(ground_truth)}"
                )
            ground_truth = ground_truth[0]
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            raise ValueError(f"math row {index} has an empty or non-string ground truth")
        # rlvr_tokenize_v2 wraps scalar ground truths once to align them with the
        # scalar verifier dispatch key. Keeping the source's singleton list here
        # would therefore produce [[answer]], which MathVerifier cannot compare.
        row["ground_truth"] = ground_truth
        row["oellm_source_dataset"] = str(row.get("dataset", ""))
        # Open-Instruct dispatches ground-truth verifiers by this column.
        row["dataset"] = "math"
    write_rows(rows, output)


def _stdio_test_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import json
import pathlib
import subprocess

cases = json.loads(pathlib.Path("/tests/cases.json").read_text())
solution = pathlib.Path("/workspace/solution.py")
passed = solution.is_file()
failure = "missing /workspace/solution.py" if not passed else ""
if passed:
    for index, case in enumerate(cases):
        try:
            result = subprocess.run(
                ["python", str(solution)],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=float(case["timeout_seconds"]),
                check=False,
            )
        except subprocess.TimeoutExpired:
            passed = False
            failure = f"case {index}: timeout"
            break
        if result.returncode != 0 or result.stdout != case["output"]:
            passed = False
            failure = f"case {index}: exit={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}"
            break
pathlib.Path("/logs/verifier/reward.txt").write_text("1.0\\n" if passed else "0.0\\n")
if not passed:
    raise SystemExit(failure)
PY
"""


def sample_code_dataset(
    source: str | Path,
    output_dir: str | Path,
    image: str,
    count: int = 4,
    copies: int = 1,
    max_steps: int = 6,
    language: str | None = None,
    min_difficulty: int | None = None,
    max_difficulty: int | None = None,
    diverse_by: str | None = None,
) -> tuple[Path, Path]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    rows = _sample_parquet_rows(
        source,
        count,
        language=language,
        min_difficulty=min_difficulty,
        max_difficulty=max_difficulty,
        diverse_by=diverse_by,
    )
    manifests: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        verification = row.get("verification_info")
        if not isinstance(verification, dict) or verification.get("language") != "python":
            raise ValueError(f"code row {index} does not contain Python verification_info")
        cases = verification.get("test_cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"code row {index} has no hidden test cases")
        if any(case.get("type") != "stdin_stdout" for case in cases):
            raise ValueError(f"code row {index} contains a non-stdio test")
        timeout = int(row.get("time_limit_seconds", 2))
        hidden_cases = [
            {"input": case["input"], "output": case["output"], "timeout_seconds": timeout} for case in cases
        ]
        problem = str(row.get("problem") or row["messages"][0]["content"])
        manifests.append(
            {
                "id": str(row["id"]),
                "instruction": problem + "\n\nWrite the complete program to `/workspace/solution.py`.",
                "image": image,
                "seed_files": {"solution.py": "# Write the complete Python 3 solution here.\n"},
                "test_files": {
                    "cases.json": json.dumps(hidden_cases, ensure_ascii=False, separators=(",", ":")),
                    "test.sh": _stdio_test_script(),
                },
                "ground_truth": row.get("ground_truth") or row.get("reference_solution") or "",
                "max_steps": max_steps,
                "copies": copies,
            }
        )
    return _pack_code_manifests(manifests, output_dir)


def pack_code_dataset(manifest_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    raw = yaml.safe_load(Path(manifest_path).read_text())
    manifests = raw if isinstance(raw, list) else [raw]
    return _pack_code_manifests(manifests, output_dir)


def _pack_code_manifests(manifests: list[dict[str, object]], output_dir: str | Path) -> tuple[Path, Path]:
    tasks = [CodeTaskManifest.model_validate(item) for item in manifests]
    output = Path(output_dir)
    task_root = output / "task-data"
    rows: list[dict[str, object]] = []
    for task in tasks:
        task_dir = task_root / task.id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "instruction.md").write_text(task.instruction.strip() + "\n")
        (task_dir / "image.txt").write_text(task.image.strip() + "\n")
        for name, content in task.seed_files.items():
            _safe_write(task_dir / "environment" / "seeds", name, content)
        for name, content in task.test_files.items():
            _safe_write(task_dir / "tests", name, content)
        for copy_index in range(task.copies):
            rows.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            # The backend copies environment seeds into the sandbox,
                            # but it does not expose instruction.md to the policy.
                            # The actual problem must therefore be in the model prompt.
                            "content": task.instruction.strip(),
                        }
                    ],
                    # Environment rewards are aggregated separately; this registered
                    # verifier intentionally contributes zero ground-truth reward.
                    "dataset": "passthrough",
                    "oellm_source_dataset": "oellm-code-rlvr",
                    "ground_truth": task.ground_truth,
                    "instance_id": f"{task.id}-{copy_index}",
                    "tools": ["bash"],
                    "env_config": {
                        "max_steps": task.max_steps,
                        "env_name": "swerl_vanillux_sandbox",
                        "task_id": task.id,
                        "image": task.image,
                    },
                }
            )
    dataset_path = output / "train.parquet"
    write_rows(rows, dataset_path)
    archive_path = output / "task-data.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for child in sorted(task_root.iterdir()):
            archive.add(child, arcname=child.name)
    return dataset_path, task_root
