from __future__ import annotations

import json
import tarfile
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CodeTaskManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    image: str
    seed_files: dict[str, str] = Field(default_factory=dict)
    test_files: dict[str, str]
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
            "dataset": "oellm_math_smoke",
        }
        for n in range(1, count + 1)
    ]
    write_rows(rows, path)


def pack_code_dataset(manifest_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    raw = yaml.safe_load(Path(manifest_path).read_text())
    manifests = raw if isinstance(raw, list) else [raw]
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
                            "content": "Solve the coding task in the sandbox. Run tests, then submit when complete.",
                        }
                    ],
                    "dataset": "oellm_code",
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
