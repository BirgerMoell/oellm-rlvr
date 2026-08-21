import json
from pathlib import Path

import pytest

from oellm_rlvr.datasets import make_math_smoke, pack_code_dataset

ROOT = Path(__file__).parents[1]


def test_math_smoke_jsonl_schema(tmp_path: Path) -> None:
    path = tmp_path / "math.jsonl"
    make_math_smoke(path, count=3)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["ground_truth"] == "3"
    assert rows[0]["messages"][0]["role"] == "user"


def test_code_packer_matches_tmax_environment_shape(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset, task_root = pack_code_dataset(ROOT / "examples/code_task.yaml", tmp_path)
    import pyarrow.parquet as pq

    rows = pq.read_table(dataset).to_pylist()
    assert len(rows) == 4
    row = rows[0]
    env = row["env_config"]
    assert env["env_name"] == "swerl_vanillux_sandbox"
    assert env["task_id"] == "add-two-integers"
    assert row["tools"] == ["bash"]
    assert (task_root / "add-two-integers/tests/test.sh").is_file()
    assert (tmp_path / "task-data.tar.gz").is_file()
