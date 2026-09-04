import json
from pathlib import Path

import pytest

from oellm_rlvr.datasets import (
    _stdio_test_script,
    make_math_calibration,
    make_math_smoke,
    pack_code_dataset,
    sample_code_dataset,
    sample_math_dataset,
)

ROOT = Path(__file__).parents[1]


def test_math_smoke_jsonl_schema(tmp_path: Path) -> None:
    path = tmp_path / "math.jsonl"
    make_math_smoke(path, count=3)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["ground_truth"] == "3"
    assert rows[0]["dataset"] == "math"
    assert rows[0]["messages"][0]["role"] == "user"


def test_math_calibration_spans_eight_groups_and_can_repeat(tmp_path: Path) -> None:
    path = tmp_path / "calibration.jsonl"
    make_math_calibration(path, copies=2)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(rows) == 16
    assert len({row["semantic_group_id"] for row in rows}) == 8
    assert {row["oellm_canary_copy"] for row in rows} == {0, 1}
    assert {row["ground_truth"] for row in rows} >= {"15", "579", "888"}
    assert all(row["dataset"] == "math" for row in rows)


def test_math_calibration_rejects_nonpositive_copies(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="copies must be positive"):
        make_math_calibration(tmp_path / "calibration.jsonl", copies=0)


def test_published_math_sample_preserves_verifier_contract(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "m1",
                    "dataset": "oellm-math-rlvr",
                    "messages": [{"role": "user", "content": "Compute 2 + 3."}],
                    "ground_truth": ["5"],
                    "verifier_kind": "integer_exact",
                    "semantic_group_id": "g1",
                    "language": "en",
                }
            ]
        ),
        source,
    )
    output = tmp_path / "sample.parquet"
    sample_math_dataset(source, output, count=1, language="en")
    row = pq.read_table(output).to_pylist()[0]
    assert row["ground_truth"] == "5"
    assert row["verifier_kind"] == "integer_exact"
    assert row["dataset"] == "math"
    assert row["oellm_source_dataset"] == "oellm-math-rlvr"


def test_published_math_sample_rejects_multiple_answers_for_single_verifier(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "messages": [{"role": "user", "content": "Compute 2 + 3."}],
                    "ground_truth": ["5", "five"],
                    "verifier_kind": "integer_exact",
                }
            ]
        ),
        source,
    )
    with pytest.raises(ValueError, match="exactly one ground truth"):
        sample_math_dataset(source, tmp_path / "sample.parquet", count=1)


def test_published_math_sample_can_filter_difficulty_and_subdomain(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "source.parquet"
    rows = [
        {
            "id": f"m{index}",
            "dataset": "oellm-math-rlvr",
            "messages": [{"role": "user", "content": f"Problem {index}"}],
            "ground_truth": [str(index)],
            "verifier_kind": "integer_exact",
            "semantic_group_id": f"g{index}",
            "language": "en",
            "difficulty": difficulty,
            "subdomain": subdomain,
        }
        for index, (difficulty, subdomain) in enumerate(
            [(3, "algebra"), (4, "number_theory"), (5, "algebra"), (5, "geometry")]
        )
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    output = tmp_path / "sample.parquet"
    sample_math_dataset(
        source,
        output,
        count=2,
        language="en",
        min_difficulty=4,
        max_difficulty=5,
        diverse_by="subdomain",
    )
    selected = pq.read_table(output).to_pylist()
    assert [row["subdomain"] for row in selected] == ["number_theory", "algebra"]
    assert all(4 <= row["difficulty"] <= 5 for row in selected)


def test_published_math_sample_can_select_one_subdomain(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "source.parquet"
    rows = [
        {
            "messages": [{"role": "user", "content": f"Problem {index}"}],
            "ground_truth": [str(index)],
            "verifier_kind": "rational_exact",
            "semantic_group_id": f"g{index}",
            "language": "en",
            "difficulty": 2,
            "subdomain": subdomain,
        }
        for index, subdomain in enumerate(("linear_equations", "fraction_operations", "fraction_operations"))
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    output = tmp_path / "sample.parquet"
    sample_math_dataset(source, output, count=2, language="en", subdomain="fraction_operations")
    selected = pq.read_table(output).to_pylist()
    assert len(selected) == 2
    assert {row["subdomain"] for row in selected} == {"fraction_operations"}


def test_published_math_sample_seed_is_reproducible_and_not_first_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "source.parquet"
    rows = [
        {
            "id": f"m{index}",
            "dataset": "oellm-math-rlvr",
            "messages": [{"role": "user", "content": f"Problem {index}"}],
            "ground_truth": [str(index)],
            "verifier_kind": "integer_exact",
            "semantic_group_id": f"g{index}",
            "language": "en",
            "difficulty": 2,
            "subdomain": "arithmetic",
        }
        for index in range(40)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)

    outputs = [tmp_path / f"sample-{index}.parquet" for index in range(3)]
    sample_math_dataset(source, outputs[0], count=8, language="en", seed=20260904)
    sample_math_dataset(source, outputs[1], count=8, language="en", seed=20260904)
    sample_math_dataset(source, outputs[2], count=8, language="en", seed=7)

    ids = [[row["id"] for row in pq.read_table(path).to_pylist()] for path in outputs]
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]
    assert ids[0] != [f"m{index}" for index in range(8)]


def test_seeded_diverse_sample_uses_distinct_values_reproducibly(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "source.parquet"
    rows = [
        {
            "id": f"m{index}",
            "dataset": "oellm-math-rlvr",
            "messages": [{"role": "user", "content": f"Problem {index}"}],
            "ground_truth": [str(index)],
            "verifier_kind": "integer_exact",
            "semantic_group_id": f"g{index}",
            "language": "en",
            "difficulty": 2,
            "subdomain": f"domain-{index % 6}",
        }
        for index in range(30)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"

    for output in (first, second):
        sample_math_dataset(source, output, count=4, diverse_by="subdomain", seed=11)

    selected_first = pq.read_table(first).to_pylist()
    selected_second = pq.read_table(second).to_pylist()
    assert [row["id"] for row in selected_first] == [row["id"] for row in selected_second]
    assert len({row["subdomain"] for row in selected_first}) == 4


def test_published_math_sample_can_repeat_a_calibration_row(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "messages": [{"role": "user", "content": "Compute 7/20 - 8/2."}],
                    "ground_truth": ["-73/20"],
                    "verifier_kind": "rational_exact",
                    "semantic_group_id": "fraction-canary",
                    "language": "en",
                    "difficulty": 2,
                    "subdomain": "fraction_operations",
                }
            ]
        ),
        source,
    )
    output = tmp_path / "sample.parquet"

    sample_math_dataset(source, output, count=1, copies=8)

    selected = pq.read_table(output).to_pylist()
    assert len(selected) == 8
    assert [row["oellm_canary_copy"] for row in selected] == list(range(8))
    assert {row["ground_truth"] for row in selected} == {"-73/20"}


def test_stdio_test_script_contains_valid_python() -> None:
    script = _stdio_test_script()
    python_source = script.split("python - <<'PY'\n", 1)[1].rsplit("\nPY\n", 1)[0]
    compile(python_source, "generated-test.sh", "exec")
    assert 'write_text("1.0\\n" if passed else "0.0\\n")' in python_source


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
    assert row["dataset"] == "passthrough"
    assert "Implement `add(a, b)`" in row["messages"][0]["content"]
    assert "/workspace/solution.py" in row["messages"][0]["content"]
    assert (task_root / "add-two-integers/tests/test.sh").is_file()
    assert (tmp_path / "task-data.tar.gz").is_file()


def test_published_code_sample_builds_hidden_stdio_task(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "code.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "c1",
                    "messages": [{"role": "user", "content": "Double the input."}],
                    "problem": "Double the input.",
                    "verification_info": {
                        "language": "python",
                        "test_cases": [{"input": "2\n", "output": "4\n", "type": "stdin_stdout"}],
                    },
                    "time_limit_seconds": 2,
                    "reference_solution": "value = int(input())\nprint(value * 2)\n",
                    "semantic_group_id": "c1",
                }
            ]
        ),
        source,
    )
    dataset, task_root = sample_code_dataset(source, tmp_path / "packed", "/images/python.sif", count=1)
    row = pq.read_table(dataset).to_pylist()[0]
    assert row["env_config"]["task_id"] == "c1"
    assert row["env_config"]["max_steps"] == 6
    assert row["ground_truth"] == "value = int(input())\nprint(value * 2)\n"
    assert "Double the input" in (task_root / "c1/instruction.md").read_text()
    cases = json.loads((task_root / "c1/tests/cases.json").read_text())
    assert cases == [{"input": "2\n", "output": "4\n", "timeout_seconds": 2}]


def test_published_code_sample_can_select_easy_diverse_tasks(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "code.parquet"
    rows = []
    for index, (difficulty, family) in enumerate([(1, "grid"), (1, "strings"), (2, "graph"), (3, "dp")]):
        rows.append(
            {
                "id": f"c{index}",
                "messages": [{"role": "user", "content": f"Problem {index}"}],
                "problem": f"Problem {index}",
                "verification_info": {
                    "language": "python",
                    "test_cases": [{"input": "1\n", "output": "1\n", "type": "stdin_stdout"}],
                },
                "time_limit_seconds": 2,
                "reference_solution": "print(input())\n",
                "semantic_group_id": f"g{index}",
                "prompt_language": "en",
                "difficulty": difficulty,
                "generator_family": family,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), source)
    dataset, _ = sample_code_dataset(
        source,
        tmp_path / "packed",
        "/images/python.sif",
        count=2,
        language="en",
        max_difficulty=1,
        diverse_by="generator_family",
    )
    selected = pq.read_table(dataset).to_pylist()
    assert [row["env_config"]["task_id"] for row in selected] == ["c0", "c1"]
