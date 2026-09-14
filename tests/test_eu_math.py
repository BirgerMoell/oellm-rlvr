from __future__ import annotations

import json

import pytest

from oellm_rlvr.eu_math import build_eu_math_pools


def _row(group: int, language: str, difficulty: int = 4) -> dict:
    return {
        "id": f"{group}-{language}",
        "messages": [{"role": "user", "content": f"{language}: solve {group}"}],
        "ground_truth": [str(group)],
        "dataset": "oellm-math-rlvr",
        "language": language,
        "difficulty": difficulty,
        "subdomain": "integer_operations",
        "verifier_kind": "integer_exact",
        "semantic_group_id": f"group-{group}",
    }


def test_build_eu_math_pools_balances_languages_and_disjoins_groups(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "source.parquet"
    languages = ("de", "sv")
    rows = [_row(group, language) for group in range(100) for language in (*languages, "en")]
    rows.extend(_row(group, "en") for group in range(100, 160))
    pq.write_table(pa.Table.from_pylist(rows), source)

    report = build_eu_math_pools(
        source,
        tmp_path / "pools",
        eu_per_language=10,
        english_count=20,
        languages=languages,
        seed=7,
    )
    eu = pq.read_table(tmp_path / "pools/eu-math.parquet").to_pylist()
    english = pq.read_table(tmp_path / "pools/english-replay.parquet").to_pylist()

    assert report["eu"]["by_language"] == {"de": 10, "sv": 10}
    assert report["english_replay"]["by_language"] == {"en": 20}
    assert {row["semantic_group_id"] for row in eu}.isdisjoint(
        {row["semantic_group_id"] for row in english}
    )
    assert len({row["semantic_group_id"] for row in eu}) == 20
    assert all(row["dataset"] == "math" and isinstance(row["ground_truth"], str) for row in eu + english)
    assert json.loads((tmp_path / "pools/manifest.json").read_text())["semantic_group_overlap"] == 0


def test_build_eu_math_pools_is_deterministic(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = tmp_path / "source.parquet"
    rows = [_row(group, language) for group in range(100) for language in ("de", "sv", "en")]
    pq.write_table(pa.Table.from_pylist(rows), source)
    reports = [
        build_eu_math_pools(source, tmp_path / name, eu_per_language=8, english_count=12, languages=("de", "sv"))
        for name in ("one", "two")
    ]
    assert reports[0]["eu"]["sha256"] == reports[1]["eu"]["sha256"]
    assert reports[0]["english_replay"]["sha256"] == reports[1]["english_replay"]["sha256"]
