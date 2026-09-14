from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from oellm_rlvr.datasets import write_rows
from oellm_rlvr.multilingual_reasoning import (
    CHALLENGE_FAMILIES,
    build_multilingual_reasoning_canary,
    parse_language_contract,
)


def test_builder_cli_accepts_explicit_family_list() -> None:
    script = (Path(__file__).parents[1] / "scripts/build_multilingual_reasoning_canary.py").read_text()
    assert '"--families"' in script
    assert "len(set(families))" in script


def test_parse_language_contract_preserves_variants(tmp_path: Path) -> None:
    contract = tmp_path / "languages"
    contract.write_text("eng: English: eng_Latn\nest: Estonian: est_Latn ekk_Latn\n")
    targets = parse_language_contract(contract)
    assert [target.alpha2 for target in targets] == ["en", "et"]
    assert targets[1].variants == ("est_Latn", "ekk_Latn")


def test_parse_language_contract_rejects_unmapped_language(tmp_path: Path) -> None:
    contract = tmp_path / "languages"
    contract.write_text("zzz: Unknown: zzz_Latn\n")
    with pytest.raises(ValueError, match="no ISO-639-1"):
        parse_language_contract(contract)


def test_build_multilingual_reasoning_canary_is_balanced_and_disjoint(tmp_path: Path) -> None:
    contract = tmp_path / "languages"
    contract.write_text("eng: English: eng_Latn\nkat: Georgian: kat_Geor\n")
    reference = tmp_path / "reference.parquet"
    write_rows(
        [
            {
                "language": "de",
                "messages": [
                    {"role": "user", "content": "Löse die folgende Aufgabe.\n\n1 + 1"},
                    {"role": "assistant", "content": "<think>1+1=2</think>\\boxed{2}"},
                ],
            }
        ],
        reference,
    )
    output = tmp_path / "output"
    manifest = build_multilingual_reasoning_canary(
        language_contract=contract,
        reference_traces=reference,
        output_dir=output,
        contract_revision="abc123",
        reference_revision="def456",
        train_per_language=8,
        eval_per_language=2,
        seed=7,
    )

    train = pq.read_table(output / "train.parquet").to_pylist()
    profile = pq.read_table(output / "profile.parquet").to_pylist()
    evaluation = pq.read_table(output / "evaluation.parquet").to_pylist()
    assert len(train) == 16
    assert len(profile) == 2
    assert len(evaluation) == 4
    assert {row["language"] for row in train} == {"en", "ka"}
    assert {row["canonical_variant"] for row in train} == {"eng_Latn", "kat_Geor"}
    assert {row["id"] for row in train}.isdisjoint({row["id"] for row in evaluation})
    assert all("<think>" in row["messages"][0]["content"] for row in train)
    assert all(row["dataset"] == "math" for row in train)
    assert manifest["language_contract"]["macro_languages"] == 2
    assert manifest["language_contract"]["internal_variants"] == 2
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_build_challenge_canary_uses_harder_verified_families(tmp_path: Path) -> None:
    contract = tmp_path / "languages"
    contract.write_text("eng: English: eng_Latn\nkat: Georgian: kat_Geor\n")
    reference = tmp_path / "reference.parquet"
    write_rows(
        [
            {
                "language": "de",
                "messages": [
                    {"role": "user", "content": "Löse die folgende Aufgabe.\n\n1 + 1"},
                    {"role": "assistant", "content": "<think>1+1=2</think>\\boxed{2}"},
                ],
            }
        ],
        reference,
    )
    output = tmp_path / "challenge"

    manifest = build_multilingual_reasoning_canary(
        language_contract=contract,
        reference_traces=reference,
        output_dir=output,
        contract_revision="abc123",
        reference_revision="def456",
        train_per_language=8,
        eval_per_language=4,
        profile_per_language=2,
        families=CHALLENGE_FAMILIES,
        seed=11,
    )

    train = pq.read_table(output / "train.parquet").to_pylist()
    profile = pq.read_table(output / "profile.parquet").to_pylist()
    assert len(profile) == 4
    assert {row["generator_family"] for row in train} == set(CHALLENGE_FAMILIES)
    assert all(row["difficulty"] >= 4 for row in train)
    assert all(row["ground_truth"].lstrip("-").isdigit() for row in train)
    assert manifest["profile_per_language"] == 2


def test_language_gated_training_uses_conjunctive_labels_and_keeps_full_eval(tmp_path: Path) -> None:
    contract = tmp_path / "languages"
    contract.write_text("eng: English: eng_Latn\nmlt: Maltese: mlt_Latn\n")
    reference = tmp_path / "reference.parquet"
    write_rows(
        [
            {
                "language": "mt",
                "messages": [
                    {"role": "user", "content": "Issolvi din il-problema.\n\n1 + 1"},
                    {"role": "assistant", "content": "<think>1+1=2</think>\\boxed{2}"},
                ],
            }
        ],
        reference,
    )
    output = tmp_path / "gated"

    manifest = build_multilingual_reasoning_canary(
        language_contract=contract,
        reference_traces=reference,
        output_dir=output,
        contract_revision="abc123",
        reference_revision="def456",
        train_per_language=2,
        eval_per_language=1,
        language_gated_training=True,
        seed=13,
    )

    train = pq.read_table(output / "train.parquet").to_pylist()
    evaluation = pq.read_table(output / "evaluation.parquet").to_pylist()
    assert {row["language"] for row in train} == {"en"}
    assert {row["language"] for row in evaluation} == {"en", "mt"}
    assert all(row["dataset"] == "multilingual_math" for row in train)
    assert json.loads(train[0]["ground_truth"])["target_language"] == "en"
    assert manifest["training_languages_excluded_from_automatic_gate"] == ["mlt"]
