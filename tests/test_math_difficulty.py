import csv
import json

import pytest

from oellm_rlvr.math_difficulty import annotate_predictions


def _attempt(problem, index, correct, group):
    return {
        "id": problem,
        "sample_index": index,
        "metadata": {"language": "sv", "semantic_group_id": group, "difficulty": 3},
        "analysis": {"correct": correct, "length_stopped": False, "high_repetition": False},
    }


def test_annotation_reports_all_attempts(tmp_path):
    source = tmp_path / "predictions.jsonl"
    rows = [_attempt("a", i, i == 0, "group-a") for i in range(8)]
    rows += [_attempt("b", i, False, "group-b") for i in range(8)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "difficulty.csv"
    report = annotate_predictions(source, output, expected_samples=8)
    with output.open(newline="") as handle:
        annotations = list(csv.DictReader(handle))
    assert report["problems"] == 2
    assert report["attempts"] == 16
    assert [row["difficulty_band"] for row in annotations] == ["unsolved_at_k", "hard"]
    assert annotations[1]["sample_accuracy"] == "0.125"


def test_annotation_rejects_incomplete_and_translation_duplicate(tmp_path):
    source = tmp_path / "predictions.jsonl"
    source.write_text(json.dumps(_attempt("a", 0, True, "same")) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        annotate_predictions(source, tmp_path / "out.csv", expected_samples=8)
    rows = [_attempt("a", 0, True, "same"), _attempt("b", 0, False, "same")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="semantic group"):
        annotate_predictions(source, tmp_path / "out.csv", expected_samples=1)
