import pytest

from oellm_rlvr.language_audit import audit_reasoning_languages, reasoning_prose


def test_reasoning_prose_excludes_final_answer_and_tags() -> None:
    assert reasoning_prose("<think>Det här är svenska.</think>\\boxed{7}") == "Det här är svenska."


def test_language_audit_validates_confidence_before_loading_detector(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("{}\n")
    with pytest.raises(ValueError, match="minimum_confidence"):
        audit_reasoning_languages(predictions, tmp_path / "report.json", minimum_confidence=1.1)
