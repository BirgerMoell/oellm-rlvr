import json
from pathlib import Path

import pytest

from oellm_rlvr.reasoning_eval import (
    analyze_reasoning_completion,
    build_blinded_reasoning_audit,
    compare_reasoning_evals,
    summarize_reasoning_predictions,
)


def _record(task_id: str, sample: int, text: str, expected: str) -> dict:
    return {
        "id": task_id,
        "sample_index": sample,
        "prompt": [{"role": "user", "content": f"Problem {task_id}"}],
        "text": text,
        "analysis": analyze_reasoning_completion(text, expected, response_tokens=20),
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_reasoning_analysis_distinguishes_correctness_form_and_structure() -> None:
    good = analyze_reasoning_completion(
        "We add the two quantities. There are 12 + 7 = 19 objects, so \\boxed{19}", "19"
    )
    short = analyze_reasoning_completion("\\boxed{19}", "19")
    malformed = analyze_reasoning_completion("<think>First calculate 12 + 7. Answer: 18", "19", finish_reason="length")
    channeled = analyze_reasoning_completion("<think>We calculate 12 + 7 = 19.</think>\n\\boxed{19}", "19")

    assert good["correct"] and good["format_pass"] and good["reasoning_structure_present"]
    assert short["correct"] and short["format_pass"] and not short["reasoning_structure_present"]
    assert not malformed["correct"]
    assert not good["uses_think_tags"]
    assert malformed["uses_think_tags"]
    assert not malformed["think_tags_balanced"]
    assert malformed["length_stopped"]
    assert channeled["reasoning_channel_format_pass"]
    assert not good["reasoning_channel_format_pass"]


def test_reasoning_analysis_accepts_gsm8k_thousands_separators() -> None:
    result = analyze_reasoning_completion("The product is \\boxed{12,345}.", "12345")
    assert result["correct"]
    assert result["format_pass"]


def test_reasoning_summary_reports_sample_and_prompt_metrics() -> None:
    rows = [
        _record("a", 0, "Reasoning long enough here. \\boxed{2}", "2"),
        _record("a", 1, "Reasoning long enough here. \\boxed{3}", "2"),
        _record("b", 0, "Reasoning long enough here. \\boxed{4}", "5"),
        _record("b", 1, "Reasoning long enough here. \\boxed{5}", "5"),
    ]
    summary = summarize_reasoning_predictions(rows)
    assert summary["sample_accuracy"] == 0.5
    assert summary["pass_at_k"] == 1.0
    assert summary["samples_per_prompt"] == [2]
    assert summary["think_tag_use_rate"] == 0.0


def test_paired_comparison_and_blinded_audit(tmp_path: Path) -> None:
    before = _write(
        tmp_path / "before.jsonl",
        [
            _record("a", 0, "Work through a carefully. \\boxed{0}", "1"),
            _record("b", 0, "Work through b carefully. \\boxed{2}", "2"),
            _record("c", 0, "Work through c carefully. \\boxed{3}", "3"),
            _record("d", 0, "Work through d carefully. \\boxed{0}", "4"),
        ],
    )
    after = _write(
        tmp_path / "after.jsonl",
        [
            _record("a", 0, "Work through a carefully. \\boxed{1}", "1"),
            _record("b", 0, "Work through b carefully. \\boxed{0}", "2"),
            _record("c", 0, "Work through c carefully. \\boxed{3}", "3"),
            _record("d", 0, "Work through d carefully. \\boxed{0}", "4"),
        ],
    )

    report = compare_reasoning_evals(before, after, bootstrap_samples=100, seed=7)
    assert report["deltas"]["sample_accuracy"] == 0
    assert report["transitions"] == {
        "both_wrong": 1,
        "improved_wrong_to_right": 1,
        "regressed_right_to_wrong": 1,
        "both_right": 1,
    }
    outputs = build_blinded_reasoning_audit(before, after, tmp_path / "audit.md", count=4, seed=7)
    assert Path(outputs["audit"]).is_file()
    assert Path(outputs["key"]).is_file()
    assert "Automatic correctness label" not in Path(outputs["audit"]).read_text()


def test_compare_requires_identical_keys(tmp_path: Path) -> None:
    before = _write(tmp_path / "before.jsonl", [_record("a", 0, "\\boxed{1}", "1")])
    after = _write(tmp_path / "after.jsonl", [_record("b", 0, "\\boxed{1}", "1")])
    with pytest.raises(ValueError, match="identical"):
        compare_reasoning_evals(before, after)
