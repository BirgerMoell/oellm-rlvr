import json

import pytest

from oellm_rlvr.backend_rollouts import inspect_backend_rollouts


def _row(prompt: int, reward: float, *, done: bool = False, finish: str = "stop", error: str = "") -> dict:
    return {
        "step": 0,
        "prompt_idx": prompt,
        "reward": reward,
        "advantage": reward - 0.5,
        "finish_reason": finish,
        "request_info": {
            "tool_errors": error,
            "rollout_state": {"done": done, "timeout": False},
        },
    }


def _write(tmp_path, rows: list[dict]):
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_inspect_rollouts_finds_grouped_signal(tmp_path) -> None:
    path = _write(
        tmp_path,
        [
            _row(0, 0),
            _row(0, 1, done=True),
            _row(1, 0, finish="length", error="Format error"),
            _row(1, 0),
        ],
    )

    report = inspect_backend_rollouts(path)

    assert report.samples == 4
    assert report.mixed_prompt_groups == 1
    assert report.zero_std_fraction == 0.5
    assert report.has_grouped_reward_signal
    assert report.submitted_count == 1
    assert report.non_submitting_count == 3
    assert report.truncation_count == 1
    assert report.tool_format_error_trajectories == 1


def test_inspect_rollouts_rejects_aggregate_accuracy_without_group_variance(tmp_path) -> None:
    path = _write(tmp_path, [_row(0, 1), _row(0, 1), _row(1, 0), _row(1, 0)])

    report = inspect_backend_rollouts(path)

    assert report.mean_reward == 0.5
    assert report.zero_std_fraction == 1
    assert not report.has_grouped_reward_signal


def test_inspect_rollouts_rejects_empty_file(tmp_path) -> None:
    with pytest.raises(ValueError, match="no rollout records"):
        inspect_backend_rollouts(_write(tmp_path, []))
