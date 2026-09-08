from __future__ import annotations

import pytest

from oellm_rlvr.harbor_sampling import expand_prompt_repetitions


def _trajectory_id(**values):
    return values


def test_expand_prompt_repetitions_builds_within_prompt_groups() -> None:
    prompts, trajectory_ids = expand_prompt_repetitions(
        [
            {"uid": "a", "prompt": "first"},
            {"uid": "b", "prompt": "second"},
        ],
        3,
        _trajectory_id,
    )

    assert prompts == ["first", "first", "first", "second", "second", "second"]
    assert trajectory_ids == [
        {"instance_id": "a", "repetition_id": 0},
        {"instance_id": "a", "repetition_id": 1},
        {"instance_id": "a", "repetition_id": 2},
        {"instance_id": "b", "repetition_id": 0},
        {"instance_id": "b", "repetition_id": 1},
        {"instance_id": "b", "repetition_id": 2},
    ]


def test_expand_prompt_repetitions_rejects_zero() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        expand_prompt_repetitions([], 0, _trajectory_id)
