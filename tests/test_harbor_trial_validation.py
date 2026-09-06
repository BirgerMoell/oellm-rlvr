from __future__ import annotations

import json
from pathlib import Path

from oellm_rlvr.harbor_results import validate_trials


def _write_result(root: Path, task: str, attempt: int, agent: str, reward: float) -> None:
    trial = root / f"{task}__{attempt}"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "trial_name": trial.name,
                "agent_info": {"name": agent},
                "verifier_result": {"rewards": {"reward": reward}},
                "exception_info": None,
            }
        )
    )


def test_validate_harbor_trials_checks_reward_and_multiplicity(tmp_path: Path) -> None:
    for task in ("one", "two"):
        for attempt in range(2):
            _write_result(tmp_path, task, attempt, "oracle", 1)
    report = validate_trials(
        tmp_path,
        expected_agent="oracle",
        expected_reward=1,
        expected_tasks=2,
        expected_attempts=2,
    )
    assert report["ok"] is True

    bad_path = tmp_path / "one__0" / "result.json"
    bad = json.loads(bad_path.read_text())
    bad["verifier_result"]["rewards"]["reward"] = 0
    bad_path.write_text(json.dumps(bad))
    assert (
        validate_trials(
            tmp_path,
            expected_agent="oracle",
            expected_reward=1,
            expected_tasks=2,
            expected_attempts=2,
        )["ok"]
        is False
    )
