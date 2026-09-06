from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .harbor_atif import index_harbor_atif


def qualify_harbor_rollouts(
    root: str | Path,
    index_output: str | Path,
    report_output: str | Path,
    *,
    run_id: str,
    policy_version: int,
    learner_version: int,
    expected_trials: int,
    min_bash_commands_per_trial: int = 1,
    min_task_complete_per_trial: int = 0,
) -> dict[str, Any]:
    """Apply the executable-agent gate to raw Harbor trials.

    Reward zero is a valid rollout outcome here. The gate proves that the policy
    actually interacted with the terminal and that the resulting trajectory is
    suitable for a learner; reward variance is checked separately before a
    gradient-bearing training update.
    """

    index_path = Path(index_output)
    report_path = Path(report_output)
    admission = index_harbor_atif(
        root,
        index_path,
        run_id=run_id,
        policy_version=policy_version,
        learner_version=learner_version,
        require_token_ids=True,
        require_logprobs=True,
    )
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]

    trial_checks = []
    rewards = []
    for row in rows:
        metadata = row.get("metadata") or {}
        bash_commands = int(metadata.get("bash_command_count") or 0)
        completion_signals = int(metadata.get("task_complete_count") or 0)
        reward = row.get("reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            rewards.append(float(reward))
        checks = {
            "learner_admissible": row.get("accepted_for_rl") is True,
            "terminus_2_agent": row.get("agent_name") == "terminus-2",
            "singularity_environment": metadata.get("environment_type") == "singularity",
            "real_terminal_action": bash_commands >= min_bash_commands_per_trial,
            "completion_signal": completion_signals >= min_task_complete_per_trial,
            "linked_terminal_observation": int(metadata.get("linked_observation_count") or 0) >= 1,
        }
        trial_checks.append(
            {
                "trial_id": row.get("trial_id"),
                "task_name": row.get("task_name"),
                "reward": reward,
                "bash_command_count": bash_commands,
                "task_complete_count": completion_signals,
                "parser_error_count": int(metadata.get("parser_error_count") or 0),
                "checks": checks,
                "ok": all(checks.values()),
            }
        )

    gates = {
        "expected_trial_count": len(rows) == expected_trials,
        "learner_admission": admission["ok"],
        "all_trials_are_executable_agent_rollouts": bool(trial_checks)
        and all(trial["ok"] for trial in trial_checks),
        "verifier_reward_for_every_trial": len(rewards) == len(rows),
    }
    report = {
        "ok": all(gates.values()),
        "run_id": run_id,
        "root": str(Path(root).resolve()),
        "index": str(index_path.resolve()),
        "requirements": {
            "expected_trials": expected_trials,
            "min_bash_commands_per_trial": min_bash_commands_per_trial,
            "min_task_complete_per_trial": min_task_complete_per_trial,
            "token_ids": True,
            "logprobs": True,
        },
        "observed": {
            "trials": len(rows),
            "accepted_for_rl": admission["observed"]["accepted"],
            "rewards": rewards,
            "reward_mean": sum(rewards) / len(rewards) if rewards else None,
            "unique_rewards": sorted(set(rewards)),
            "bash_commands": sum(trial["bash_command_count"] for trial in trial_checks),
            "parser_errors": sum(trial["parser_error_count"] for trial in trial_checks),
        },
        "gates": gates,
        "admission": admission,
        "trials": trial_checks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
