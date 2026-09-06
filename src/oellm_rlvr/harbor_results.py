from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _trial_results(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    results: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("result.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and {"task_name", "trial_name"} <= payload.keys():
            results.append((path, payload))
    return results


def validate_trials(
    root: str | Path,
    *,
    expected_agent: str,
    expected_reward: float,
    expected_tasks: int,
    expected_attempts: int,
) -> dict[str, Any]:
    base = Path(root)
    trials = _trial_results(base)
    expected_total = expected_tasks * expected_attempts
    task_counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    reward_rows: list[dict[str, Any]] = []
    for path, payload in trials:
        task = str(payload.get("task_name"))
        task_counts[task] += 1
        agent = str((payload.get("agent_info") or {}).get("name"))
        exception = payload.get("exception_info")
        rewards = (payload.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        row = {
            "task": task,
            "trial": str(payload.get("trial_name")),
            "agent": agent,
            "reward": reward,
            "result": str(path.relative_to(base)),
        }
        reward_rows.append(row)
        if exception is not None:
            failures.append({"trial": row["trial"], "reason": f"exception: {exception}"})
        elif agent != expected_agent:
            failures.append({"trial": row["trial"], "reason": f"agent={agent!r}"})
        elif reward is None or float(reward) != expected_reward:
            failures.append({"trial": row["trial"], "reason": f"reward={reward!r}"})

    count_ok = len(trials) == expected_total
    multiplicity_ok = (
        len(task_counts) == expected_tasks
        and all(count == expected_attempts for count in task_counts.values())
    )
    return {
        "ok": count_ok and multiplicity_ok and not failures,
        "root": str(base.resolve()),
        "expected": {
            "agent": expected_agent,
            "reward": expected_reward,
            "tasks": expected_tasks,
            "attempts_per_task": expected_attempts,
            "trials": expected_total,
        },
        "observed": {
            "trials": len(trials),
            "tasks": len(task_counts),
            "task_multiplicity": dict(sorted(task_counts.items())),
            "failures": failures,
        },
        "trials": reward_rows,
    }
