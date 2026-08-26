from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


@dataclass(frozen=True)
class BackendRolloutReport:
    samples: int
    prompt_groups: int
    mixed_prompt_groups: int
    zero_std_fraction: float
    mean_reward: float
    min_advantage: float
    max_advantage: float
    nonzero_advantage_fraction: float
    has_grouped_reward_signal: bool
    submitted_count: int | None
    non_submitting_count: int | None
    truncation_count: int
    timeout_count: int
    tool_format_error_trajectories: int
    rewards_by_group: dict[str, list[float]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"rollout line {line_number} is not a JSON object")
            yield value


def inspect_backend_rollouts(path: str | Path) -> BackendRolloutReport:
    items = list(_records(path))
    if not items:
        raise ValueError("no rollout records")

    groups: dict[tuple[int, int], list[float]] = defaultdict(list)
    rewards: list[float] = []
    advantages: list[float] = []
    done_values: list[bool] = []
    truncations = 0
    timeouts = 0
    format_errors = 0

    for line_number, item in enumerate(items, start=1):
        try:
            step = int(item.get("step", 0))
            prompt_idx = int(item["prompt_idx"])
            reward = float(item["reward"])
            advantage = float(item.get("advantage", 0.0))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid rollout fields on line {line_number}: {error}") from error
        groups[(step, prompt_idx)].append(reward)
        rewards.append(reward)
        advantages.append(advantage)
        truncations += item.get("finish_reason") == "length"

        request_info = item.get("request_info") or {}
        rollout_state = request_info.get("rollout_state") or {}
        if "done" in rollout_state:
            done_values.append(bool(rollout_state["done"]))
        timeouts += bool(rollout_state.get("timeout", False))
        format_errors += bool(request_info.get("tool_errors", ""))

    mixed = sum(pstdev(values) > 0 for values in groups.values())
    nonzero_advantages = sum(abs(value) > 1e-12 for value in advantages)
    submitted = sum(done_values) if done_values else None
    rewards_by_group = {f"step={step},prompt={prompt}": values for (step, prompt), values in sorted(groups.items())}
    return BackendRolloutReport(
        samples=len(items),
        prompt_groups=len(groups),
        mixed_prompt_groups=mixed,
        zero_std_fraction=(len(groups) - mixed) / len(groups),
        mean_reward=fmean(rewards),
        min_advantage=min(advantages),
        max_advantage=max(advantages),
        nonzero_advantage_fraction=nonzero_advantages / len(items),
        has_grouped_reward_signal=mixed > 0 and nonzero_advantages > 0,
        submitted_count=submitted,
        non_submitting_count=(len(done_values) - submitted) if submitted is not None else None,
        truncation_count=truncations,
        timeout_count=timeouts,
        tool_format_error_trajectories=format_errors,
        rewards_by_group=rewards_by_group,
    )
