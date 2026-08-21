from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import fmean, pstdev

from .config import GateConfig
from .schemas import TrajectoryRecord


@dataclass(frozen=True)
class GateReport:
    passed: bool
    samples: int
    mean_reward: float
    zero_std_fraction: float
    truncation_fraction: float
    error_fraction: float
    max_policy_lag: int
    mean_entropy: float | None
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_gates(records: Iterable[TrajectoryRecord], limits: GateConfig) -> GateReport:
    items = list(records)
    if not items:
        return GateReport(False, 0, 0, 1, 0, 0, 0, None, ("no trajectories",))
    rewards = [item.verifier.reward for item in items]
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for item in items:
        groups[(item.task_id, item.policy_version)].append(item.verifier.reward)
    zero_std = sum(pstdev(values) == 0 for values in groups.values()) / len(groups)
    truncation = sum(item.truncated for item in items) / len(items)
    errors = sum(item.verifier.error_type is not None for item in items) / len(items)
    max_lag = max(item.policy_lag for item in items)
    entropy_values = [item.entropy for item in items if item.entropy is not None]
    failures: list[str] = []
    if zero_std > limits.max_zero_std_fraction:
        failures.append(f"zero-std groups {zero_std:.3f} > {limits.max_zero_std_fraction:.3f}")
    if truncation > limits.max_truncation_fraction:
        failures.append(f"truncation {truncation:.3f} > {limits.max_truncation_fraction:.3f}")
    if errors > limits.max_error_fraction:
        failures.append(f"verifier errors {errors:.3f} > {limits.max_error_fraction:.3f}")
    if max_lag > limits.max_policy_lag:
        failures.append(f"policy lag {max_lag} > {limits.max_policy_lag}")
    mean_reward = fmean(rewards)
    if limits.min_mean_reward is not None and mean_reward < limits.min_mean_reward:
        failures.append(f"mean reward {mean_reward:.3f} < {limits.min_mean_reward:.3f}")
    return GateReport(
        passed=not failures,
        samples=len(items),
        mean_reward=mean_reward,
        zero_std_fraction=zero_std,
        truncation_fraction=truncation,
        error_fraction=errors,
        max_policy_lag=max_lag,
        mean_entropy=fmean(entropy_values) if entropy_values else None,
        failures=tuple(failures),
    )
