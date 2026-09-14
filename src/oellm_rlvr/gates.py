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
    reward_coverage: float
    distillation_coverage: float
    mean_sampled_reverse_kl: float | None
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_gates(
    records: Iterable[TrajectoryRecord],
    limits: GateConfig,
    *,
    require_reward_signal: bool = True,
    require_distillation: bool = False,
) -> GateReport:
    items = list(records)
    if not items:
        return GateReport(
            passed=False,
            samples=0,
            mean_reward=0,
            zero_std_fraction=1,
            truncation_fraction=0,
            error_fraction=0,
            max_policy_lag=0,
            mean_entropy=None,
            reward_coverage=0,
            distillation_coverage=0,
            mean_sampled_reverse_kl=None,
            failures=("no trajectories",),
        )
    rewards = [item.verifier.reward for item in items if item.verifier is not None]
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for item in items:
        if item.verifier is not None:
            groups[(item.task_id, item.policy_version)].append(item.verifier.reward)
    zero_std = sum(pstdev(values) == 0 for values in groups.values()) / len(groups) if groups else 0
    truncation = sum(item.truncated for item in items) / len(items)
    errors = sum(item.verifier is not None and item.verifier.error_type is not None for item in items) / len(items)
    max_lag = max(item.policy_lag for item in items)
    entropy_values = [item.entropy for item in items if item.entropy is not None]
    distillation_values = [
        item.distillation.sampled_reverse_kl for item in items if item.distillation is not None
    ]
    reward_coverage = len(rewards) / len(items)
    distillation_coverage = len(distillation_values) / len(items)
    failures: list[str] = []
    if require_reward_signal:
        if reward_coverage < 1:
            failures.append(f"reward coverage {reward_coverage:.3f} < 1.000")
        if zero_std > limits.max_zero_std_fraction:
            failures.append(f"zero-std groups {zero_std:.3f} > {limits.max_zero_std_fraction:.3f}")
    if truncation > limits.max_truncation_fraction:
        failures.append(f"truncation {truncation:.3f} > {limits.max_truncation_fraction:.3f}")
    if errors > limits.max_error_fraction:
        failures.append(f"verifier errors {errors:.3f} > {limits.max_error_fraction:.3f}")
    if max_lag > limits.max_policy_lag:
        failures.append(f"policy lag {max_lag} > {limits.max_policy_lag}")
    mean_reward = fmean(rewards) if rewards else 0
    if require_reward_signal and limits.min_mean_reward is not None and mean_reward < limits.min_mean_reward:
        failures.append(f"mean reward {mean_reward:.3f} < {limits.min_mean_reward:.3f}")
    mean_sampled_reverse_kl = fmean(distillation_values) if distillation_values else None
    if require_distillation and distillation_coverage < limits.min_distillation_coverage:
        failures.append(
            f"distillation coverage {distillation_coverage:.3f} < {limits.min_distillation_coverage:.3f}"
        )
    if (
        require_distillation
        and limits.max_mean_sampled_reverse_kl is not None
        and mean_sampled_reverse_kl is not None
        and mean_sampled_reverse_kl > limits.max_mean_sampled_reverse_kl
    ):
        failures.append(
            f"mean sampled reverse KL {mean_sampled_reverse_kl:.3f} > "
            f"{limits.max_mean_sampled_reverse_kl:.3f}"
        )
    return GateReport(
        passed=not failures,
        samples=len(items),
        mean_reward=mean_reward,
        zero_std_fraction=zero_std,
        truncation_fraction=truncation,
        error_fraction=errors,
        max_policy_lag=max_lag,
        mean_entropy=fmean(entropy_values) if entropy_values else None,
        reward_coverage=reward_coverage,
        distillation_coverage=distillation_coverage,
        mean_sampled_reverse_kl=mean_sampled_reverse_kl,
        failures=tuple(failures),
    )
