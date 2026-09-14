import pytest

from oellm_rlvr.config import GateConfig
from oellm_rlvr.gates import evaluate_gates
from oellm_rlvr.schemas import DistillationTrace, TrajectoryRecord, VerifierResult


def _record(task: str, reward: float, policy: int = 2, learner: int = 3) -> TrajectoryRecord:
    return TrajectoryRecord(
        run_id="r",
        task_id=task,
        task_kind="math",
        prompt="p",
        completion="c",
        verifier=VerifierResult(
            reward=reward,
            passed=bool(reward),
            verifier="test",
            reason="fixture",
        ),
        policy_version=policy,
        learner_version=learner,
        response_tokens=10,
        max_response_tokens=20,
        entropy=0.5,
    )


def test_gate_accepts_mixed_signal() -> None:
    report = evaluate_gates([_record("a", 0), _record("a", 1), _record("b", 0), _record("b", 1)], GateConfig())
    assert report.passed
    assert report.zero_std_fraction == 0
    assert report.mean_reward == 0.5


def test_gate_rejects_zero_signal() -> None:
    report = evaluate_gates([_record("a", 1), _record("a", 1)], GateConfig(max_zero_std_fraction=0.5))
    assert not report.passed
    assert "zero-std" in report.failures[0]


def test_pure_opd_gate_uses_distillation_coverage_without_verifier_rewards() -> None:
    record = TrajectoryRecord(
        run_id="r",
        task_id="p",
        task_kind="prompt",
        prompt="p",
        completion="c",
        distillation=DistillationTrace(
            teacher_name="teacher_model",
            teacher_revision="abc",
            token_ids=[10, 11],
            rollout_logprobs=[-1.0, -2.0],
            teacher_logprobs=[-1.2, -2.4],
            trainable_mask=[True, True],
        ),
        policy_version=0,
        learner_version=0,
        response_tokens=2,
        max_response_tokens=8,
    )

    report = evaluate_gates(
        [record], GateConfig(), require_reward_signal=False, require_distillation=True
    )
    assert report.passed
    assert report.reward_coverage == 0
    assert report.distillation_coverage == 1
    assert report.mean_sampled_reverse_kl == pytest.approx(0.3)
