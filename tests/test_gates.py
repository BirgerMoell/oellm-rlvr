from oellm_rlvr.config import GateConfig
from oellm_rlvr.gates import evaluate_gates
from oellm_rlvr.schemas import TrajectoryRecord, VerifierResult


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
