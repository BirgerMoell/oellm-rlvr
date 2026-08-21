import sys

import pytest

from oellm_rlvr.schemas import TaskSpec
from oellm_rlvr.verifiers import CodeVerifier, LocalRunner, MathVerifier, equivalent_math_answers


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [("0.75", "3/4"), ("\\frac{3}{4}", "3/4"), ("2**3", "8"), ("-2", "-2")],
)
def test_math_equivalence(candidate: str, expected: str) -> None:
    assert equivalent_math_answers(candidate, expected)


def test_math_verifier_extracts_last_boxed_answer() -> None:
    task = TaskSpec(id="m", kind="math", prompt="", ground_truth="3/4")
    result = MathVerifier().verify(task, "First \\boxed{1/2}, corrected to \\boxed{3/4}.")
    assert result.passed
    assert result.extracted_answer == "3/4"


def test_local_runner_requires_explicit_unsafe_opt_in() -> None:
    with pytest.raises(ValueError, match="untrusted code"):
        LocalRunner()


def test_code_verifier_runs_trusted_fixture() -> None:
    task = TaskSpec(
        id="c",
        kind="code",
        prompt="implement add",
        candidate_path="solution.py",
        test_command=[sys.executable, "test_solution.py"],
        files={
            "test_solution.py": "from solution import add\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\n"
        },
    )
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    result = CodeVerifier(LocalRunner(allow_unsafe=True), timeout=5).verify(task, completion)
    assert result.passed
    assert result.reward == 1
