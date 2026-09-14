from __future__ import annotations

from typing import Any

from oellm_rlvr.verifiers import equivalent_math_answers, extract_math_answer


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    """Neutral reward used when OPD is intentionally run without task rewards.

    verl still executes its reward plumbing in a pure distillation run. Returning
    zero makes that plumbing explicit and prevents accidental benchmark-specific
    reward lookup; the teacher loss remains the only learning signal.
    """
    del data_source, solution_str, ground_truth, extra_info
    return 0.0


def compute_math_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    """Project-owned exact/numeric math reward for hybrid OPD+RLVR."""
    del data_source, extra_info
    if solution_str is None or ground_truth is None:
        return 0.0
    candidate = extract_math_answer(solution_str)
    return float(equivalent_math_answers(candidate, str(ground_truth)))
