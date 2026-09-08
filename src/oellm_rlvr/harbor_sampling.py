from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

Trajectory = TypeVar("Trajectory")


def expand_prompt_repetitions(
    records: Iterable[dict[str, Any]],
    samples_per_prompt: int,
    trajectory_id_factory: Callable[..., Trajectory],
) -> tuple[list[Any], list[Trajectory]]:
    """Expand Harbor prompts into distinct within-prompt trajectory IDs.

    SkyRL v0.3.0's debugging-only ``main_harbor_generate`` entrypoint assigns
    repetition ID zero to every dataset item and ignores
    ``generator.n_samples_per_prompt``.  The training entrypoint expands these
    groups internally, but our learner-off qualification needs the same shape
    without starting an optimizer.  Keep the expansion small and explicit so
    the strict trial-count gate can prove every requested repetition exists.
    """

    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be positive")

    prompts: list[Any] = []
    trajectory_ids: list[Trajectory] = []
    for record in records:
        prompt = record["prompt"]
        instance_id = record["uid"]
        for repetition_id in range(samples_per_prompt):
            prompts.append(prompt)
            trajectory_ids.append(
                trajectory_id_factory(instance_id=instance_id, repetition_id=repetition_id)
            )
    return prompts, trajectory_ids
