from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskSpec(StrictModel):
    id: str
    kind: Literal["math", "code"]
    prompt: str
    ground_truth: str | list[str] | None = None
    candidate_path: str = "solution.py"
    test_command: list[str] | None = None
    files: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("candidate_path")
    @classmethod
    def safe_candidate_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("candidate_path must stay inside the task workspace")
        return value


class VerifierResult(StrictModel):
    reward: float = Field(ge=0, le=1)
    passed: bool
    verifier: str
    reason: str
    extracted_answer: str | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = Field(default=0, ge=0)
    error_type: str | None = None


class DistillationTrace(StrictModel):
    """Auditable token-level contract for one OPD completion."""

    teacher_name: str
    teacher_revision: str
    token_ids: list[int] = Field(min_length=1)
    rollout_logprobs: list[float] = Field(min_length=1)
    teacher_logprobs: list[float] = Field(min_length=1)
    trainable_mask: list[bool] = Field(min_length=1)
    teacher_topk_token_ids: list[list[int]] | None = None
    teacher_topk_logprobs: list[list[float]] | None = None

    @model_validator(mode="after")
    def aligned_token_fields(self) -> DistillationTrace:
        size = len(self.token_ids)
        for name, values in (
            ("rollout_logprobs", self.rollout_logprobs),
            ("teacher_logprobs", self.teacher_logprobs),
            ("trainable_mask", self.trainable_mask),
        ):
            if len(values) != size:
                raise ValueError(f"{name} must align one-for-one with token_ids")
        if not any(self.trainable_mask):
            raise ValueError("distillation trace has no trainable response tokens")
        if not all(math.isfinite(value) for value in self.rollout_logprobs + self.teacher_logprobs):
            raise ValueError("distillation log-probabilities must be finite")
        if (self.teacher_topk_token_ids is None) != (self.teacher_topk_logprobs is None):
            raise ValueError("teacher top-k token IDs and log-probabilities must be recorded together")
        if self.teacher_topk_token_ids is not None and self.teacher_topk_logprobs is not None:
            if len(self.teacher_topk_token_ids) != size or len(self.teacher_topk_logprobs) != size:
                raise ValueError("teacher top-k fields must align one-for-one with token_ids")
            for ids, logprobs in zip(self.teacher_topk_token_ids, self.teacher_topk_logprobs, strict=True):
                if len(ids) != len(logprobs) or not all(math.isfinite(value) for value in logprobs):
                    raise ValueError("each teacher top-k ID row must align with finite log-probabilities")
        return self

    @property
    def sampled_reverse_kl(self) -> float:
        estimates = [
            rollout - teacher
            for rollout, teacher, active in zip(
                self.rollout_logprobs, self.teacher_logprobs, self.trainable_mask, strict=True
            )
            if active
        ]
        return sum(estimates) / len(estimates)


class TrajectoryRecord(StrictModel):
    run_id: str
    task_id: str
    task_kind: Literal["math", "code", "prompt"]
    prompt: str
    completion: str
    verifier: VerifierResult | None = None
    distillation: DistillationTrace | None = None
    policy_version: int = Field(ge=0)
    learner_version: int = Field(ge=0)
    response_tokens: int = Field(ge=0)
    max_response_tokens: int = Field(gt=0)
    entropy: float | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.response_tokens >= self.max_response_tokens

    @property
    def policy_lag(self) -> int:
        return max(0, self.learner_version - self.policy_version)


class AgentTrajectoryIndexRecord(StrictModel):
    """Small campaign index for a raw Harbor ATIF trajectory.

    The ATIF document remains the source of truth for multi-turn messages,
    actions, observations, and per-token data.  This record contains only the
    fields needed to select and audit trajectories before learner ingestion.
    """

    run_id: str
    trial_id: str
    task_name: str
    task_checksum: str | None = None
    result_path: str
    result_sha256: str
    atif_path: str | None = None
    atif_sha256: str | None = None
    atif_schema_version: str | None = None
    agent_name: str
    agent_version: str | None = None
    model_name: str | None = None
    reward: float | None = None
    reward_components: dict[str, float] = Field(default_factory=dict)
    policy_version: int = Field(ge=0)
    learner_version: int = Field(ge=0)
    policy_lag: int = Field(ge=0)
    accepted_for_rl: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    step_count: int = Field(ge=0)
    fresh_step_count: int = Field(ge=0)
    llm_call_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    trainable_agent_steps: int = Field(ge=0)
    agent_steps_with_token_ids: int = Field(ge=0)
    agent_steps_with_logprobs: int = Field(ge=0)
    copied_context_steps: int = Field(ge=0)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
