from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class TrajectoryRecord(StrictModel):
    run_id: str
    task_id: str
    task_kind: Literal["math", "code"]
    prompt: str
    completion: str
    verifier: VerifierResult
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
