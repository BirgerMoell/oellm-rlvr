from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepositoryPin(StrictModel):
    name: str
    url: str
    ref: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    decision: Literal["primary", "fallback", "reference", "watch", "reject"]
    role: str


class StageResources(StrictModel):
    nodes: int = Field(default=0, ge=0)
    gcds_per_node: int = Field(default=8, ge=1)
    expected_gcd_hours: float = Field(default=0.0, ge=0)
    hard_ceiling_gcd_hours: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def check_budget(self) -> StageResources:
        if self.expected_gcd_hours > self.hard_ceiling_gcd_hours:
            raise ValueError("expected_gcd_hours cannot exceed hard_ceiling_gcd_hours")
        if self.nodes == 0 and self.hard_ceiling_gcd_hours != 0:
            raise ValueError("CPU-only stages (nodes=0) must have a zero GCD-hour ceiling")
        if self.nodes > 0 and self.hard_ceiling_gcd_hours == 0:
            raise ValueError("GPU stages must have a positive GCD-hour ceiling")
        return self


class CampaignStage(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str
    day: str
    lane: Literal["foundation", "single_turn", "agentic", "recovery", "evaluation", "contingency"]
    backend: str
    readiness: Literal["ready", "build_required", "contingency"]
    required: bool = True
    needs: list[str] = Field(default_factory=list)
    objective: str
    commands: list[str] = Field(default_factory=list)
    implementation_tasks: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(min_length=1)
    gates: list[str] = Field(min_length=1)
    resources: StageResources = Field(default_factory=StageResources)

    @model_validator(mode="after")
    def check_readiness(self) -> CampaignStage:
        if self.readiness == "ready" and not self.commands:
            raise ValueError("ready stages require at least one command")
        if self.readiness == "build_required" and not self.implementation_tasks:
            raise ValueError("build_required stages require implementation_tasks")
        if self.readiness == "contingency" and self.required:
            raise ValueError("contingency stages must set required=false")
        if not self.required and self.readiness != "contingency":
            raise ValueError("only contingency stages may set required=false")
        return self


class CampaignConfig(StrictModel):
    version: Literal[1] = 1
    name: str
    objective: str
    starting_checkpoint: str
    replacement_checkpoint: str
    hard_ceiling_gcd_hours: float = Field(gt=0)
    repositories: list[RepositoryPin] = Field(min_length=1)
    stages: list[CampaignStage] = Field(min_length=1)
    campaign_gates: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_graph_and_budget(self) -> CampaignConfig:
        repository_names = [repository.name for repository in self.repositories]
        duplicates = sorted(name for name, count in Counter(repository_names).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate repository names: {', '.join(duplicates)}")

        stage_ids = [stage.id for stage in self.stages]
        duplicates = sorted(stage_id for stage_id, count in Counter(stage_ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate stage ids: {', '.join(duplicates)}")

        positions = {stage_id: index for index, stage_id in enumerate(stage_ids)}
        for index, stage in enumerate(self.stages):
            for dependency in stage.needs:
                if dependency not in positions:
                    raise ValueError(f"stage {stage.id} depends on unknown stage {dependency}")
                if positions[dependency] >= index:
                    raise ValueError(f"stage {stage.id} dependency {dependency} must appear earlier")

        required_ceiling = sum(stage.resources.hard_ceiling_gcd_hours for stage in self.stages if stage.required)
        if required_ceiling > self.hard_ceiling_gcd_hours:
            raise ValueError(
                f"required stage ceilings total {required_ceiling:g} GCD-hours, "
                f"above campaign ceiling {self.hard_ceiling_gcd_hours:g}"
            )
        return self

    def summary(self) -> dict[str, object]:
        required = [stage for stage in self.stages if stage.required]
        contingency = [stage for stage in self.stages if not stage.required]
        return {
            "valid": True,
            "name": self.name,
            "starting_checkpoint": self.starting_checkpoint,
            "replacement_checkpoint": self.replacement_checkpoint,
            "stage_count": len(self.stages),
            "ready_stage_count": sum(stage.readiness == "ready" for stage in required),
            "build_required": [stage.id for stage in required if stage.readiness == "build_required"],
            "required_expected_gcd_hours": sum(stage.resources.expected_gcd_hours for stage in required),
            "required_hard_ceiling_gcd_hours": sum(stage.resources.hard_ceiling_gcd_hours for stage in required),
            "campaign_hard_ceiling_gcd_hours": self.hard_ceiling_gcd_hours,
            "contingency_expected_gcd_hours": sum(stage.resources.expected_gcd_hours for stage in contingency),
            "contingency_hard_ceiling_gcd_hours": sum(
                stage.resources.hard_ceiling_gcd_hours for stage in contingency
            ),
        }


def load_campaign(path: str | Path) -> CampaignConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return CampaignConfig.model_validate(raw)


def render_campaign_markdown(campaign: CampaignConfig) -> str:
    summary = campaign.summary()
    lines = [
        f"# {campaign.name}",
        "",
        campaign.objective,
        "",
        f"- Dry-run checkpoint: `{campaign.starting_checkpoint}`",
        f"- Production replacement: `{campaign.replacement_checkpoint}`",
        (
            "- Required budget: "
            f"{summary['required_expected_gcd_hours']:g} expected / "
            f"{summary['required_hard_ceiling_gcd_hours']:g} hard-ceiling GCD-hours"
        ),
        f"- Campaign ceiling: {campaign.hard_ceiling_gcd_hours:g} GCD-hours",
        "",
        "## Stages",
        "",
        "| Stage | Day | Lane | Backend | Readiness | Needs | Expected / ceiling GCDh |",
        "|---|---:|---|---|---|---|---:|",
    ]
    for stage in campaign.stages:
        needs = ", ".join(stage.needs) or "—"
        budget = f"{stage.resources.expected_gcd_hours:g} / {stage.resources.hard_ceiling_gcd_hours:g}"
        lines.append(
            f"| `{stage.id}` | {stage.day} | {stage.lane} | {stage.backend} | "
            f"{stage.readiness} | {needs} | {budget} |"
        )

    build_required = [stage for stage in campaign.stages if stage.required and stage.readiness == "build_required"]
    lines.extend(["", "## Work required before launch", ""])
    if not build_required:
        lines.append("All required stages are marked ready.")
    else:
        for stage in build_required:
            lines.append(f"### `{stage.id}` — {stage.title}")
            lines.append("")
            lines.extend(f"- {task}" for task in stage.implementation_tasks)
            lines.append("")

    lines.extend(["## Campaign gates", ""])
    lines.extend(f"- {gate}" for gate in campaign.campaign_gates)
    lines.append("")
    return "\n".join(lines)
