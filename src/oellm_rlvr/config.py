from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlatformConfig(StrictModel):
    accelerator: Literal["rocm", "cuda"]
    nodes: int = Field(ge=1)
    gpus_per_node: int = Field(ge=1)
    container: str
    account: str
    partition: str
    qos: str | None = None
    walltime: str = "01:00:00"
    cpus_per_task: int = Field(default=64, ge=1)
    memory: str | None = None
    modules: list[str] = Field(default_factory=list)
    binds: list[str] = Field(default_factory=lambda: ["/scratch", "/project", "/flash", "/appl"])
    ray_port: int = Field(default=6379, ge=1024, le=65535)
    network_interface: str | None = None


class BackendConfig(StrictModel):
    repo_path: str
    commit: str
    python: str = "python"
    script: str = "open_instruct/grpo_fast.py"
    extra_args: list[str] = Field(default_factory=list)


class ModelConfig(StrictModel):
    name_or_path: str
    local_path: str | None = None
    max_prompt_tokens: int = Field(default=2048, ge=1)
    response_tokens: int = Field(default=8192, ge=1)
    per_turn_tokens: int = Field(default=4096, ge=1)
    pack_length: int = Field(default=10240, ge=1)

    @model_validator(mode="after")
    def check_lengths(self) -> ModelConfig:
        if self.per_turn_tokens > self.response_tokens:
            raise ValueError("per_turn_tokens cannot exceed response_tokens")
        if self.pack_length < self.max_prompt_tokens + self.response_tokens:
            raise ValueError("pack_length must cover max_prompt_tokens + response_tokens")
        return self


class DatasetConfig(StrictModel):
    path: str
    weight: float = Field(default=1.0, gt=0)
    split: str = "train"


class SandboxConfig(StrictModel):
    backend: Literal["apptainer", "prepared_apptainer", "docker"] = "apptainer"
    image: str
    binary: str = "singularity"
    fakeroot: bool = False
    task_data_hf_repo: str | None = None
    task_data_dir: str | None = None
    test_timeout: int = Field(default=600, ge=1)
    command_timeout: int = Field(default=120, ge=1)
    cache_dir: str | None = None
    tmp_dir: str | None = None
    prepared_root: str | None = None
    prepared_cache_dir: str | None = None
    prepared_scratch_root: str | None = None


class TaskConfig(StrictModel):
    kind: Literal["math", "code"]
    sandbox: SandboxConfig | None = None
    system_prompt_file: str | None = None
    max_steps: int = Field(default=1, ge=1)
    pool_size: int = Field(default=128, ge=1)

    @model_validator(mode="after")
    def code_needs_sandbox(self) -> TaskConfig:
        if self.kind == "code" and self.sandbox is None:
            raise ValueError("code tasks require task.sandbox")
        if self.kind == "math" and self.sandbox is not None:
            raise ValueError("math tasks use ground-truth verification and must not configure a sandbox")
        return self


class RolloutConfig(StrictModel):
    engines: int = Field(ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    unique_prompts: int = Field(default=8, ge=1)
    samples_per_prompt: int = Field(default=16, ge=2)
    temperature: float = Field(default=1.0, gt=0)
    gpu_memory_utilization: float = Field(default=0.8, gt=0, le=1)
    async_steps: int = Field(default=1, ge=0)
    active_sampling: bool = True
    inflight_updates: bool = True
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    gdn_prefill_backend: str | None = "triton"


class TrainingConfig(StrictModel):
    learner_gpus_per_node: list[int]
    sequence_parallel_size: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-6, gt=0)
    total_episodes: int = Field(default=1024, ge=1)
    epochs: int = Field(default=1, ge=1)
    deepspeed_stage: Literal[2, 3] = 3
    loss: Literal["grpo", "dppo"] = "dppo"
    dppo_divergence_type: Literal["tv", "kl", "js"] = "tv"
    dppo_divergence_threshold: float = Field(default=0.1, gt=0)
    beta: float = Field(default=0.0, ge=0)
    seed: int = 42
    save_freq: int = Field(default=20, ge=0)
    checkpoint_state_freq: int = Field(default=10, ge=0)
    gradient_checkpointing: bool = True
    use_liger_loss: bool = True


class OutputConfig(StrictModel):
    directory: str
    rollout_directory: str
    experiment_name: str
    with_tracking: bool = True
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    wandb_entity: str | None = None

    @model_validator(mode="after")
    def tracking_needs_entity(self) -> OutputConfig:
        if self.with_tracking and self.wandb_mode != "disabled" and not self.wandb_entity:
            raise ValueError("tracked runs require output.wandb_entity; use 'local' for offline runs")
        return self


class GateConfig(StrictModel):
    max_zero_std_fraction: float = Field(default=0.80, ge=0, le=1)
    max_truncation_fraction: float = Field(default=0.15, ge=0, le=1)
    max_error_fraction: float = Field(default=0.02, ge=0, le=1)
    max_policy_lag: int = Field(default=4, ge=0)
    min_mean_reward: float | None = Field(default=None, ge=0, le=1)


class RunConfig(StrictModel):
    version: Literal[1] = 1
    name: str
    platform: PlatformConfig
    backend: BackendConfig
    model: ModelConfig
    datasets: list[DatasetConfig] = Field(min_length=1)
    task: TaskConfig
    rollout: RolloutConfig
    training: TrainingConfig
    output: OutputConfig
    gates: GateConfig = Field(default_factory=GateConfig)
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_topology(self) -> RunConfig:
        learners = self.training.learner_gpus_per_node
        if len(learners) > self.platform.nodes:
            raise ValueError("learner_gpus_per_node has more entries than allocated nodes")
        if any(n < 1 or n > self.platform.gpus_per_node for n in learners):
            raise ValueError("every learner_gpus_per_node value must be in [1, gpus_per_node]")
        learner_total = sum(learners)
        rollout_total = self.rollout.engines * self.rollout.tensor_parallel_size
        capacity = self.platform.nodes * self.platform.gpus_per_node
        if learner_total + rollout_total > capacity:
            raise ValueError(
                f"topology requests {learner_total} learner + {rollout_total} rollout GPUs but capacity is {capacity}"
            )
        if learner_total % self.training.sequence_parallel_size:
            raise ValueError("total learner GPUs must be divisible by sequence_parallel_size")
        train_ranks = learner_total // self.training.sequence_parallel_size
        samples = self.rollout.unique_prompts * self.rollout.samples_per_prompt
        if samples < train_ranks:
            raise ValueError("rollout samples per step must be >= learner data-parallel ranks")
        if self.rollout.unique_prompts < self.rollout.engines:
            raise ValueError("unique_prompts must be >= rollout engines to avoid idle engines")
        if self.rollout.active_sampling and self.rollout.async_steps <= 1:
            raise ValueError("active_sampling requires async_steps > 1")
        if self.task.kind == "code" and self.rollout.async_steps < 1:
            raise ValueError("code rollouts should use at least one async step to hide verifier latency")
        return self

    def as_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text())
    return RunConfig.model_validate(raw)
