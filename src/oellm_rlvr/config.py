from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class VerlRuntimeConfig(StrictModel):
    """Backend-specific knobs for verl's FSDP/vLLM OPD trainer.

    These values are deliberately isolated from the TMAX configuration so
    adding OPD cannot change the argv of an existing RLVR run.
    """

    actor_strategy: Literal["fsdp", "fsdp2"] = "fsdp"
    ppo_mini_batch_size: int | None = Field(default=None, ge=1)
    ppo_micro_batch_size_per_gpu: int = Field(default=1, ge=1)
    ppo_max_token_len_per_gpu: int = Field(default=24576, ge=1)
    use_dynamic_batching: bool = True
    use_torch_compile: bool = False
    parameter_offload: bool = True
    optimizer_offload: bool = True
    validation_files: list[str] = Field(default_factory=list)
    validate_before_training: bool = False
    validation_frequency: int = Field(default=-1, ge=-1)
    ray_runtime_python: str | None = None


class BackendConfig(StrictModel):
    kind: Literal["tmax", "verl_opd"] = "tmax"
    repo_path: str
    commit: str
    python: str = "python"
    script: str | None = None
    verl: VerlRuntimeConfig | None = None
    extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def set_backend_defaults(self) -> BackendConfig:
        if self.script is None:
            self.script = "open_instruct/grpo_fast.py" if self.kind == "tmax" else "verl.trainer.main_ppo"
        if self.kind == "tmax" and self.verl is not None:
            raise ValueError("backend.verl is only valid when backend.kind is 'verl_opd'")
        return self


class ModelConfig(StrictModel):
    name_or_path: str
    local_path: str | None = None
    revision: str | None = None
    manifest_path: str | None = None
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
    backend: Literal["apptainer", "prepared_apptainer", "slurm_apptainer", "docker"] = "apptainer"
    image: str
    binary: str = "singularity"
    fakeroot: bool = False
    task_data_hf_repo: str | None = None
    task_data_dir: str | None = None
    test_timeout: int = Field(default=600, ge=1)
    command_timeout: int = Field(default=120, ge=1)
    last_step_warning: bool = False
    append_turns_remaining: bool = False
    tool_call_format_error_feedback: bool = False
    cache_dir: str | None = None
    tmp_dir: str | None = None
    prepared_root: str | None = None
    prepared_cache_dir: str | None = None
    prepared_scratch_root: str | None = None


class TaskConfig(StrictModel):
    kind: Literal["math", "code", "prompt"]
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
        if self.kind == "prompt" and self.sandbox is not None:
            raise ValueError("prompt-only OPD tasks must not configure a sandbox")
        return self


class RolloutConfig(StrictModel):
    engines: int = Field(ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    weight_transfer: Literal["broadcast", "hierarchical"] = "broadcast"
    unique_prompts: int = Field(default=8, ge=1)
    samples_per_prompt: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    gpu_memory_utilization: float = Field(default=0.8, gt=0, le=1)
    async_steps: int = Field(default=1, ge=0)
    active_sampling: bool = True
    inflight_updates: bool = True
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    gdn_prefill_backend: str | None = "triton"
    colocate_with_learner: bool = False


class TeacherInferenceConfig(StrictModel):
    engine: Literal["vllm", "sglang"] = "vllm"
    tensor_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    expert_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.5, gt=0, le=1)
    enforce_eager: bool = True
    max_num_batched_tokens: int = Field(default=8192, ge=1)
    max_num_sequences: int = Field(default=1024, ge=1)

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.data_parallel_size * self.pipeline_parallel_size

    @model_validator(mode="after")
    def validate_parallelism(self) -> TeacherInferenceConfig:
        if self.expert_parallel_size > 1:
            expected = self.tensor_parallel_size * self.data_parallel_size
            if self.expert_parallel_size != expected:
                raise ValueError(
                    "teacher expert_parallel_size must equal tensor_parallel_size * data_parallel_size"
                )
        return self


class TeacherConfig(StrictModel):
    name: str = "teacher_model"
    key: str = "default"
    name_or_path: str
    local_path: str | None = None
    revision: str
    manifest_path: str
    num_replicas: int = Field(default=1, ge=1)
    inference: TeacherInferenceConfig = Field(default_factory=TeacherInferenceConfig)

    @field_validator("name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", value):
            raise ValueError("teacher name must be a Hydra-safe identifier")
        return value

    @property
    def model_path(self) -> str:
        return self.local_path or self.name_or_path

    @property
    def gpu_count(self) -> int:
        return self.num_replicas * self.inference.world_size


class DistillationConfig(StrictModel):
    teacher_gpus_per_node: list[int] = Field(min_length=1)
    teachers: list[TeacherConfig] = Field(min_length=1)
    teacher_key: str = "data_source"
    loss_mode: Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3", "forward_kl_topk"] = "k1"
    top_k: int = Field(default=64, ge=1)
    use_policy_gradient: bool = True
    use_task_rewards: bool = False
    coefficient: float = Field(default=1.0, gt=0)
    loss_max_clamp: float | None = Field(default=10.0, gt=0)
    logprob_min_clamp: float | None = Field(default=-10.0, lt=0)
    shuffle: bool = True

    @model_validator(mode="after")
    def validate_recipe(self) -> DistillationConfig:
        names = [teacher.name for teacher in self.teachers]
        keys = [teacher.key for teacher in self.teachers]
        if len(names) != len(set(names)):
            raise ValueError("teacher names must be unique")
        if len(keys) != len(set(keys)):
            raise ValueError("teacher routing keys must be unique")
        if len(self.teachers) == 1 and names[0] != "teacher_model":
            raise ValueError("a single verl teacher must be named 'teacher_model'")
        if len(self.teachers) > 1 and "teacher_model" in names:
            raise ValueError("multi-teacher OPD must not use verl's reserved 'teacher_model' entry")
        if self.loss_mode == "k1" and not self.use_policy_gradient:
            raise ValueError("k1 requires use_policy_gradient=true")
        if self.loss_mode == "forward_kl_topk" and self.use_policy_gradient:
            raise ValueError("forward_kl_topk must use direct backpropagation, not policy gradient")
        return self

    @property
    def gpu_count(self) -> int:
        return sum(self.teacher_gpus_per_node)


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
    checkpoint_state_directory: str | None = None
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
    min_distillation_coverage: float = Field(default=0.99, ge=0, le=1)
    max_mean_sampled_reverse_kl: float | None = Field(default=None, ge=0)


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
    distillation: DistillationConfig | None = None
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
        physical_rollout_total = 0 if self.rollout.colocate_with_learner else rollout_total
        teacher_total = self.distillation.gpu_count if self.distillation else 0
        capacity = self.platform.nodes * self.platform.gpus_per_node
        if learner_total + physical_rollout_total + teacher_total > capacity:
            raise ValueError(
                f"topology requests {learner_total} learner + {physical_rollout_total} rollout + "
                f"{teacher_total} teacher GPUs but capacity is {capacity}"
            )
        if learner_total % self.training.sequence_parallel_size:
            raise ValueError("total learner GPUs must be divisible by sequence_parallel_size")
        train_ranks = learner_total // self.training.sequence_parallel_size
        samples = self.rollout.unique_prompts * self.rollout.samples_per_prompt
        if samples < train_ranks:
            raise ValueError("rollout samples per step must be >= learner data-parallel ranks")
        if not self.rollout.colocate_with_learner and self.rollout.unique_prompts < self.rollout.engines:
            raise ValueError("unique_prompts must be >= rollout engines to avoid idle engines")
        if self.rollout.weight_transfer == "hierarchical":
            if self.rollout.engines < 2:
                raise ValueError("hierarchical weight transfer requires at least two rollout engines")
            if self.rollout.tensor_parallel_size != 1:
                raise ValueError("hierarchical weight transfer currently requires tensor_parallel_size=1")
        if self.rollout.active_sampling and self.rollout.async_steps <= 1:
            raise ValueError("active_sampling requires async_steps > 1")
        if self.task.kind == "code" and self.backend.kind == "tmax" and self.rollout.async_steps < 1:
            raise ValueError("code rollouts should use at least one async step to hide verifier latency")

        if self.backend.kind == "tmax":
            if self.distillation is not None:
                raise ValueError("distillation currently requires backend.kind='verl_opd'")
            if self.rollout.colocate_with_learner:
                raise ValueError("the TMAX backend requires a dedicated rollout GPU pool")
            if self.rollout.samples_per_prompt < 2:
                raise ValueError("TMAX GRPO/DPPO requires at least two samples per prompt")
            return self

        if self.distillation is None:
            raise ValueError("backend.kind='verl_opd' requires a distillation section")
        if self.task.kind == "code":
            raise ValueError("verl_opd currently supports prompt and single-turn math tasks; use TMAX for code RLVR")
        if not self.rollout.colocate_with_learner:
            raise ValueError("verl_opd requires rollout.colocate_with_learner=true")
        if rollout_total != learner_total:
            raise ValueError(
                "colocated verl rollout engines must cover every learner GPU: "
                "rollout.engines * tensor_parallel_size must equal total learner GPUs"
            )
        if self.rollout.weight_transfer != "broadcast":
            raise ValueError("hierarchical TMAX weight transfer is not valid for colocated verl rollouts")
        if self.rollout.inflight_updates:
            raise ValueError("verl_opd manages rollout synchronization internally; set inflight_updates=false")
        if self.rollout.async_steps != 0 or self.rollout.active_sampling:
            raise ValueError("verl_opd currently uses synchronous sampling; set async_steps=0 and active_sampling=false")
        if self.rollout.gdn_prefill_backend is not None:
            raise ValueError("gdn_prefill_backend is a TMAX-only option; set it to null for verl_opd")
        if self.training.loss != "grpo":
            raise ValueError("verl_opd uses verl's GRPO-compatible actor path; set training.loss='grpo'")
        if self.training.sequence_parallel_size != 1:
            raise ValueError("verl_opd adapter currently requires sequence_parallel_size=1")
        if self.training.checkpoint_state_freq or self.training.checkpoint_state_directory:
            raise ValueError("verl_opd uses trainer.save_freq checkpoints; disable TMAX checkpoint-state settings")
        if len(set(learners)) != 1:
            raise ValueError("verl_opd requires a uniform learner_gpus_per_node layout")
        if len(self.distillation.teacher_gpus_per_node) > self.platform.nodes:
            raise ValueError("teacher_gpus_per_node has more entries than allocated nodes")
        if any(
            value < 1 or value > self.platform.gpus_per_node
            for value in self.distillation.teacher_gpus_per_node
        ):
            raise ValueError("every teacher_gpus_per_node value must be in [1, gpus_per_node]")
        if len(set(self.distillation.teacher_gpus_per_node)) != 1:
            raise ValueError("verl_opd requires a uniform teacher_gpus_per_node layout")
        configured_teacher_gpus = sum(teacher.gpu_count for teacher in self.distillation.teachers)
        if configured_teacher_gpus != teacher_total:
            raise ValueError(
                f"teacher replicas require {configured_teacher_gpus} GPUs but teacher pool has {teacher_total}"
            )
        teacher_node_width = self.distillation.teacher_gpus_per_node[0]
        teacher_offset = 0
        for teacher in self.distillation.teachers:
            replica_width = teacher.inference.world_size
            expected_span = (replica_width + teacher_node_width - 1) // teacher_node_width
            for replica in range(teacher.num_replicas):
                first_node = teacher_offset // teacher_node_width
                last_node = (teacher_offset + replica_width - 1) // teacher_node_width
                if last_node - first_node + 1 != expected_span:
                    raise ValueError(
                        f"teacher {teacher.name} replica {replica} crosses an avoidable node boundary; "
                        "reorder teachers or adjust replica parallelism"
                    )
                teacher_offset += replica_width
        if not self.model.local_path or not self.model.revision or not self.model.manifest_path:
            raise ValueError("verl_opd requires model.local_path, model.revision, and model.manifest_path")
        if self.training.total_episodes % samples:
            raise ValueError("verl_opd total_episodes must be divisible by rollout samples per step")
        if (
            self.backend.verl is not None
            and self.backend.verl.ppo_mini_batch_size is not None
            and self.rollout.unique_prompts % self.backend.verl.ppo_mini_batch_size
        ):
            raise ValueError("verl ppo_mini_batch_size must divide rollout.unique_prompts")
        if self.distillation.use_task_rewards and self.task.kind == "prompt":
            raise ValueError("prompt-only OPD has no verifier reward; set use_task_rewards=false")
        if self.distillation.use_task_rewards and self.rollout.samples_per_prompt < 2:
            raise ValueError("hybrid OPD+RLVR requires at least two samples per prompt")
        if any(dataset.weight != 1.0 for dataset in self.datasets):
            raise ValueError("verl_opd does not map dataset weights; pre-mix data and set every weight to 1.0")
        return self

    def as_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    raw: Any = yaml.safe_load(config_path.read_text())
    config = RunConfig.model_validate(raw)
    system_prompt = config.task.system_prompt_file
    if system_prompt and not Path(system_prompt).is_absolute():
        config.task.system_prompt_file = str((config_path.parent / system_prompt).resolve())
    return config


def materialize_config(
    template: str | Path,
    output: str | Path,
    *,
    run_name: str,
    model_id: str,
    model_path: str,
    output_root: str,
    dataset_path: str | None = None,
) -> RunConfig:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run_name):
        raise ValueError("run_name must contain only lowercase letters, digits, underscores, and hyphens")

    config = load_config(template)
    raw = config.model_dump(mode="json")
    raw["name"] = run_name
    raw["model"]["name_or_path"] = model_id
    raw["model"]["local_path"] = model_path
    if dataset_path is not None:
        if len(raw["datasets"]) != 1:
            raise ValueError("--dataset can only replace a template containing exactly one dataset")
        raw["datasets"][0]["path"] = dataset_path

    root = output_root.rstrip("/")
    raw["output"]["directory"] = f"{root}/outputs/{run_name}"
    raw["output"]["rollout_directory"] = f"{root}/rollouts/{run_name}"
    raw["output"]["experiment_name"] = run_name.replace("-", "_")
    raw["training"]["checkpoint_state_directory"] = f"{root}/outputs/{run_name}-state"

    materialized = RunConfig.model_validate(raw)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(materialized.model_dump(mode="json"), sort_keys=False))
    return materialized
