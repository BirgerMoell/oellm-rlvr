from __future__ import annotations

import json
from pathlib import Path

from .config import RunConfig, VerlRuntimeConfig


def _bool(value: bool) -> str:
    return "True" if value else "False"


def _value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return _bool(value)
    return str(value)


def _override(name: str, value: object) -> str:
    return f"{name}={_value(value)}"


def build_verl_opd_argv(config: RunConfig) -> list[str]:
    """Translate a validated run config into verl's OPD Hydra overrides.

    Resource and recipe invariants live in :class:`RunConfig`; this function
    only performs a deterministic translation to the pinned verl entry point.
    """
    if config.backend.kind != "verl_opd" or config.distillation is None:
        raise ValueError("the verl OPD adapter requires backend.kind='verl_opd' and distillation settings")

    distillation = config.distillation
    runtime = config.backend.verl or VerlRuntimeConfig()
    script = config.backend.script or "verl.trainer.main_ppo"
    samples_per_step = config.rollout.unique_prompts * config.rollout.samples_per_prompt
    mini_batch_size = runtime.ppo_mini_batch_size or config.rollout.unique_prompts
    total_steps = config.training.total_episodes // samples_per_step
    max_model_len = config.model.max_prompt_tokens + config.model.response_tokens + 1
    train_files = json.dumps([dataset.path for dataset in config.datasets], separators=(",", ":"))
    validation_files = json.dumps(runtime.validation_files, separators=(",", ":"))
    loggers = ["console"]
    if config.output.with_tracking and config.output.wandb_mode != "disabled":
        loggers.append("wandb")

    argv = [config.backend.python, "-u", "-m", script]
    settings: list[tuple[str, object]] = [
        ("algorithm.adv_estimator", "grpo"),
        ("algorithm.use_kl_in_reward", False),
        ("data.train_files", train_files),
        ("data.val_files", validation_files),
        # verl counts prompt groups here and expands each group by rollout.n.
        ("data.train_batch_size", config.rollout.unique_prompts),
        ("data.max_prompt_length", config.model.max_prompt_tokens),
        ("data.max_response_length", config.model.response_tokens),
        ("data.filter_overlong_prompts", True),
        ("data.truncation", "error"),
        ("data.shuffle", distillation.shuffle),
        ("actor_rollout_ref.model.path", config.model.local_path),
        ("actor_rollout_ref.model.use_remove_padding", True),
        ("actor_rollout_ref.model.enable_gradient_checkpointing", config.training.gradient_checkpointing),
        ("actor_rollout_ref.actor.strategy", runtime.actor_strategy),
        ("actor_rollout_ref.actor.use_torch_compile", runtime.use_torch_compile),
        ("actor_rollout_ref.actor.optim.lr", config.training.learning_rate),
        ("actor_rollout_ref.actor.ppo_mini_batch_size", mini_batch_size),
        ("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", runtime.ppo_micro_batch_size_per_gpu),
        ("actor_rollout_ref.actor.use_dynamic_bsz", runtime.use_dynamic_batching),
        ("actor_rollout_ref.actor.ppo_max_token_len_per_gpu", runtime.ppo_max_token_len_per_gpu),
        ("actor_rollout_ref.actor.fsdp_config.param_offload", runtime.parameter_offload),
        ("actor_rollout_ref.actor.fsdp_config.optimizer_offload", runtime.optimizer_offload),
        ("actor_rollout_ref.rollout.name", "vllm"),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", config.rollout.tensor_parallel_size),
        ("actor_rollout_ref.rollout.gpu_memory_utilization", config.rollout.gpu_memory_utilization),
        ("actor_rollout_ref.rollout.n", config.rollout.samples_per_prompt),
        ("actor_rollout_ref.rollout.temperature", config.rollout.temperature),
        ("actor_rollout_ref.rollout.max_model_len", max_model_len),
        ("actor_rollout_ref.rollout.enable_prefix_caching", config.rollout.enable_prefix_caching),
        ("actor_rollout_ref.rollout.enforce_eager", config.rollout.enforce_eager),
        ("actor_rollout_ref.rollout.calculate_log_probs", True),
        ("actor_rollout_ref.rollout.log_prob_use_dynamic_bsz", runtime.use_dynamic_batching),
        ("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", runtime.ppo_micro_batch_size_per_gpu),
        ("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu", runtime.ppo_max_token_len_per_gpu),
        ("trainer.balance_batch", True),
        ("trainer.logger", json.dumps(loggers, separators=(",", ":"))),
        ("trainer.project_name", "OpenEuroLLM-OPD"),
        ("trainer.experiment_name", config.output.experiment_name),
        ("trainer.n_gpus_per_node", config.training.learner_gpus_per_node[0]),
        ("trainer.nnodes", len(config.training.learner_gpus_per_node)),
        ("trainer.val_before_train", runtime.validate_before_training),
        ("trainer.save_freq", config.training.save_freq if config.training.save_freq else -1),
        ("trainer.test_freq", runtime.validation_frequency),
        ("trainer.total_epochs", config.training.epochs),
        ("trainer.total_training_steps", total_steps),
        ("trainer.default_local_dir", config.output.directory),
        ("trainer.rollout_data_dir", config.output.rollout_directory),
        ("distillation.enabled", True),
        ("distillation.n_gpus_per_node", distillation.teacher_gpus_per_node[0]),
        ("distillation.nnodes", len(distillation.teacher_gpus_per_node)),
        ("distillation.teacher_key", distillation.teacher_key),
        ("distillation.distillation_loss.loss_mode", distillation.loss_mode),
        ("distillation.distillation_loss.topk", distillation.top_k),
        ("distillation.distillation_loss.use_task_rewards", distillation.use_task_rewards),
        ("distillation.distillation_loss.use_policy_gradient", distillation.use_policy_gradient),
        ("distillation.distillation_loss.distillation_loss_coef", distillation.coefficient),
        ("distillation.distillation_loss.loss_max_clamp", distillation.loss_max_clamp),
        ("distillation.distillation_loss.log_prob_min_clamp", distillation.logprob_min_clamp),
    ]
    argv.extend(_override(name, value) for name, value in settings)

    multi_teacher = len(distillation.teachers) > 1
    for teacher in distillation.teachers:
        prefix = f"distillation.teacher_models.{teacher.name}"
        add = "+" if multi_teacher else ""
        inference = teacher.inference
        teacher_settings: list[tuple[str, object]] = [
            ("key", teacher.key),
            ("model_path", teacher.model_path),
            ("num_replicas", teacher.num_replicas),
            ("inference.name", inference.engine),
            ("inference.tensor_model_parallel_size", inference.tensor_parallel_size),
            ("inference.data_parallel_size", inference.data_parallel_size),
            ("inference.pipeline_model_parallel_size", inference.pipeline_parallel_size),
            ("inference.expert_parallel_size", inference.expert_parallel_size),
            ("inference.gpu_memory_utilization", inference.gpu_memory_utilization),
            ("inference.enforce_eager", inference.enforce_eager),
            ("inference.max_num_batched_tokens", inference.max_num_batched_tokens),
            ("inference.max_num_seqs", inference.max_num_sequences),
            ("inference.max_model_len", max_model_len),
        ]
        argv.extend(_override(f"{add}{prefix}.{name}", value) for name, value in teacher_settings)

    reward_path = Path(__file__).with_name("opd_reward.py").resolve()
    reward_function = "compute_math_score" if distillation.use_task_rewards else "compute_score"
    argv.extend(
        [
            _override("custom_reward_function.path", reward_path),
            _override("custom_reward_function.name", reward_function),
        ]
    )
    if runtime.ray_runtime_python:
        argv.append(_override("ray_kwargs.ray_init.runtime_env.py_executable", runtime.ray_runtime_python))
    else:
        argv.append(_override("ray_kwargs.ray_init.runtime_env.py_executable", None))
    argv.extend(config.backend.extra_args)
    return argv
