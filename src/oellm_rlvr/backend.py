from __future__ import annotations

import json
import shlex
from pathlib import Path

from .config import RunConfig


def _flag(argv: list[str], name: str, enabled: bool) -> None:
    if enabled:
        argv.append(name)


def _tool_config(config: RunConfig) -> dict[str, object]:
    sandbox = config.task.sandbox
    if sandbox is None:
        return {}
    values: dict[str, object] = {
        "backend": sandbox.backend,
        "image": sandbox.image,
        "test_timeout": sandbox.test_timeout,
        "timeout": sandbox.command_timeout,
        "fakeroot": sandbox.fakeroot,
    }
    if sandbox.backend in {"apptainer", "prepared_apptainer", "slurm_apptainer"}:
        values["apptainer_binary"] = sandbox.binary
    optional = (
        "task_data_hf_repo",
        "task_data_dir",
        "cache_dir",
        "tmp_dir",
        "prepared_root",
        "prepared_cache_dir",
        "prepared_scratch_root",
    )
    for key in optional:
        value = getattr(sandbox, key)
        if value:
            values[key] = value
    return values


def build_backend_argv(config: RunConfig) -> list[str]:
    """Build argv for the pinned Open-Instruct/TMAX GRPO entry point."""
    if config.model.local_path:
        argv = [config.backend.python, "-u", "-m", "oellm_rlvr.tmax_launcher", config.backend.script]
    else:
        argv = [config.backend.python, "-u", config.backend.script]
    argv.append("--dataset_mixer_list")
    for dataset in config.datasets:
        argv.extend([dataset.path, str(dataset.weight)])
    argv.append("--dataset_mixer_list_splits")
    argv.extend(dataset.split for dataset in config.datasets)
    argv.extend(
        [
            "--max_prompt_token_length",
            str(config.model.max_prompt_tokens),
            "--per_turn_max_tokens",
            str(config.model.per_turn_tokens),
            "--response_length",
            str(config.model.response_tokens),
            "--pack_length",
            str(config.model.pack_length),
            "--per_device_train_batch_size",
            "1",
            "--num_unique_prompts_rollout",
            str(config.rollout.unique_prompts),
            "--num_samples_per_prompt_rollout",
            str(config.rollout.samples_per_prompt),
            "--async_steps",
            str(config.rollout.async_steps),
            "--model_name_or_path",
            config.model.name_or_path,
            "--temperature",
            str(config.rollout.temperature),
            "--learning_rate",
            str(config.training.learning_rate),
            "--total_episodes",
            str(config.training.total_episodes),
            "--lr_scheduler_type",
            "constant",
            "--deepspeed_stage",
            str(config.training.deepspeed_stage),
            "--sequence_parallel_size",
            str(config.training.sequence_parallel_size),
            "--num_epochs",
            str(config.training.epochs),
            "--num_learners_per_node",
            *(str(value) for value in config.training.learner_gpus_per_node),
            "--vllm_num_engines",
            str(config.rollout.engines),
            "--vllm_tensor_parallel_size",
            str(config.rollout.tensor_parallel_size),
            "--vllm_gpu_memory_utilization",
            str(config.rollout.gpu_memory_utilization),
            "--beta",
            str(config.training.beta),
            "--use_vllm_logprobs",
            "true",
            "--truncated_importance_sampling_ratio_cap",
            "0.0",
            "--seed",
            str(config.training.seed),
            "--push_to_hub",
            "false",
            "--save_traces",
            "--save_trainer_logprobs",
            "true",
            "--verification_reward",
            "1.0",
            "--checkpoint_state_freq",
            str(config.training.checkpoint_state_freq),
            "--advantage_normalization_type",
            "centered",
            "--loss_fn",
            config.training.loss,
            "--rollouts_save_path",
            config.output.rollout_directory,
            "--output_dir",
            config.output.directory,
            "--exp_name",
            config.output.experiment_name,
            "--local_eval_every",
            "10",
            "--save_freq",
            str(config.training.save_freq),
            "--try_launch_beaker_eval_jobs_on_weka",
            "False",
        ]
    )
    _flag(argv, "--gradient_checkpointing", config.training.gradient_checkpointing)
    _flag(argv, "--vllm_enable_prefix_caching", config.rollout.enable_prefix_caching)
    _flag(argv, "--vllm_enforce_eager", config.rollout.enforce_eager)
    _flag(argv, "--active_sampling", config.rollout.active_sampling)
    if not config.rollout.active_sampling:
        argv.extend(["--filter_zero_std_samples", "false"])
    if config.rollout.gdn_prefill_backend:
        argv.extend(["--vllm_gdn_prefill_backend", config.rollout.gdn_prefill_backend])
    if config.rollout.inflight_updates:
        argv.extend(["--inflight_updates", "true"])
    if config.output.with_tracking and config.output.wandb_mode != "disabled":
        argv.append("--with_tracking")
        argv.extend(["--wandb_entity", config.output.wandb_entity])
    if config.training.use_liger_loss:
        argv.extend(["--lm_head_fp32", "true", "--use_liger_grpo_loss", "--liger_grpo_loss_chunk_size", "8"])
    if config.training.loss == "dppo":
        argv.extend(
            [
                "--dppo_divergence_type",
                config.training.dppo_divergence_type,
                "--dppo_divergence_threshold",
                str(config.training.dppo_divergence_threshold),
            ]
        )
    if config.task.kind == "code":
        argv.extend(
            [
                "--tools",
                "swerl_vanillux_sandbox",
                "--tool_configs",
                json.dumps(_tool_config(config), separators=(",", ":")),
                "--pool_size",
                str(config.task.pool_size),
                "--max_steps",
                str(config.task.max_steps),
                "--tool_parser_type",
                "vllm_qwen3_xml",
                "--backend_timeout",
                "1200",
            ]
        )
        if config.task.system_prompt_file:
            argv.extend(["--system_prompt_override_file", config.task.system_prompt_file])
    argv.extend(config.backend.extra_args)
    return argv


def shell_command(config: RunConfig) -> str:
    return shlex.join(build_backend_argv(config))


def assert_backend_revision(config: RunConfig) -> None:
    """Fail before allocating GPUs if the checkout is not the configured revision."""
    head_file = Path(config.backend.repo_path) / ".git" / "HEAD"
    if not head_file.exists():
        raise RuntimeError(f"backend repository is missing: {config.backend.repo_path}")
