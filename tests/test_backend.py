import json
from pathlib import Path

from oellm_rlvr.backend import build_backend_argv
from oellm_rlvr.config import load_config

ROOT = Path(__file__).parents[1]


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_math_smoke_is_bounded_and_keeps_online_weight_updates() -> None:
    config = load_config(ROOT / "configs/lumi-math-qwen35-2b-smoke.yaml")
    argv = build_backend_argv(config)
    assert argv[:5] == [config.backend.python, "-u", "-m", "oellm_rlvr.tmax_launcher", config.backend.script]
    assert _value(argv, "--vllm_num_engines") == "4"
    assert _value(argv, "--num_learners_per_node") == "4"
    assert "--active_sampling" not in argv
    assert _value(argv, "--filter_zero_std_samples") == "false"
    assert "--inflight_updates" in argv
    assert _value(argv, "--vllm_gdn_prefill_backend") == "triton"
    assert _value(argv, "--wandb_entity") == "local"
    assert "--tools" not in argv


def test_code_backend_uses_slurm_apptainer_swerl_environment() -> None:
    config = load_config(ROOT / "configs/lumi-code-qwen35-2b-smoke.yaml")
    argv = build_backend_argv(config)
    assert _value(argv, "--tools") == "swerl_vanillux_sandbox"
    tool_config = json.loads(_value(argv, "--tool_configs"))
    assert tool_config["backend"] == "slurm_apptainer"
    assert tool_config["apptainer_binary"] == "/usr/bin/singularity"
    assert tool_config["fakeroot"] is False
    assert tool_config["task_data_dir"].endswith("/data/code-hf-e1cae771-s6/task-data")
    assert tool_config["last_step_warning"] is True
    assert tool_config["append_turns_remaining"] is True
    assert tool_config["tool_call_format_error_feedback"] is True
    assert tool_config["timeout"] == 30
    system_prompt = Path(_value(argv, "--system_prompt_override_file"))
    assert system_prompt == ROOT / "prompts/code-agent-system.txt"
    assert system_prompt.is_file()
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in system_prompt.read_text()


def test_production_profile_enables_active_sampling() -> None:
    config = load_config(ROOT / "configs/lumi-code-qwen35-2b-4node.yaml")
    argv = build_backend_argv(config)
    assert "--active_sampling" in argv


def test_math_active_sampling_profile_runs_two_updates_and_saves_filtered_groups() -> None:
    config = load_config(ROOT / "configs/lumi-math-qwen35-2b-active-sampling.yaml")
    argv = build_backend_argv(config)
    episodes_per_update = config.rollout.unique_prompts * config.rollout.samples_per_prompt

    assert "--active_sampling" in argv
    assert "--filter_zero_std_samples" not in argv
    assert "--save_filtered_rollouts" in argv
    assert config.training.total_episodes == 2 * episodes_per_update
    assert config.training.checkpoint_state_freq == 1
    assert _value(argv, "--checkpoint_state_dir") == f"{config.output.directory}_state"


def test_oellm9b_profile_uses_two_nodes_and_the_staged_sft_checkpoint() -> None:
    config = load_config(ROOT / "configs/lumi-math-oellm9b-256k-sft-active-2node.yaml")
    argv = build_backend_argv(config)
    episodes_per_update = config.rollout.unique_prompts * config.rollout.samples_per_prompt

    assert config.platform.nodes == 2
    assert config.model.name_or_path == "openeurollm/oellm-9b-256k-sft"
    assert config.model.local_path.endswith("/artifacts/models/oellm-9b-256k-sft")
    assert config.training.learner_gpus_per_node == [8]
    assert config.rollout.engines == 1
    assert config.rollout.tensor_parallel_size == 1
    assert config.training.total_episodes == 2 * episodes_per_update
    assert "--active_sampling" in argv
    assert "--save_filtered_rollouts" in argv
    assert "--vllm_gdn_prefill_backend" not in argv


def test_oellm9b_ladder_profile_keeps_bounded_zero_variance_groups() -> None:
    config = load_config(ROOT / "configs/lumi-math-oellm9b-256k-sft-ladder-2node.yaml")
    argv = build_backend_argv(config)
    episodes_per_update = config.rollout.unique_prompts * config.rollout.samples_per_prompt

    assert config.platform.nodes == 2
    assert config.training.learner_gpus_per_node == [8]
    assert config.rollout.engines == 1
    assert config.training.total_episodes == 2 * episodes_per_update
    assert "--active_sampling" not in argv
    assert _value(argv, "--filter_zero_std_samples") == "false"
    assert _value(argv, "--response_length") == "2048"


def test_oellm9b_pilot_runs_ten_restartable_hierarchical_updates() -> None:
    config = load_config(ROOT / "configs/lumi-math-oellm9b-256k-sft-pilot-2node.yaml")
    argv = build_backend_argv(config)
    episodes_per_update = config.rollout.unique_prompts * config.rollout.samples_per_prompt

    assert config.platform.partition == "standard-g"
    assert config.rollout.engines == 8
    assert config.rollout.weight_transfer == "hierarchical"
    assert config.training.learner_gpus_per_node == [8]
    assert config.training.total_episodes == 10 * episodes_per_update
    assert config.training.save_freq == 10
    assert config.training.checkpoint_state_freq == 5
    assert _value(argv, "--checkpoint_state_dir").endswith("math-oellm9b-sft-pilot-10step-state")
    assert "--active_sampling" in argv
    assert "--save_filtered_rollouts" in argv
