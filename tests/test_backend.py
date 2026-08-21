import json
from pathlib import Path

from oellm_rlvr.backend import build_backend_argv
from oellm_rlvr.config import load_config

ROOT = Path(__file__).parents[1]


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_math_backend_has_online_rollout_and_active_sampling_flags() -> None:
    config = load_config(ROOT / "configs/lumi-math-qwen35-2b-smoke.yaml")
    argv = build_backend_argv(config)
    assert argv[:3] == [config.backend.python, "-u", "open_instruct/grpo_fast.py"]
    assert _value(argv, "--vllm_num_engines") == "4"
    assert _value(argv, "--num_learners_per_node") == "4"
    assert "--active_sampling" in argv
    assert "--inflight_updates" in argv
    assert _value(argv, "--vllm_gdn_prefill_backend") == "triton"
    assert "--tools" not in argv


def test_code_backend_uses_apptainer_swerl_environment() -> None:
    config = load_config(ROOT / "configs/lumi-code-qwen35-2b-smoke.yaml")
    argv = build_backend_argv(config)
    assert _value(argv, "--tools") == "swerl_vanillux_sandbox"
    tool_config = json.loads(_value(argv, "--tool_configs"))
    assert tool_config["backend"] == "apptainer"
    assert tool_config["apptainer_binary"] == "singularity"
    assert tool_config["fakeroot"] is False
    assert tool_config["task_data_dir"].endswith("/data/code-smoke/task-data")
