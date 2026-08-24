from pathlib import Path

import pytest
from pydantic import ValidationError

from oellm_rlvr.config import load_config
from oellm_rlvr.topology import build_topology

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "lumi-math-qwen35-2b-smoke.yaml",
        "lumi-code-qwen35-2b-smoke.yaml",
        "lumi-code-qwen35-2b-4node.yaml",
        "cuda-code-qwen35-2b-smoke.yaml",
    ],
)
def test_example_profiles_validate(name: str) -> None:
    config = load_config(ROOT / "configs" / name)
    topology = build_topology(config)
    assert topology.spare_gpus == 0
    assert topology.samples_per_step >= topology.data_parallel_ranks


def test_oversubscribed_profile_is_rejected() -> None:
    config = load_config(ROOT / "configs/lumi-math-qwen35-2b-smoke.yaml").model_dump()
    config["rollout"]["engines"] = 5
    with pytest.raises(ValidationError, match="topology requests"):
        type(load_config(ROOT / "configs/lumi-math-qwen35-2b-smoke.yaml")).model_validate(config)


def test_code_requires_sandbox() -> None:
    config = load_config(ROOT / "configs/lumi-code-qwen35-2b-smoke.yaml").model_dump()
    config["task"]["sandbox"] = None
    with pytest.raises(ValidationError, match="code tasks require"):
        type(load_config(ROOT / "configs/lumi-code-qwen35-2b-smoke.yaml")).model_validate(config)


def test_active_sampling_requires_two_async_steps() -> None:
    profile = ROOT / "configs/lumi-code-qwen35-2b-4node.yaml"
    config = load_config(profile).model_dump()
    config["rollout"]["async_steps"] = 1
    with pytest.raises(ValidationError, match="active_sampling requires async_steps > 1"):
        type(load_config(profile)).model_validate(config)


def test_tracked_runs_require_wandb_entity() -> None:
    config = load_config(ROOT / "configs/lumi-math-qwen35-2b-smoke.yaml").model_dump()
    config["output"]["wandb_entity"] = None
    with pytest.raises(ValidationError, match="tracked runs require output.wandb_entity"):
        type(load_config(ROOT / "configs/lumi-math-qwen35-2b-smoke.yaml")).model_validate(config)
