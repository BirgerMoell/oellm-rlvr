from pathlib import Path

import pytest
from pydantic import ValidationError

from oellm_rlvr.config import load_config, materialize_config
from oellm_rlvr.topology import build_topology

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("name", "expected_spare_gpus"),
    [
        ("lumi-math-qwen35-2b-smoke.yaml", 0),
        ("lumi-code-qwen35-2b-smoke.yaml", 0),
        ("lumi-math-qwen35-2b-signal-probe.yaml", 0),
        ("lumi-math-qwen35-2b-active-sampling.yaml", 0),
        ("lumi-math-oellm9b-256k-sft-active-2node.yaml", 7),
        ("lumi-math-oellm9b-256k-sft-hierarchical-2node.yaml", 0),
        ("lumi-dryrun-reasoning-oellm9b-16step.yaml", 0),
        ("lumi-dryrun-code-oellm9b-4step.yaml", 0),
        ("lumi-code-qwen35-2b-signal-probe.yaml", 0),
        ("lumi-code-qwen35-2b-4node.yaml", 0),
        ("cuda-code-qwen35-2b-smoke.yaml", 0),
    ],
)
def test_example_profiles_validate(name: str, expected_spare_gpus: int) -> None:
    config = load_config(ROOT / "configs" / name)
    topology = build_topology(config)
    assert topology.spare_gpus == expected_spare_gpus
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


def test_hierarchical_transfer_requires_multiple_tp1_engines() -> None:
    profile = ROOT / "configs/lumi-math-oellm9b-256k-sft-hierarchical-2node.yaml"
    config_type = type(load_config(profile))
    one_engine = load_config(profile).model_dump()
    one_engine["rollout"]["engines"] = 1
    with pytest.raises(ValidationError, match="at least two rollout engines"):
        config_type.model_validate(one_engine)

    tensor_parallel = load_config(profile).model_dump()
    tensor_parallel["rollout"]["engines"] = 4
    tensor_parallel["rollout"]["tensor_parallel_size"] = 2
    with pytest.raises(ValidationError, match="requires tensor_parallel_size=1"):
        config_type.model_validate(tensor_parallel)


def test_materialize_config_binds_checkpoint_dataset_and_outputs(tmp_path: Path) -> None:
    destination = tmp_path / "reasoning.yaml"
    config = materialize_config(
        ROOT / "configs/lumi-dryrun-reasoning-oellm9b-16step.yaml",
        destination,
        run_name="candidate-reasoning-01",
        model_id="openeurollm/candidate@abc123",
        model_path="/scratch/project_465002530/models/candidate",
        output_root="/scratch/project_465002530/campaigns/candidate-01",
        dataset_path="/scratch/project_465002530/data/reasoning-train.parquet",
    )

    assert destination.exists()
    assert config.model.local_path == "/scratch/project_465002530/models/candidate"
    assert config.datasets[0].path.endswith("reasoning-train.parquet")
    assert config.output.directory.endswith("/outputs/candidate-reasoning-01")
    assert config.training.checkpoint_state_directory.endswith("/outputs/candidate-reasoning-01-state")
    assert load_config(destination) == config


def test_materialize_config_rejects_unsafe_run_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_name"):
        materialize_config(
            ROOT / "configs/lumi-dryrun-reasoning-oellm9b-16step.yaml",
            tmp_path / "run.yaml",
            run_name="../escape",
            model_id="org/model",
            model_path="/model",
            output_root="/campaign",
        )
