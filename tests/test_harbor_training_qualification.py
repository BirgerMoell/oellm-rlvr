from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_qualifier():
    path = Path("scripts/qualify_harbor_training.py")
    spec = importlib.util.spec_from_file_location("qualify_harbor_training", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qualifies_mixed_reward_optimizer_sync_and_post_update_rollout(tmp_path: Path) -> None:
    qualifier = _load_qualifier()
    log = tmp_path / "training.log"
    log.write_text(
        "OELLM_HARBOR_POLICY_VERSION=0\n"
        "Finished: 'policy_train', time cost: 1.00s\n"
        "{'grad_norm': 0.25}\n"
        "Finished: 'sync_weights', time cost: 0.50s\n"
        "OELLM_HARBOR_POLICY_VERSION=1\n"
        "Finished: 'policy_train', time cost: 1.10s\n"
        "{'grad_norm': 0.12}\n"
        "Finished: 'sync_weights', time cost: 0.40s\n"
        "Training done!\n"
    )
    rollouts = tmp_path / "rollouts.json"
    rollouts.write_text(json.dumps({"ok": True, "observed": {"unique_rewards": [0.0, 1.0]}}))
    export = tmp_path / "exports" / "global_step_2"
    export.mkdir(parents=True)
    (export / "config.json").write_text("{}")
    (export / "model.safetensors").write_bytes(b"weights")

    report = qualifier.qualify_training(
        log,
        rollouts,
        tmp_path / "exports",
        tmp_path / "qualification.json",
        expected_steps=2,
    )

    assert report["ok"] is True
    assert report["observed"]["policy_versions"] == [0, 1]
    assert report["observed"]["grad_norms"] == [0.25, 0.12]


def test_rejects_zero_gradient_and_no_post_update_rollout(tmp_path: Path) -> None:
    qualifier = _load_qualifier()
    log = tmp_path / "training.log"
    log.write_text(
        "OELLM_HARBOR_POLICY_VERSION=0\n"
        "Finished: 'policy_train'\n"
        "{'grad_norm': 0.0}\n"
        "Finished: 'sync_weights'\n"
        "Training done!\n"
    )
    rollouts = tmp_path / "rollouts.json"
    rollouts.write_text(json.dumps({"ok": True, "observed": {"unique_rewards": [0.0]}}))
    export = tmp_path / "exports" / "global_step_1"
    export.mkdir(parents=True)
    (export / "config.json").write_text("{}")
    (export / "model.safetensors").write_bytes(b"weights")

    report = qualifier.qualify_training(
        log,
        rollouts,
        tmp_path / "exports",
        tmp_path / "qualification.json",
        expected_steps=1,
    )

    assert report["ok"] is False
    assert report["gates"]["nonzero_gradient_observed"] is False
    assert report["gates"]["post_update_rollout_observed"] is False


def test_reads_skyrl_policy_version_metrics_when_ray_drops_print_markers(tmp_path: Path) -> None:
    qualifier = _load_qualifier()
    log = tmp_path / "training.log"
    log.write_text(
        "{'grad_norm': 1.5}\n"
        "Finished: 'policy_train'\n"
        "Finished: 'sync_weights'\n"
        " 'generate/policy_weight_version': 1,\n"
        "{'grad_norm': 0.75}\n"
        "Finished: 'policy_train'\n"
        "Finished: 'sync_weights'\n"
        " 'generate/policy_weight_version': 2,\n"
        "Training done!\n"
    )
    rollouts = tmp_path / "rollouts.json"
    rollouts.write_text(json.dumps({"ok": True, "observed": {"unique_rewards": [0.0, 1.0]}}))
    export = tmp_path / "exports"
    export.mkdir()
    (export / "config.json").write_text("{}")
    (export / "model.safetensors").write_bytes(b"weights")

    report = qualifier.qualify_training(
        log,
        rollouts,
        export,
        tmp_path / "qualification.json",
        expected_steps=2,
    )

    assert report["ok"] is True
    assert report["observed"]["policy_versions"] == [1, 2]
