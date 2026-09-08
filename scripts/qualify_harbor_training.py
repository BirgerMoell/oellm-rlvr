"""Qualify an end-to-end SkyRL optimizer run from Harbor trajectories."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GRAD_NORM = re.compile(r"(?<![/A-Za-z_])'grad_norm':\s*([-+0-9.eE]+)")
POLICY_VERSION = re.compile(r"OELLM_HARBOR_POLICY_VERSION=(\d+)")
POLICY_VERSION_METRIC = re.compile(r"'generate/policy_weight_version':\s*'?([0-9]+(?:\.0+)?)'?")


def qualify_training(
    log_path: str | Path,
    rollout_report_path: str | Path,
    export_root: str | Path,
    output_path: str | Path,
    *,
    expected_steps: int,
) -> dict[str, Any]:
    log_file = Path(log_path)
    rollout_file = Path(rollout_report_path)
    export_dir = Path(export_root)
    output_file = Path(output_path)

    log = ANSI.sub("", log_file.read_text(errors="replace"))
    rollout_report = json.loads(rollout_file.read_text())
    grad_norms = [float(value) for value in GRAD_NORM.findall(log)]
    marker_versions = [int(value) for value in POLICY_VERSION.findall(log)]
    metric_versions = [int(float(value)) for value in POLICY_VERSION_METRIC.findall(log)]
    policy_versions = marker_versions or metric_versions
    completed_policy_steps = log.count("Finished: 'policy_train'")
    completed_weight_syncs = log.count("Finished: 'sync_weights'")

    observed = rollout_report.get("observed") or {}
    unique_rewards = observed.get("unique_rewards") or []
    weight_files = sorted(
        str(path.relative_to(export_dir))
        for pattern in ("*.safetensors", "*.bin")
        for path in export_dir.rglob(pattern)
        if path.is_file()
    )
    config_files = sorted(
        str(path.relative_to(export_dir)) for path in export_dir.rglob("config.json") if path.is_file()
    )

    finite_gradients = bool(grad_norms) and all(math.isfinite(value) for value in grad_norms)
    nonzero_gradients = [value for value in grad_norms if value > 0.0]
    gates = {
        "harbor_rollouts_learner_admissible": rollout_report.get("ok") is True,
        "mixed_verifier_rewards": 0.0 in unique_rewards and any(float(value) > 0.0 for value in unique_rewards),
        "optimizer_steps_completed": completed_policy_steps >= expected_steps,
        "finite_gradient_metrics": finite_gradients,
        "nonzero_gradient_observed": bool(nonzero_gradients),
        "weight_syncs_completed": completed_weight_syncs >= expected_steps,
        "post_update_rollout_observed": len(set(policy_versions)) >= 2 and max(policy_versions) > min(policy_versions),
        "training_completed": "Training done!" in log,
        "restartable_hf_export": bool(config_files) and bool(weight_files),
    }
    report = {
        "schema_version": "oellm-harbor-training-qualification-v1",
        "ok": all(gates.values()),
        "inputs": {
            "log": str(log_file.resolve()),
            "rollout_report": str(rollout_file.resolve()),
            "export_root": str(export_dir.resolve()),
            "expected_steps": expected_steps,
        },
        "observed": {
            "optimizer_steps": completed_policy_steps,
            "weight_syncs": completed_weight_syncs,
            "grad_norms": grad_norms,
            "nonzero_grad_norms": nonzero_gradients,
            "policy_versions": policy_versions,
            "unique_rewards": unique_rewards,
            "hf_configs": config_files,
            "hf_weight_files": weight_files,
        },
        "gates": gates,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--rollout-report", required=True)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-steps", required=True, type=int)
    args = parser.parse_args()
    report = qualify_training(
        args.log,
        args.rollout_report,
        args.export_root,
        args.output,
        expected_steps=args.expected_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
