from __future__ import annotations

from pathlib import Path


def test_host_launcher_drops_outer_container_bind_environment() -> None:
    launcher = Path("scripts/lumi-host-launcher-bin/singularity").read_text()
    for variable in ("APPTAINER_BIND", "APPTAINER_BINDPATH", "SINGULARITY_BIND", "SINGULARITY_BINDPATH"):
        assert f"-u {variable}" in launcher
    assert 'SINGULARITY_TMPDIR="$HARBOR_LUMI_CACHE_ROOT/tmp"' in launcher
    assert "unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE" in launcher
    assert "--cpu-bind=none" in launcher


def test_skyrl_smoke_preserves_ray_worker_diagnostics() -> None:
    script = Path("scripts/lumi_skyrl_amd_smoke.sbatch").read_text()
    assert 'RAY_NODE_TMP="/tmp/oellm-ray-$SLURM_JOB_ID"' in script
    assert 'RAY_TMPDIR="$RAY_NODE_TMP"' in script
    assert 'cp -a "$RAY_NODE_TMP"/. "$RUN_ROOT/ray"/' in script
    assert "trap preserve_ray_logs EXIT" in script
    assert 'mkdir -p logs "$RUN_ROOT"/{checkpoints,exports,logs,compatibility,ray}' in script
    assert "generator.inference_engine.gpu_memory_utilization=0.90" in script
    assert "generator.inference_engine.max_num_seqs=16" in script
    assert "generator.inference_engine.distributed_executor_backend=mp" in script
    assert "unset VLLM_USE_V1 VLLM_ENABLE_V1_MULTIPROCESSING" in script
    assert "TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1" not in script
    assert ': "${MAX_GENERATE_LENGTH:=128}"' in script
    assert 'generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH"' in script


def test_agentic_rollout_uses_real_harbor_sandbox_and_rl_trace_gate() -> None:
    script = Path("scripts/lumi_harbor_agentic_rollout.sbatch").read_text()
    assert "examples.train_integrations.harbor.entrypoints.main_harbor_generate" in script
    assert "harbor_trial_config.agent.name=terminus-2" in script
    assert "harbor_trial_config.environment.type=singularity" in script
    assert "harbor_trial_config.agent.kwargs.collect_rollout_details=true" in script
    assert "probe_harbor_litellm_vllm.py" in script
    assert "generator.step_wise_trajectories=true" in script
    assert "generator.merge_stepwise_output=true" in script
    assert "generator.inference_engine.distributed_executor_backend=mp" in script
    assert 'PATH="$LAUNCHER_PATH:' in script
    assert 'cp -a "${SELECTED_TASKS[@]}" "$ROLLOUT_PACK"/' in script
    assert "qualify-harbor-rollouts" in script
    assert "--min-bash-commands-per-trial 1" in script
    assert 'TASK_GLOB:=repo-repair-clamp' in script
    assert '--expected-trials "$EXPECTED_TRIALS"' in script
    assert "srun --label --cpu-bind=none --gpu-bind=none" in script
