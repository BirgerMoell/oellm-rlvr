from pathlib import Path

from oellm_rlvr.config import load_config
from oellm_rlvr.slurm import render_slurm

ROOT = Path(__file__).parents[1]


def test_lumi_job_renders_ray_and_rocm_preflight() -> None:
    path = ROOT / "configs/lumi-code-qwen35-2b-smoke.yaml"
    rendered = render_slurm(load_config(path), path)
    assert "#SBATCH --gpus-per-node=8" in rendered
    assert "singularity exec --rocm" not in rendered
    assert "singularity exec  -B" in rendered
    assert "scripts/ray_node.sh" in rendered
    assert "run-backend --config" in rendered
    assert "actual_commit" in rendered
    assert 'export TRITON_CACHE_DIR="$RAY_TMP/triton-cache"' in rendered
    assert 'export XDG_CACHE_HOME="$RAY_TMP/xdg-cache"' in rendered
    assert 'export MIOPEN_USER_DB_PATH="$RAY_TMP/miopen-cache"' in rendered
    assert 'export JOB_TMPDIR="$RAY_TMP/tmp"' in rendered
    assert "mkdir -p \"$JOB_TMPDIR\" \"$XDG_CACHE_HOME\"" in rendered
    assert 'export TMPDIR="$JOB_TMPDIR"' in rendered
    ray_launch = rendered.split("bash \"$CONTROL_ROOT/scripts/ray_node.sh\"", 1)[0].rsplit("srun --label", 1)[1]
    assert '--gpus-per-task="$GPUS_PER_NODE"' in ray_launch
    assert "--ntasks-per-node=1" in ray_launch
    assert "--overlap" not in ray_launch


def test_cuda_job_uses_nv_flag() -> None:
    path = ROOT / "configs/cuda-code-qwen35-2b-smoke.yaml"
    rendered = render_slurm(load_config(path), path)
    assert "singularity exec --nv" in rendered


def test_hierarchical_job_exports_weight_transfer_mode() -> None:
    path = ROOT / "configs/lumi-math-oellm9b-256k-sft-hierarchical-2node.yaml"
    rendered = render_slurm(load_config(path), path)
    assert 'export OELLM_WEIGHT_TRANSFER="hierarchical"' in rendered
