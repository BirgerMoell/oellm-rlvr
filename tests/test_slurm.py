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


def test_cuda_job_uses_nv_flag() -> None:
    path = ROOT / "configs/cuda-code-qwen35-2b-smoke.yaml"
    rendered = render_slurm(load_config(path), path)
    assert "singularity exec --nv" in rendered
