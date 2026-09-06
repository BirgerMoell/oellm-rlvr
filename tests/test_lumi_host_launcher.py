from __future__ import annotations

from pathlib import Path


def test_host_launcher_drops_outer_container_bind_environment() -> None:
    launcher = Path("scripts/lumi-host-launcher-bin/singularity").read_text()
    for variable in ("APPTAINER_BIND", "APPTAINER_BINDPATH", "SINGULARITY_BIND", "SINGULARITY_BINDPATH"):
        assert f"-u {variable}" in launcher
    assert 'SINGULARITY_TMPDIR="$HARBOR_LUMI_CACHE_ROOT/tmp"' in launcher


def test_skyrl_smoke_preserves_ray_worker_diagnostics() -> None:
    script = Path("scripts/lumi_skyrl_amd_smoke.sbatch").read_text()
    assert 'RAY_NODE_TMP="/tmp/oellm-ray-$SLURM_JOB_ID"' in script
    assert 'RAY_TMPDIR="$RAY_NODE_TMP"' in script
    assert 'cp -a "$RAY_NODE_TMP"/. "$RUN_ROOT/ray"/' in script
    assert "trap preserve_ray_logs EXIT" in script
    assert 'mkdir -p logs "$RUN_ROOT"/{checkpoints,exports,logs,compatibility,ray}' in script
