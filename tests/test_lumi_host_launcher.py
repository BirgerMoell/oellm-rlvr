from __future__ import annotations

from pathlib import Path


def test_host_launcher_drops_outer_container_bind_environment() -> None:
    launcher = Path("scripts/lumi-host-launcher-bin/singularity").read_text()
    for variable in ("APPTAINER_BIND", "APPTAINER_BINDPATH", "SINGULARITY_BIND", "SINGULARITY_BINDPATH"):
        assert f"-u {variable}" in launcher
    assert 'SINGULARITY_TMPDIR="$HARBOR_LUMI_CACHE_ROOT/tmp"' in launcher
