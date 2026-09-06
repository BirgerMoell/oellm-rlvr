from __future__ import annotations

from pathlib import Path

from oellm_rlvr.harbor_tasks import build_harbor_dryrun_pack, validate_harbor_dryrun_pack


def test_harbor_pack_has_four_tasks_per_capability_and_replays(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    report = build_harbor_dryrun_pack(pack, "/immutable/task-runtime.sif")

    assert report["ok"] is True
    assert report["tasks"] == 16
    assert set(report["by_category"].values()) == {4}
    assert all(record["oracle_passes_twice"] for record in report["records"])
    assert all(record["unchanged_environment_fails"] for record in report["records"])
    assert 'docker_image = "/immutable/task-runtime.sif"' in (pack / "function-convert-temperature/task.toml").read_text()

    replay = validate_harbor_dryrun_pack(pack)
    assert replay["ok"] is True
