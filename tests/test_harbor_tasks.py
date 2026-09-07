from __future__ import annotations

import os
import subprocess
from pathlib import Path

import tomllib

from oellm_rlvr.harbor_tasks import build_harbor_dryrun_pack, validate_harbor_dryrun_pack


def test_harbor_pack_has_four_tasks_per_capability_and_replays(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    report = build_harbor_dryrun_pack(pack, "/immutable/task-runtime.sif")

    assert report["ok"] is True
    assert report["tasks"] == 16
    assert set(report["by_category"].values()) == {4}
    assert all(record["oracle_passes_twice"] for record in report["records"])
    assert all(record["unchanged_environment_fails"] for record in report["records"])
    task_config = tomllib.loads((pack / "function-convert-temperature/task.toml").read_text())
    assert task_config["task"]["authors"] == [{"name": "OpenEuroLLM contributors"}]
    assert task_config["environment"]["docker_image"] == "/immutable/task-runtime.sif"
    assert task_config["environment"]["workdir"] == "/tmp/oellm-task"
    assert (pack / "repo-repair-clamp/environment/Dockerfile").read_text() == (
        "FROM scratch\nWORKDIR /tmp/oellm-task\n"
    )
    assert (pack / "terminal-edit-workers/environment/config.json").is_file()
    assert (pack / "terminal-edit-workers/environment/files/setup.sh").is_file()
    assert (pack / "terminal-edit-workers/environment/files/config.json").read_text() == (
        pack / "terminal-edit-workers/environment/config.json"
    ).read_text()
    assert (pack / "repo-repair-clamp/environment/files/app.py").read_text() == (
        pack / "repo-repair-clamp/environment/app.py"
    ).read_text()
    repair_instruction = (pack / "repo-repair-clamp/instruction.md").read_text()
    assert "exactly one command, `cat app.py\\n`" in repair_instruction
    assert "If the terminal output already shows the source" in repair_instruction
    assert "do not run `cat` again" in repair_instruction
    staged = pack / "repo-repair-clamp/environment/files"
    workdir = tmp_path / "sif-workdir"
    workdir.mkdir()
    subprocess.run(
        ["bash", str(staged / "setup.sh")],
        check=True,
        env={**os.environ, "HARBOR_STAGING": str(staged), "WORKDIR": str(workdir)},
    )
    assert (workdir / "app.py").read_text() == (staged / "app.py").read_text()
    assert not (workdir / "setup.sh").exists()

    replay = validate_harbor_dryrun_pack(pack)
    assert replay["ok"] is True
