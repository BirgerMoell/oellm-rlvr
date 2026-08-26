from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

from oellm_rlvr.slurm_sandbox import SlurmApptainerBackend


def _backend(tmp_path: Path, monkeypatch) -> SlurmApptainerBackend:
    image = tmp_path / "image"
    image.mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "nid000001")
    backend = SlurmApptainerBackend(image=str(image), tmp_dir=str(tmp_path))
    backend.start()
    return backend


def test_slurm_command_uses_host_runtime_and_isolated_binds(tmp_path: Path, monkeypatch) -> None:
    backend = _backend(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    environments: list[dict[str, str]] = []
    monkeypatch.setenv("SLURM_LABELIO", "1")

    def fake_run(argv, **kwargs):
        calls.append(argv)
        environments.append(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = backend.run_command("python /workspace/solution.py")
    assert result.stdout == "ok\n"
    argv = calls[0]
    assert argv[:4] == [
        "/hostlib64/ld-linux-x86-64.so.2",
        "--library-path",
        "/hostlib64:/hostusr/lib64:/usr/lib64/slurm",
        "/usr/bin/srun",
    ]
    assert ["-w", "nid000001"] == argv[argv.index("-w") : argv.index("-w") + 2]
    assert "/usr/bin/singularity" in argv
    assert "--containall" in argv
    assert any(value.endswith(":/workspace") for value in argv)
    assert environments[0]["SLURM_LABELIO"] == "0"


def test_slurm_sandbox_file_and_archive_io(tmp_path: Path, monkeypatch) -> None:
    backend = _backend(tmp_path, monkeypatch)
    backend.write_file("/workspace/solution.py", "print(42)\n")
    assert backend.read_file("/workspace/solution.py") == "print(42)\n"

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        content = b"#!/bin/sh\n"
        member = tarfile.TarInfo("tests/test.sh")
        member.size = len(content)
        member.mode = 0o755
        archive.addfile(member, io.BytesIO(content))
    backend.put_archive("/", stream.getvalue())
    assert backend.read_file("/tests/test.sh", binary=True) == b"#!/bin/sh\n"
    assert backend._host_path("/tests/test.sh").stat().st_mode & 0o111


def test_slurm_sandbox_rejects_paths_outside_mounts(tmp_path: Path, monkeypatch) -> None:
    backend = _backend(tmp_path, monkeypatch)
    try:
        backend.write_file("/etc/passwd", "nope")
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("write outside the sandbox mounts was accepted")
