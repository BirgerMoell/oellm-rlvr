from __future__ import annotations

import io
import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int


class SlurmApptainerBackend:
    """Run isolated commands with host Apptainer via overlapping Slurm steps.

    Ray workers live inside the training SIF on LUMI, where nested setuid and
    user-namespace Apptainer are unavailable. Each command therefore asks the
    allocation's slurmd to launch the host runtime on the same node. Only a
    per-episode filesystem and the read-only task image are exposed.
    """

    _MAX_OUTPUT_BYTES = 1_000_000
    _MOUNT_POINTS = ("/workspace", "/output", "/logs", "/tests", "/root", "/tmp")

    def __init__(
        self,
        image: str,
        timeout: int = 120,
        pwd: str = "/workspace",
        tmp_dir: str | None = None,
        apptainer_binary: str = "/usr/bin/singularity",
        srun_binary: str = "/usr/bin/srun",
        slurm_loader: str = "/hostlib64/ld-linux-x86-64.so.2",
        slurm_library_path: str = "/hostlib64:/hostusr/lib64:/usr/lib64/slurm",
        **_: Any,
    ) -> None:
        self._image = image
        self._timeout = timeout
        self._pwd = pwd
        self._tmp_dir = tmp_dir
        self._apptainer = apptainer_binary
        self._srun = srun_binary
        self._slurm_loader = slurm_loader
        self._slurm_library_path = slurm_library_path
        self._root: Path | None = None
        self._mounts: dict[str, Path] = {}

    def start(self) -> None:
        image = Path(self._image)
        if not image.exists():
            raise FileNotFoundError(f"task sandbox image is missing: {image}")
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("slurm_apptainer requires an active Slurm allocation")
        if self._root is not None:
            return
        if self._tmp_dir:
            Path(self._tmp_dir).mkdir(parents=True, exist_ok=True)
        self._root = Path(tempfile.mkdtemp(prefix="oellm-slurm-sandbox-", dir=self._tmp_dir))
        self._mounts = {}
        for mount_point in self._MOUNT_POINTS:
            host_path = self._root / mount_point.lstrip("/")
            host_path.mkdir(parents=True)
            self._mounts[mount_point] = host_path
        # Vanillux treats /app as its persistent working directory.
        self._mounts["/app"] = self._mounts["/workspace"]

    def _ensure_started(self) -> None:
        if self._root is None:
            raise RuntimeError("slurm_apptainer backend is not started")

    def _host_path(self, container_path: str) -> Path:
        self._ensure_started()
        normalized = "/" + container_path.lstrip("/")
        for mount_point in sorted(self._mounts, key=len, reverse=True):
            if normalized == mount_point or normalized.startswith(mount_point + "/"):
                relative = normalized[len(mount_point) :].lstrip("/")
                root = self._mounts[mount_point].resolve()
                target = (root / relative).resolve()
                if target != root and root not in target.parents:
                    break
                return target
        raise ValueError(f"path is outside the sandbox's writable mounts: {container_path}")

    def _command(self, command: str) -> list[str]:
        node = os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".", 1)[0]
        argv = [
            self._slurm_loader,
            "--library-path",
            self._slurm_library_path,
            self._srun,
            "--overlap",
            "--cpu-bind=none",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            "-w",
            node,
            self._apptainer,
            "exec",
            "--containall",
            "--cleanenv",
            "--no-home",
            "--pwd",
            self._pwd,
        ]
        for mount_point, host_path in self._mounts.items():
            argv.extend(["--bind", f"{host_path}:{mount_point}"])
        argv.extend([self._image, "/bin/bash", "-lc", command])
        return argv

    def run_command(self, command: str, timeout: int | None = None) -> ExecutionResult:
        self._ensure_started()
        effective_timeout = self._timeout if timeout is None else timeout
        try:
            proc = subprocess.run(
                self._command(command),
                capture_output=True,
                timeout=effective_timeout + 30,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stdout = (error.stdout or b"")[: self._MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            stderr_bytes = (error.stderr or b"")[: self._MAX_OUTPUT_BYTES]
            stderr = f"Command timed out after {effective_timeout}s.\n" + stderr_bytes.decode(
                "utf-8", errors="replace"
            )
            return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=124)
        return ExecutionResult(
            stdout=proc.stdout[: self._MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
            stderr=proc.stderr[: self._MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"),
            exit_code=proc.returncode,
        )

    def write_file(self, path: str, content: str | bytes) -> None:
        target = self._host_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target.write_text(content)
        else:
            target.write_bytes(content)

    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        target = self._host_path(path)
        if target.is_dir():
            raise IsADirectoryError(f"Path '{path}' is a directory, not a file.")
        if not target.exists():
            raise FileNotFoundError(f"File not found in sandbox: '{path}'")
        return target.read_bytes() if binary else target.read_text()

    def put_archive(self, root: str, tar_bytes: bytes) -> None:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as archive:
            for member in archive.getmembers():
                virtual_path = str(Path(root) / member.name)
                target = self._host_path(virtual_path)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    target.chmod(member.mode & 0o777)
                    continue
                if not member.isfile():
                    raise ValueError(f"unsupported archive member: {member.name}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"archive member has no content: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read())
                target.chmod(member.mode & 0o777)

    def close(self) -> None:
        if self._root is not None:
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = None
        self._mounts = {}
