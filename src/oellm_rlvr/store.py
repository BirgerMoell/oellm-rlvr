from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from .schemas import TrajectoryRecord


class JsonlTrajectoryStore:
    """Append-only trajectory log; one file per writer is recommended at scale."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: TrajectoryRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump_json() + "\n"
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
        try:
            os.write(fd, payload.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def extend(self, records: Iterable[TrajectoryRecord]) -> None:
        for record in records:
            self.append(record)

    def __iter__(self) -> Iterator[TrajectoryRecord]:
        if not self.path.exists():
            return
        with self.path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield TrajectoryRecord.model_validate_json(line)
                    except Exception as error:
                        raise ValueError(f"invalid trajectory at {self.path}:{line_number}: {error}") from error


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
