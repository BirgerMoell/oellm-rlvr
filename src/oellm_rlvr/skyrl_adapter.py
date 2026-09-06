from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .datasets import write_rows


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Parquet input requires pyarrow (install oellm-rlvr[data])") from error
        return pq.read_table(path).to_pylist()
    result = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not a JSON object")
            result.append(value)
    return result


def export_skyrl_math(
    source: str | Path,
    output: str | Path,
    *,
    count: int | None = None,
    copies: int = 1,
) -> dict[str, Any]:
    if copies < 1:
        raise ValueError("copies must be positive")
    source_path = Path(source)
    rows = _rows(source_path)
    if count is not None:
        if count < 1:
            raise ValueError("count must be positive")
        rows = rows[:count]
    converted = []
    for index, row in enumerate(rows):
        messages = row.get("messages") or row.get("prompt")
        ground_truth = row.get("ground_truth")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"row {index} has no messages/prompt list")
        if ground_truth is None:
            raise ValueError(f"row {index} has no ground_truth")
        prepared_messages = [dict(message) for message in messages]
        if prepared_messages[-1].get("role") != "user":
            raise ValueError(f"row {index} must end with a user message")
        content = str(prepared_messages[-1].get("content", "")).rstrip()
        prepared_messages[-1]["content"] = (
            content + '\nEnd the response with exactly "#### " followed by the numeric answer.'
        )
        source_id = str(row.get("id", f"row-{index:06d}"))
        for copy_index in range(copies):
            converted.append(
                {
                    "data_source": str(row.get("oellm_source_dataset", "oellm-math-rlvr")),
                    "prompt": prepared_messages,
                    "env_class": "gsm8k",
                    "reward_spec": {"method": "rule", "ground_truth": str(ground_truth)},
                    "extra_info": {
                        "source_id": source_id,
                        "copy_index": copy_index,
                        "source_path": str(source_path),
                    },
                }
            )
    output_path = Path(output)
    write_rows(converted, output_path)
    return {"output": str(output_path), "rows": len(converted), "source_rows": len(rows), "copies": copies}
