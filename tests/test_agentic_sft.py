from __future__ import annotations

import json
from pathlib import Path

from oellm_rlvr.agentic_sft import build_agentic_sft_bridge, validate_agentic_sft_bridge


def test_agentic_sft_bridge_is_valid_llamafactory_openai_data(tmp_path: Path) -> None:
    report = build_agentic_sft_bridge(tmp_path)

    assert report["schema_version"] == "oellm-agentic-sft-bridge-v1"
    assert report["records"] == 16
    assert report["categories"] == {
        "function-calling": 4,
        "repository-repair": 4,
        "stateful-tools": 4,
        "terminal-edit": 4,
    }
    assert report["validation"]["ok"] is True
    assert report["validation"]["command_turns"] == 36

    dataset_info = json.loads((tmp_path / "dataset_info.json").read_text())
    entry = dataset_info["oellm_agentic_bridge"]
    assert entry["formatting"] == "sharegpt"
    assert entry["columns"] == {"messages": "messages"}
    assert entry["tags"]["assistant_tag"] == "assistant"

    records = [json.loads(line) for line in (tmp_path / "agentic_bridge.jsonl").read_text().splitlines()]
    clamp = next(record for record in records if record["metadata"]["task_name"] == "openeurollm/repo-repair-clamp")
    assert "Terminus" not in clamp["messages"][0]["content"]
    assert "Format your response as JSON" in clamp["messages"][0]["content"]
    assert "return min(low, max(value, high))" in clamp["messages"][2]["content"]
    assert "return max(low, min(value, high))" in clamp["messages"][3]["content"]
    assert all("private-repair" not in message["content"] for message in clamp["messages"])
    for record in records:
        for message in record["messages"]:
            if message["role"] == "assistant":
                payload = json.loads(message["content"])
                assert set(payload) == {"analysis", "plan", "commands", "task_complete"}


def test_agentic_sft_bridge_rejects_non_newline_command(tmp_path: Path) -> None:
    build_agentic_sft_bridge(tmp_path)
    path = tmp_path / "agentic_bridge.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    payload = json.loads(records[0]["messages"][1]["content"])
    payload["commands"][0]["keystrokes"] = "broken"
    records[0]["messages"][1]["content"] = json.dumps(payload)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    report = validate_agentic_sft_bridge(path)
    assert report["ok"] is False
    assert "command must end with newline" in "\n".join(report["errors"])
