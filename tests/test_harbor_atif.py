from __future__ import annotations

import json
from pathlib import Path

from oellm_rlvr.harbor_atif import index_harbor_atif


def _write_trial(root: Path, *, with_atif: bool = True) -> Path:
    trial = root / "jobs" / "agent" / "task-one__abc"
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-id",
                "task_name": "openeurollm/task-one",
                "trial_name": "task-one__abc",
                "task_checksum": "a" * 64,
                "config": {"environment": {"type": "singularity"}},
                "agent_info": {"name": "harbor-skyrl", "version": "0.1", "model_info": None},
                "verifier_result": {"rewards": {"reward": 1, "format": 0.5}},
                "exception_info": None,
                "started_at": "2026-09-06T12:00:00Z",
                "finished_at": "2026-09-06T12:01:00Z",
            }
        )
    )
    if with_atif:
        (trial / "agent" / "trajectory.json").write_text(
            json.dumps(
                {
                    "schema_version": "ATIF-v1.7",
                    "session_id": "run-session",
                    "trajectory_id": "trajectory-one",
                    "agent": {"name": "harbor-skyrl", "version": "0.1", "model_name": "oellm-9b"},
                    "steps": [
                        {"step_id": 1, "source": "user", "message": "Fix the program."},
                        {
                            "step_id": 2,
                            "source": "agent",
                            "message": "Old copied response",
                            "is_copied_context": True,
                            "metrics": {"prompt_tokens": 99, "completion_tokens": 99},
                        },
                        {
                            "step_id": 3,
                            "source": "agent",
                            "message": "I will inspect it.",
                            "tool_calls": [
                                {"tool_call_id": "call-1", "function_name": "shell", "arguments": {"cmd": "ls"}}
                            ],
                            "observation": {"results": [{"source_call_id": "call-1", "content": "solution.py"}]},
                            "metrics": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "completion_token_ids": [101, 102],
                                "logprobs": [-0.1, -0.2],
                            },
                            "llm_call_count": 1,
                        },
                        {
                            "step_id": 4,
                            "source": "agent",
                            "message": "deterministic dispatch",
                            "tool_calls": [
                                {"tool_call_id": "call-2", "function_name": "submit", "arguments": {}}
                            ],
                            "llm_call_count": 0,
                        },
                    ],
                }
            )
        )
    return trial


def test_index_harbor_atif_preserves_raw_trace_and_counts_only_fresh_llm_steps(tmp_path: Path) -> None:
    trial = _write_trial(tmp_path)
    output = tmp_path / "index.jsonl"
    report = index_harbor_atif(
        tmp_path,
        output,
        run_id="dryrun-1",
        policy_version=7,
        learner_version=8,
        require_token_ids=True,
        require_logprobs=True,
    )

    assert report["ok"] is True
    record = json.loads(output.read_text())
    assert record["atif_path"] == str((trial / "agent" / "trajectory.json").relative_to(tmp_path))
    assert len(record["atif_sha256"]) == 64
    assert record["policy_lag"] == 1
    assert record["reward_components"] == {"format": 0.5, "reward": 1.0}
    assert record["step_count"] == 4
    assert record["fresh_step_count"] == 3
    assert record["copied_context_steps"] == 1
    assert record["trainable_agent_steps"] == 1
    assert record["llm_call_count"] == 1
    assert record["tool_call_count"] == 2
    assert record["metadata"]["bash_command_count"] == 0
    assert record["metadata"]["task_complete_count"] == 0
    assert record["metadata"]["linked_observation_count"] == 1
    assert record["prompt_tokens"] == 10
    assert record["completion_tokens"] == 2
    assert json.loads((trial / "agent" / "trajectory.json").read_text())["steps"][2]["tool_calls"]


def test_index_harbor_atif_rejects_missing_trace_without_dropping_result(tmp_path: Path) -> None:
    _write_trial(tmp_path, with_atif=False)
    output = tmp_path / "index.jsonl"
    report = index_harbor_atif(
        tmp_path,
        output,
        run_id="dryrun-2",
        policy_version=0,
        learner_version=0,
    )

    assert report["ok"] is False
    assert report["observed"] == {"trials": 1, "accepted": 0, "rejected": 1}
    record = json.loads(output.read_text())
    assert record["accepted_for_rl"] is False
    assert record["atif_path"] is None
    assert "agent/trajectory.json is missing" in record["rejection_reasons"]


def test_index_harbor_atif_rejects_misaligned_logprobs(tmp_path: Path) -> None:
    trial = _write_trial(tmp_path)
    atif_path = trial / "agent" / "trajectory.json"
    payload = json.loads(atif_path.read_text())
    payload["steps"][2]["metrics"]["logprobs"] = [-0.1]
    atif_path.write_text(json.dumps(payload))

    output = tmp_path / "index.jsonl"
    report = index_harbor_atif(
        tmp_path,
        output,
        run_id="dryrun-3",
        policy_version=0,
        learner_version=0,
        require_logprobs=True,
    )

    assert report["ok"] is False
    record = json.loads(output.read_text())
    assert any("logprobs do not align" in reason for reason in record["rejection_reasons"])


def test_index_harbor_atif_rejects_private_verifier_marker_in_trace(tmp_path: Path) -> None:
    trial = _write_trial(tmp_path)
    verifier = trial / "verifier"
    verifier.mkdir()
    (verifier / "result.json").write_text(json.dumps({"reward": 1, "marker": "private-secret-marker"}))
    atif_path = trial / "agent" / "trajectory.json"
    payload = json.loads(atif_path.read_text())
    payload["steps"][2]["message"] = "private-secret-marker"
    atif_path.write_text(json.dumps(payload))

    output = tmp_path / "index.jsonl"
    report = index_harbor_atif(
        tmp_path,
        output,
        run_id="dryrun-4",
        policy_version=0,
        learner_version=0,
    )

    assert report["ok"] is False
    record = json.loads(output.read_text())
    assert "private verifier marker leaked into ATIF trajectory" in record["rejection_reasons"]
