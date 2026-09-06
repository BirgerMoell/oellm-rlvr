from __future__ import annotations

import json
from pathlib import Path

from oellm_rlvr.harbor_qualification import qualify_harbor_rollouts


def _trial(root: Path, *, with_command: bool = True) -> Path:
    trial = root / "jobs" / "repo-repair__abc"
    (trial / "agent").mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-1",
                "task_name": "openeurollm/repo-repair",
                "trial_name": "repo-repair__abc",
                "config": {"environment": {"type": "singularity"}},
                "agent_info": {"name": "terminus-2", "version": "2.0.0"},
                "verifier_result": {"rewards": {"reward": 0}},
                "exception_info": None,
            }
        )
    )
    tool_calls = []
    observations = [{"content": "Current terminal state"}]
    if with_command:
        tool_calls.append(
            {
                "tool_call_id": "call-1",
                "function_name": "bash_command",
                "arguments": {"keystrokes": "sed -n '1,20p' app.py\n", "duration": 0.1},
            }
        )
        observations = [{"source_call_id": "call-1", "content": "def broken(): pass"}]
    (trial / "agent" / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "session-1",
                "agent": {"name": "terminus-2", "version": "2.0.0", "model_name": "checkpoint"},
                "steps": [
                    {"step_id": 1, "source": "user", "message": "Repair app.py"},
                    {
                        "step_id": 2,
                        "source": "agent",
                        "message": "Inspect first.",
                        "tool_calls": tool_calls,
                        "observation": {"results": observations},
                        "metrics": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "completion_token_ids": [4, 5],
                            "logprobs": [-0.1, -0.2],
                        },
                    },
                ],
            }
        )
    )
    return trial


def test_qualify_harbor_rollouts_accepts_real_failed_solution(tmp_path: Path) -> None:
    _trial(tmp_path)
    report = qualify_harbor_rollouts(
        tmp_path,
        tmp_path / "index.jsonl",
        tmp_path / "report.json",
        run_id="agentic-1",
        policy_version=0,
        learner_version=0,
        expected_trials=1,
    )

    assert report["ok"] is True
    assert report["observed"]["rewards"] == [0.0]
    assert report["observed"]["bash_commands"] == 1


def test_qualify_harbor_rollouts_rejects_llm_text_without_terminal_action(tmp_path: Path) -> None:
    _trial(tmp_path, with_command=False)
    report = qualify_harbor_rollouts(
        tmp_path,
        tmp_path / "index.jsonl",
        tmp_path / "report.json",
        run_id="agentic-2",
        policy_version=0,
        learner_version=0,
        expected_trials=1,
    )

    assert report["ok"] is False
    assert report["trials"][0]["checks"]["real_terminal_action"] is False
