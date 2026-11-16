#!/usr/bin/env python3
"""Per-task oracle/nop summary for a LAION dataset re-validated on Leonardo.

A task is Leonardo-compatible only if all of its oracle trials finished without an exception with reward 1 and all of
its nop trials finished without an exception with reward 0, with at least --attempts trials in each arm. Tasks that
fail are listed with the reason; nothing is modified or replaced. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def collect(jobs: Path, arm: str) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((jobs / arm).rglob("result.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not {"task_name", "trial_name"} <= payload.keys():
            continue
        rewards = (payload.get("verifier_result") or {}).get("rewards") or {}
        # Harbor's task_name is the name in task.toml, which need not match the directory; the pack path does.
        task_path = (payload.get("task_id") or {}).get("path") if isinstance(payload.get("task_id"), dict) else None
        key = Path(task_path).name if task_path else str(payload["task_name"])
        by_task[key].append(
            {"reward": rewards.get("reward"), "exception": payload.get("exception_info")}
        )
    return by_task


def verdict(trials: list[dict], want: float, attempts: int) -> str | None:
    """None if the arm is clean, else a short reason."""
    if len(trials) < attempts:
        return f"only {len(trials)}/{attempts} trials"
    for t in trials:
        if t["exception"] is not None:
            return "exception"
        if t["reward"] is None or float(t["reward"]) != want:
            return f"reward={t['reward']!r}"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=Path, required=True, help="Harbor jobs dir containing oracle/ and nop/")
    p.add_argument("--tasks-dir", type=Path, required=True, help="task pack that was run")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--compatible-out", type=Path, required=True)
    p.add_argument("--report-out", type=Path, required=True)
    a = p.parse_args()

    expected = sorted(d.name for d in a.tasks_dir.iterdir() if d.is_dir() or d.is_symlink())
    oracle, nop = collect(a.jobs, "oracle"), collect(a.jobs, "nop")
    ok, failed = [], []
    for task in expected:
        o = verdict(oracle.get(task, []), 1.0, a.attempts)
        n = verdict(nop.get(task, []), 0.0, a.attempts)
        if o is None and n is None:
            ok.append(task)
        else:
            failed.append({"task_id": task, "oracle": o, "nop": n})

    a.compatible_out.write_text("".join(json.dumps({"task_id": t}) + "\n" for t in ok))
    report = {
        "expected_tasks": len(expected), "compatible": len(ok), "failed": len(failed),
        "attempts_per_arm": a.attempts, "failures": failed,
    }
    a.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
