#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oellm_rlvr.harbor_results import validate_trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--reward", required=True, type=float)
    parser.add_argument("--tasks", type=int, default=16)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = validate_trials(
        args.root,
        expected_agent=args.agent,
        expected_reward=args.reward,
        expected_tasks=args.tasks,
        expected_attempts=args.attempts,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
