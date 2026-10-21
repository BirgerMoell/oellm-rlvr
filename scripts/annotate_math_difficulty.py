#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oellm_rlvr.math_difficulty import annotate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate frozen math survey difficulty")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--expected-samples", type=int, default=8)
    args = parser.parse_args()
    report = annotate_predictions(args.predictions, args.output_csv, expected_samples=args.expected_samples)
    output = Path(args.output_csv).with_suffix(".summary.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
