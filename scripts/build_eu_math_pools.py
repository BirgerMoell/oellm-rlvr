#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from oellm_rlvr.eu_math import EU24_NON_ENGLISH, build_eu_math_pools


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced EU and disjoint English math RLVR pools")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eu-per-language", type=int, default=128)
    parser.add_argument("--english-count", type=int, default=2048)
    parser.add_argument("--min-difficulty", type=int, default=3)
    parser.add_argument("--max-difficulty", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--languages", default=",".join(EU24_NON_ENGLISH))
    args = parser.parse_args()
    languages = tuple(value.strip() for value in args.languages.split(",") if value.strip())
    report = build_eu_math_pools(
        args.source,
        args.output_dir,
        eu_per_language=args.eu_per_language,
        english_count=args.english_count,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        seed=args.seed,
        languages=languages,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
