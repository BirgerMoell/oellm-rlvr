#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from oellm_rlvr.multilingual_reasoning import build_multilingual_reasoning_canary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical-language symbolic reasoning canary.")
    parser.add_argument("--language-contract", required=True)
    parser.add_argument("--reference-traces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contract-revision", required=True)
    parser.add_argument("--reference-revision", required=True)
    parser.add_argument("--train-per-language", type=int, default=64)
    parser.add_argument("--eval-per-language", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    manifest = build_multilingual_reasoning_canary(
        language_contract=args.language_contract,
        reference_traces=args.reference_traces,
        output_dir=args.output,
        contract_revision=args.contract_revision,
        reference_revision=args.reference_revision,
        train_per_language=args.train_per_language,
        eval_per_language=args.eval_per_language,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
