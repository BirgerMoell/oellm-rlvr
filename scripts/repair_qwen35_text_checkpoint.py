#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from oellm_rlvr.checkpoint_repair import repair_qwen35_text_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair VLM-prefixed weights in a text-only Qwen3.5 checkpoint export."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-shard-gib", type=float, default=4.0)
    args = parser.parse_args()
    manifest = repair_qwen35_text_checkpoint(
        args.source,
        args.output,
        max_shard_bytes=int(args.max_shard_gib * 1024**3),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
