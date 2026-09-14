#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from oellm_rlvr.datasets import write_rows
from oellm_rlvr.multilingual_reasoning import parse_language_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prefix_rows(path: Path, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(path).iter_batches(columns=["messages"], batch_size=min(1024, count)):
        rows.extend(batch.to_pylist())
        if len(rows) >= count:
            return rows[:count]
    if len(rows) < count:
        raise ValueError(f"{path} contains only {len(rows)} rows; requested {count}")
    return rows


def _valid_messages(messages: object, *, require_think: bool) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    if messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
        return False
    assistant = str(messages[-1].get("content") or "")
    if require_think:
        return (
            assistant.casefold().count("<think>") == 1
            and assistant.casefold().count("</think>") == 1
            and "\\boxed{" in assistant
        )
    return bool(str(messages[0].get("content") or "").strip() and assistant.strip())


def build_bridge(
    *,
    multilingual: Path,
    english_reasoning: Path,
    general_replay: Path,
    language_contract: Path,
    output: Path,
    replay_rows: int,
    multilingual_revision: str,
    contract_revision: str,
) -> dict[str, Any]:
    target_alpha2 = {target.alpha2 for target in parse_language_contract(language_contract)} - {"en"}
    translated = pq.read_table(multilingual).to_pylist()
    rows: list[dict[str, Any]] = []
    dropped = Counter()
    for row in translated:
        language = str(row.get("language") or "")
        quality = row.get("quality") or {}
        if language not in target_alpha2:
            dropped["outside_target_contract"] += 1
            continue
        if not bool(quality.get("accepted")):
            dropped["quality_rejected"] += 1
            continue
        messages = row.get("messages")
        if not _valid_messages(messages, require_think=True):
            dropped["invalid_think_box_contract"] += 1
            continue
        rows.append(
            {
                "messages": messages,
                "bridge_role": "multilingual_reasoning",
                "language": language,
                "source_id": str(row.get("source_id") or ""),
                "source_license": str(row.get("source_license") or ""),
            }
        )

    for role, source, require_think in (
        ("english_reasoning_replay", english_reasoning, True),
        ("general_instruction_replay", general_replay, False),
    ):
        accepted = 0
        for row in _prefix_rows(source, replay_rows * 2):
            messages = row.get("messages")
            if not _valid_messages(messages, require_think=require_think):
                continue
            rows.append(
                {
                    "messages": messages,
                    "bridge_role": role,
                    "language": "en" if require_think else "unknown",
                    "source_id": "",
                    "source_license": "see source manifest",
                }
            )
            accepted += 1
            if accepted == replay_rows:
                break
        if accepted != replay_rows:
            raise ValueError(f"only found {accepted}/{replay_rows} valid rows for {role}")

    destination = output / "train.parquet"
    output.mkdir(parents=True, exist_ok=True)
    write_rows(rows, destination)
    counts = Counter(str(row["bridge_role"]) for row in rows)
    languages = Counter(str(row["language"]) for row in rows if row["bridge_role"] == "multilingual_reasoning")
    manifest = {
        "schema": "oellm-multilingual-reasoning-sft-bridge-v1",
        "purpose": "control-only interface bridge before language-gated RLVR",
        "sources": {
            "multilingual": {
                "path": str(multilingual),
                "revision": multilingual_revision,
                "sha256": _sha256(multilingual),
            },
            "english_reasoning_replay": {"path": str(english_reasoning), "sha256": _sha256(english_reasoning)},
            "general_instruction_replay": {"path": str(general_replay), "sha256": _sha256(general_replay)},
            "language_contract": {
                "path": str(language_contract),
                "revision": contract_revision,
                "sha256": _sha256(language_contract),
            },
        },
        "rows": len(rows),
        "rows_by_role": dict(sorted(counts.items())),
        "multilingual_rows_by_language": dict(sorted(languages.items())),
        "dropped": dict(sorted(dropped.items())),
        "artifact": {"path": str(destination), "sha256": _sha256(destination)},
        "limitations": [
            "The translated pilot contains only 99 source problems and is an interface bridge, not a production corpus.",
            "Georgian is absent from the translated source and requires new verified traces.",
            "Mixed upstream terms require review before publishing derived checkpoint weights.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Qwen multilingual reasoning interface bridge.")
    parser.add_argument("--multilingual", required=True)
    parser.add_argument("--english-reasoning", required=True)
    parser.add_argument("--general-replay", required=True)
    parser.add_argument("--language-contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay-rows", type=int, default=384)
    parser.add_argument("--multilingual-revision", required=True)
    parser.add_argument("--contract-revision", required=True)
    args = parser.parse_args()
    if args.replay_rows < 1:
        parser.error("--replay-rows must be positive")
    manifest = build_bridge(
        multilingual=Path(args.multilingual),
        english_reasoning=Path(args.english_reasoning),
        general_replay=Path(args.general_replay),
        language_contract=Path(args.language_contract),
        output=Path(args.output),
        replay_rows=args.replay_rows,
        multilingual_revision=args.multilingual_revision,
        contract_revision=args.contract_revision,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
