#!/bin/bash
# Ablation A1 pool (docs: qwen35-posttrain OELLM_RL_RUN_PROGRAM.md): the RL-0 think-probe pool with ONE change.
# The 1,008 EU-language oellm-math-rlvr rows are re-labelled for the `multilingual_math` verifier
# (reward 1 only if the answer is correct AND the trace is in the prompt language AND the think/box form is valid);
# English and GSM8K rows, row order and prompts are byte-identical to train-mixed.parquet.
# Run after leonardo_think_probe_data.sh (needs its parts/ and train-mixed.parquet); no network needed.
# CHECK=answer (main line, decided 2026-10-23) labels the rows so that only the final answer after </think> must be
# in the prompt language and the trace may be in any language; output goes to data/answer-language.
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
CHECK=${CHECK:-trace}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
PROBE=$ROOT/data/think-probe
case "$CHECK" in trace) OUT=$ROOT/data/a1-language;; answer) OUT=$ROOT/data/answer-language;; *) echo "CHECK must be trace or answer" >&2; exit 2;; esac
mkdir -p "$OUT"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" python - "$PROBE" "$OUT" "$CHECK" <<'PY'
import glob, hashlib, json, os, sys, collections
import pyarrow as pa, pyarrow.parquet as pq
probe, out, check = sys.argv[1:]
language = {}
for path in sorted(glob.glob(f"{probe}/parts/eu-*.parquet")):
    lang = os.path.basename(path)[3:-8]
    for i in pq.read_table(path, columns=["id"])["id"].to_pylist():
        language[i] = lang
table = pq.read_table(f"{probe}/train-mixed.parquet")
rows = table.to_pylist()
counts = collections.Counter()
for row in rows:
    lang = language.get(row["id"])
    if lang is None:
        counts[row["dataset"]] += 1
        continue
    assert row["dataset"] == "math", row
    contract = {"answer": row["ground_truth"], "target_language": lang}
    if check != "trace":  # keep the A1 pool byte-identical
        contract["check"] = check
    row["ground_truth"] = json.dumps(contract, ensure_ascii=False)
    row["dataset"] = "multilingual_math"
    counts[f"multilingual_math:{lang}"] += 1
assert sum(v for k, v in counts.items() if k.startswith("multilingual_math")) == len(language) == 1008, counts
relabelled = pa.Table.from_pylist(rows, schema=table.schema)
pq.write_table(relabelled, f"{out}/train-mixed.parquet")
sha = hashlib.sha256(open(f"{out}/train-mixed.parquet", "rb").read()).hexdigest()
manifest = {"kind": "a1_language_pool" if check == "trace" else f"{check}_language_pool", "check": check, "source": f"{probe}/train-mixed.parquet", "rows": len(rows),
            "counts": dict(sorted(counts.items())), "sha256": sha}
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
PY
