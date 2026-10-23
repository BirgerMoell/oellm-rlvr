#!/bin/bash
# Ablation A5 pool (qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md): single-turn code RL on birgermoell/oellm-code-rlvr
# @ e1cae771 (train split only; validation/test stay held out). 2,004 rows = 167 per generator family x 12 families,
# difficulty 1-4, seeded. Each row is scored by TMAX's `code_stdio` verifier against its mutation-checked hidden
# stdin/stdout tests (ground_truth unchanged); the prompt gains one sentence asking for a single ```python block.
# Needs the parquet downloaded to $ROOT/data/sources/oellm-code-rlvr (login node).
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
SRC=$ROOT/data/sources/oellm-code-rlvr/data/train-00000-of-00001.parquet
OUT=$ROOT/data/a5-code
mkdir -p "$OUT"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" python - "$SRC" "$OUT" <<'PY'
import collections, hashlib, json, random, sys
import pyarrow as pa, pyarrow.parquet as pq
src, out = sys.argv[1:]
SUFFIX = "\n\nPut the complete program in a single ```python code block."
rows = pq.read_table(src, columns=["id", "messages", "ground_truth", "generator_family", "difficulty",
                                   "semantic_group_id", "verifier_kind"]).to_pylist()
rng = random.Random(20261023)
by_family = collections.defaultdict(list)
for r in rows:
    if r["verifier_kind"] == "code_stdio" and 1 <= r["difficulty"] <= 4:
        by_family[r["generator_family"]].append(r)
picked, groups = [], set()
for family in sorted(by_family):
    pool = by_family[family]
    rng.shuffle(pool)
    n = 0
    for r in pool:
        if r["semantic_group_id"] in groups:
            continue
        groups.add(r["semantic_group_id"])
        picked.append(r)
        n += 1
        if n == 167:
            break
rng.shuffle(picked)
records = []
for r in picked:
    messages = [dict(m) for m in r["messages"]]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    messages[0]["content"] = messages[0]["content"].rstrip() + SUFFIX
    json.loads(r["ground_truth"])  # must parse
    records.append({"id": r["id"], "messages": messages, "ground_truth": r["ground_truth"], "dataset": "code_stdio"})
table = pa.Table.from_pylist(records)
pq.write_table(table, f"{out}/train.parquet")
sha = hashlib.sha256(open(f"{out}/train.parquet", "rb").read()).hexdigest()
manifest = {"kind": "a5_code_pool", "source": "birgermoell/oellm-code-rlvr@e1cae7711049e3b5ff021fb3e9c752424882998c train",
            "rows": len(records), "families": dict(collections.Counter(r["generator_family"] for r in picked)),
            "difficulty": dict(sorted(collections.Counter(r["difficulty"] for r in picked).items())), "sha256": sha}
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
PY
