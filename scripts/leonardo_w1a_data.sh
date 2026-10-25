#!/bin/bash
# W1a pool (qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md): the first combined RL window, from the Stage-3 <-> DPO-v2
# merge (alpha 0.5). It joins what the ablations adopted, with exact counts (2,048 prompts = 128 updates x 16):
#   1,100 thinking-mode math from data/answer-language (CHECK=answer pool: 560 EU-language rows scored by
#         `multilingual_math` with check=answer - correct + final answer in the prompt language, trace any language -
#         and 540 English/GSM8K rows scored by `math`);
#     600 default-mode rows from data/a3-mixed (300 `math_direct`, 300 `mcq_letter_direct`: bare correct answer only);
#     348 thinking-mode code rows from data/a5-code (`code_stdio`, contained execution server).
# Run after leonardo_a1_language_data.sh (CHECK=answer), leonardo_a3_mixed_mode_data.sh, leonardo_a5_code_data.sh.
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
D=$ROOT/data; OUT=$D/w1a
mkdir -p "$OUT"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" python - "$D" "$OUT" <<'PY'
import collections, hashlib, json, random, sys
import pyarrow as pa, pyarrow.parquet as pq
d, out = sys.argv[1:]
rng = random.Random(20261025)
a3 = pq.read_table(f"{d}/a3-mixed/train-mixed.parquet")
schema = a3.schema  # messages may carry rl_mode on the last turn
def rows(path):
    return pq.read_table(path).to_pylist()
def take(pool, n, what):
    pool = sorted(pool, key=lambda r: r["id"]); rng.shuffle(pool)
    if len(pool) < n:
        raise SystemExit(f"{what}: want {n}, have {len(pool)}")
    return pool[:n]
lang = rows(f"{d}/answer-language/train-mixed.parquet")
for r in lang:
    if r["dataset"] == "multilingual_math":
        assert json.loads(r["ground_truth"]).get("check") == "answer", r["id"]
default = [r for r in a3.to_pylist() if r["dataset"] in ("math_direct", "mcq_letter_direct")]
code = rows(f"{d}/a5-code/train.parquet")
picked = (take([r for r in lang if r["dataset"] == "multilingual_math"], 560, "eu")
          + take([r for r in lang if r["dataset"] == "math"], 540, "en/gsm8k")
          + take([r for r in default if r["dataset"] == "math_direct"], 300, "math_direct")
          + take([r for r in default if r["dataset"] == "mcq_letter_direct"], 300, "mcq_letter_direct")
          + take(code, 348, "code"))
ids = [r["id"] for r in picked]
assert len(set(ids)) == len(ids) == 2048, (len(ids), len(set(ids)))
rng.shuffle(picked)
fields = {f.name for f in schema.field("messages").type.value_type}
for r in picked:
    r["messages"] = [{k: m.get(k) for k in fields} for m in r["messages"]]
pq.write_table(pa.Table.from_pylist(picked, schema=schema), f"{out}/train-mixed.parquet")
sha = hashlib.sha256(open(f"{out}/train-mixed.parquet", "rb").read()).hexdigest()
manifest = {"kind": "w1a_pool", "rows": len(picked), "by_dataset": dict(collections.Counter(r["dataset"] for r in picked)),
            "sources": {"thinking_math": f"{d}/answer-language", "default_mode": f"{d}/a3-mixed", "code": f"{d}/a5-code"},
            "default_rows_with_rl_mode": sum(1 for r in picked if r["messages"][-1].get("rl_mode") == "default"),
            "sha256": sha}
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
PY
