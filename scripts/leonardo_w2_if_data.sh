#!/bin/bash
# W2 instruction-following pool (qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md): verifiable constraints (IFEval and
# IFBench-style: paragraphs, word counts, first/last words, forbidden words, case, JSON, response language, ...) from
# nvidia/Nemotron-RL-instruction_following @ bcdec57a (ODC-BY), 46,391 English prompts.
#   - 1,000 default-mode rows (`ifeval_direct`: constraints checked on the whole reply, no think tags; rl_mode=default)
#     - the mode EUROPA measures, and the one whose strict-format compliance is weak;
#   - 500 thinking-mode rows (TMAX `ifeval`: constraints checked on the text after the closed trace).
# Every kept row's constraints are built with TMAX's own registry (open_instruct.IFEvalG), so the verifier cannot fail
# at train time; prompts over MAX_PROMPT tokens (model template) are dropped. Score = fraction of constraints met.
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
MODEL=${MODEL:?model dir whose template renders the prompts (e.g. $ROOT/models/stage4-merge-a05-mixed)}
MAX_PROMPT=${MAX_PROMPT:-2048}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
SRC=$ROOT/data/sources/if-rl/Nemotron-RL-instruction_following/instruction_following.jsonl
OUT=$ROOT/data/w2-if
mkdir -p "$OUT"
cd "$ROOT/tmax-reproduction"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" .venv/bin/python - "$SRC" "$OUT" "$MODEL" "$MAX_PROMPT" <<'PY'
import collections, hashlib, json, random, sys
import pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoTokenizer
from open_instruct.IFEvalG import instructions_registry
src, out, model, max_prompt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(model)
registry = instructions_registry.INSTRUCTION_DICT
rng = random.Random(20261026)
rows = [json.loads(line) for line in open(src, encoding="utf-8")]
rng.shuffle(rows)
skipped = collections.Counter()
kept = []
for r in rows:
    ids, kwargs = r["instruction_id_list"], r["kwargs"]
    if not ids or len(ids) != len(kwargs):
        skipped["bad_constraints"] += 1
        continue
    if any(i not in registry for i in ids):
        skipped["unsupported_constraint"] += 1
        continue
    try:
        for key, kw in zip(ids, kwargs):
            registry[key](key).build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
    except Exception:
        skipped["build_error"] += 1
        continue
    prompt = r["prompt"]
    if "<think>" in prompt or "</think>" in prompt or "<tool_call>" in prompt:
        skipped["tag_in_prompt"] += 1
        continue
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    if len(tok(text, add_special_tokens=False)["input_ids"]) > max_prompt:
        skipped["too_long"] += 1
        continue
    kept.append(r)
    if len(kept) >= 1500:
        break
records = []
for k, r in enumerate(kept):
    label = json.dumps({"instruction_id": r["instruction_id_list"], "kwargs": r["kwargs"]}, ensure_ascii=False)
    default = k < 1000
    msg = {"role": "user", "content": r["prompt"], "rl_mode": "default" if default else None}
    records.append({"id": f"if-{r['id']}", "messages": [msg], "ground_truth": label,
                    "dataset": "ifeval_direct" if default else "ifeval"})
rng.shuffle(records)
schema = pa.schema([("id", pa.string()),
                    ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()), ("rl_mode", pa.string())]))),
                    ("ground_truth", pa.string()), ("dataset", pa.string())])
pq.write_table(pa.Table.from_pylist(records, schema=schema), f"{out}/train.parquet")
sha = hashlib.sha256(open(f"{out}/train.parquet", "rb").read()).hexdigest()
constraint_counts = collections.Counter(i for r in kept for i in r["instruction_id_list"])
manifest = {"kind": "w2_if_pool", "rows": len(records), "by_dataset": dict(collections.Counter(r["dataset"] for r in records)),
            "constraints_per_row": dict(collections.Counter(len(r["instruction_id_list"]) for r in kept)),
            "top_constraints": constraint_counts.most_common(15), "skipped": dict(skipped),
            "source": "nvidia/Nemotron-RL-instruction_following@bcdec57a62a1c817dfc7d1796efd6838b4a173e1 (ODC-BY)",
            "max_prompt_tokens": max_prompt, "sha256": sha}
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
PY
