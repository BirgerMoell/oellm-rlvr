#!/bin/bash
# W3 pool (qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md), exact counts, 2,560 prompts = 160 updates x 16:
#   512 eu_qa    - calibrated factual QA from Wikidata (oellm_rlvr.eu_tasks; +1 correct, 0 "ANSWER: UNKNOWN", -1 wrong),
#                  171/171/170 head/torso/tail, built locally into data/eu-tasks/w3-eu.parquet;
#   640 eu_doc   - document tasks (invoices, staff tables, contracts, room bookings, letter -> JSON), 128 per family,
#                  held-out question variants excluded; ~40 % of both eu slices in thinking mode;
#   512 math     - answer-language pool rows unused by W1a and W2 (EU `multilingual_math` check=answer, English `math`);
#   256 IF       - 192 `ifeval_direct` + 64 `ifeval_think` unused by W2;
#   256 code     - `code_stdio` rows unused by W1a and W2;
#   384 tools    - A4b English rows unused by W2 (prompt <= TOOL_MAX_PROMPT tokens).
# The replay share keeps W2's skills while the two new suites are learned. Every prompt must fit MAX_PROMPT tokens.
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
MODEL=${MODEL:?model dir with the RL mixed template (for token counts)}
MAX_PROMPT=${MAX_PROMPT:-3072}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
D=$ROOT/data; OUT=$D/w3
mkdir -p "$OUT"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" python - "$D" "$OUT" "$MODEL" "$MAX_PROMPT" <<'PY'
import collections, hashlib, json, random, sys
import pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoTokenizer
d, out, model, max_prompt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(model)
rng = random.Random(20261028)
fields = ["role", "content", "rl_mode"]
def norm(r, dataset=None):
    return {"id": r["id"], "messages": [{k: m.get(k) for k in fields} for m in r["messages"]],
            "ground_truth": r["ground_truth"], "dataset": dataset or r["dataset"]}
def rows(path):
    return pq.read_table(path).to_pylist()
def fits(r):
    text = tok.apply_chat_template([{k: v for k, v in m.items() if v is not None and k != "rl_mode"} for m in r["messages"]],
                                   tokenize=False, add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False)["input_ids"]) <= max_prompt
def take(pool, n, what):
    pool = sorted(pool, key=lambda r: r["id"]); rng.shuffle(pool)
    got = []
    for r in pool:
        if len(got) == n:
            break
        if fits(r):
            got.append(r)
    if len(got) < n:
        raise SystemExit(f"{what}: want {n}, have {len(got)}")
    return got
used = {r["id"] for r in rows(f"{d}/w1a/train-mixed.parquet")} | {r["id"] for r in rows(f"{d}/w2/train-mixed.parquet")}
picked = []
eu = rows(f"{d}/eu-tasks/w3-eu.parquet")
too_long = [r["id"] for r in eu if not fits(r)]
if too_long:
    raise SystemExit(f"eu rows over {max_prompt} tokens: {too_long[:5]}")
picked += [norm(r) for r in eu]
picked += [norm(r) for r in take([r for r in rows(f"{d}/answer-language/train-mixed.parquet") if r["id"] not in used], 512, "math")]
iff = [r for r in rows(f"{d}/w2-if/train.parquet") if r["id"] not in used]
picked += [norm(r) for r in take([r for r in iff if r["dataset"] == "ifeval_direct"], 192, "ifeval_direct")]
picked += [norm(r) for r in take([r for r in iff if r["dataset"] == "ifeval_think"], 64, "ifeval_think")]
picked += [norm(r) for r in take([r for r in rows(f"{d}/a5-code/train.parquet") if r["id"] not in used], 256, "code")]
picked += [norm(r) for r in take([r for r in rows(f"{d}/a4b-tool/train.parquet") if r["id"] not in used], 384, "tools")]
ids = [r["id"] for r in picked]
assert len(ids) == len(set(ids)) == 2560, (len(ids), len(set(ids)))
rng.shuffle(picked)
schema = pa.schema([("id", pa.string()),
                    ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()), ("rl_mode", pa.string())]))),
                    ("ground_truth", pa.string()), ("dataset", pa.string())])
pq.write_table(pa.Table.from_pylist(picked, schema=schema), f"{out}/train-mixed.parquet")
sha = hashlib.sha256(open(f"{out}/train-mixed.parquet", "rb").read()).hexdigest()
modes = collections.Counter("default" if r["messages"][-1]["rl_mode"] == "default" else "thinking" for r in picked)
manifest = {"kind": "w3_pool", "rows": len(picked), "by_dataset": dict(collections.Counter(r["dataset"] for r in picked)),
            "modes": dict(modes), "eu_slice_sha256": hashlib.sha256(open(f"{d}/eu-tasks/w3-eu.parquet", "rb").read()).hexdigest(),
            "max_prompt_tokens": max_prompt, "sha256": sha}
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
PY
