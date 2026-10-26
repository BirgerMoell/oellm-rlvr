#!/bin/bash
# W2 core-window pool (qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md), exact counts, 2,560 prompts = 160 updates x 16:
#   896 thinking-mode math   - profiled on the W2 base (scripts/leonardo_w2_math_profile.sbatch): only prompts solved
#                              1-7 of 8 times (hard first, then medium, then easy); EU languages scored by
#                              `multilingual_math` with check=answer, English by `math`. Topped up from the
#                              answer-language pool if the profile yields fewer;
#   384 default-mode bare    - 192 `math_direct_shaped` (graded credit, bare = 1) + 192 `mcq_letter_direct` (A3 pool);
#   384 instruction following - 256 `ifeval_direct` (default mode) + 128 `ifeval_think` (data/w2-if);
#   384 code                 - `code_stdio` rows from data/a5-code not used in W1a;
#   512 tool calls           - multilingual slice (data/a4b-tool-translated) first, then English A4b rows.
# Tool prompts are capped at TOOL_MAX_PROMPT tokens (default 3,072) to bound the pack length.
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
MODEL=${MODEL:?W2 base model dir with the RL mixed template (for token counts)}
TOOL_MAX_PROMPT=${TOOL_MAX_PROMPT:-3072}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
D=$ROOT/data; OUT=$D/w2
mkdir -p "$OUT"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" python - "$D" "$OUT" "$MODEL" "$TOOL_MAX_PROMPT" <<'PY'
import collections, glob, hashlib, json, random, sys
import pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoTokenizer
d, out, model, tool_max = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(model)
rng = random.Random(20261027)
fields = ["role", "content", "rl_mode"]
def norm(r, dataset=None, ground_truth=None):
    msgs = [{k: m.get(k) for k in fields} for m in r["messages"]]
    return {"id": r["id"], "messages": msgs, "ground_truth": ground_truth if ground_truth is not None else r["ground_truth"],
            "dataset": dataset or r["dataset"]}
def rows(path):
    return pq.read_table(path).to_pylist()
def take(pool, n, what, strict=True):
    pool = sorted(pool, key=lambda r: r["id"]); rng.shuffle(pool)
    if strict and len(pool) < n:
        raise SystemExit(f"{what}: want {n}, have {len(pool)}")
    return pool[:n]
w1a_ids = {r["id"] for r in rows(f"{d}/w1a/train-mixed.parquet")}
picked, notes = [], {}

# Thinking math from the profile.
try:
    profile = {r["id"]: r for r in json.load(open(f"{d}/w2-math-profile/profile.json"))}
except FileNotFoundError:  # profiling did not run: all math comes from the filler below
    profile = {}
    notes["profile_missing"] = True
cands = {r["id"]: r for p in glob.glob(f"{d}/w2-math-profile/candidates-*.parquet") for r in rows(p)}
order = {"hard": 0, "medium": 1, "easy": 2}
informative = [cands[i] for i, p in profile.items() if p["band"] in order and i in cands]
informative.sort(key=lambda r: (order[profile[r["id"]]["band"]], hashlib.sha256(r["id"].encode()).hexdigest()))
math = []
for r in informative[:896]:
    if r["language"] == "en":
        math.append(norm(r, "math", r["ground_truth"]))
    else:
        label = json.dumps({"answer": r["ground_truth"], "target_language": r["language"], "check": "answer"}, ensure_ascii=False)
        math.append(norm(r, "multilingual_math", label))
notes["math_from_profile"] = len(math)
if len(math) < 896:
    filler = [r for r in rows(f"{d}/answer-language/train-mixed.parquet") if r["id"] not in w1a_ids]
    math += [norm(r) for r in take(filler, 896 - len(math), "math filler")]
picked += math
# Default-mode bare answers.
a3 = rows(f"{d}/a3-mixed/train-mixed.parquet")
direct = [r for r in a3 if r["dataset"] == "math_direct"]
fresh = take([r for r in direct if r["id"] not in w1a_ids], 192, "math_direct fresh", strict=False)
reused = take([r for r in direct if r["id"] in w1a_ids], 192 - len(fresh), "math_direct reused")  # W1a never scored these (0 bare)
picked += [norm(r, "math_direct_shaped") for r in fresh + reused]
notes["math_direct_fresh"] = len(fresh)
picked += [norm(r) for r in take([r for r in a3 if r["dataset"] == "mcq_letter_direct"], 192, "mcq")]
# Instruction following.
iff = rows(f"{d}/w2-if/train.parquet")
picked += [norm(r) for r in take([r for r in iff if r["dataset"] == "ifeval_direct"], 256, "ifeval_direct")]
picked += [norm(r) for r in take([r for r in iff if r["dataset"] == "ifeval_think"], 128, "ifeval_think")]
# Code.
code = rows(f"{d}/a5-code/train.parquet")
picked += [norm(r) for r in take([r for r in code if r["id"] not in w1a_ids], 384, "code")]
# Tools: translated slice first.
def fits(r):
    text = tok.apply_chat_template([{k: v for k, v in m.items() if v is not None} for m in r["messages"]], tokenize=False, add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False)["input_ids"]) <= tool_max
translated = [r for r in rows(f"{d}/a4b-tool-translated/train.parquet") if fits(r)] if glob.glob(f"{d}/a4b-tool-translated/train.parquet") else []
tools = take(translated, min(len(translated), 192), "translated", strict=False)
source_ids = {r["id"].rsplit("-", 1)[0] for r in tools}
english = [r for r in rows(f"{d}/a4b-tool/train.parquet") if r["id"] not in source_ids and fits(r)]
tools += take(english, 512 - len(tools), "tools")
notes["tools_translated"] = sum(1 for r in tools if r["id"].rsplit("-", 1)[-1] in {"cs","de","el","es","fi","fr","it","nl","pl","ro","sv","uk","da","no","is","pt"})
picked += [norm(r) for r in tools]
ids = [r["id"] for r in picked]
assert len(ids) == len(set(ids)) == 2560, (len(ids), len(set(ids)))
rng.shuffle(picked)
schema = pa.schema([("id", pa.string()),
                    ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()), ("rl_mode", pa.string())]))),
                    ("ground_truth", pa.string()), ("dataset", pa.string())])
pq.write_table(pa.Table.from_pylist(picked, schema=schema), f"{out}/train-mixed.parquet")
sha = hashlib.sha256(open(f"{out}/train-mixed.parquet", "rb").read()).hexdigest()
manifest = {"kind": "w2_pool", "rows": len(picked), "by_dataset": dict(collections.Counter(r["dataset"] for r in picked)),
            "notes": notes, "profile_bands": dict(collections.Counter(p["band"] for p in profile.values())),
            "tool_max_prompt_tokens": tool_max, "sha256": sha}
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
PY
