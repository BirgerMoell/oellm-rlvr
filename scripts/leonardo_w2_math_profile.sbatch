#!/bin/bash
#SBATCH --job-name=w2-math-profile
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# W2 harder-math pool by difficulty profiling (qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md; RL plan "profile the
# current checkpoint, keep informative prompts"). Candidates: birgermoell/oellm-math-rlvr @ 0ffc9d6d train, difficulty
# 3-5, 40 per non-English language + 240 English, excluding every prompt already in an RL pool (think-probe,
# answer-language, A3, W1a), the format-compliance pairs and the quick-eval items. Each is sampled 8 x in thinking mode
# (temperature 1.0, 8,192 new tokens) on MODEL (with the RL thinking-default template) by oellm_rlvr eval-reasoning,
# 4 GPUs in parallel; prompts are then banded (unsolved / hard / medium / easy / solved at 8).
#   MODEL=<dir> [TEMPLATE=<rl mixed jinja>] sbatch scripts/leonardo_w2_math_profile.sbatch   (submit from $SCRATCH/oellm-rl/runs)
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
ROOT=$OELLM_SCRATCH/oellm-rl
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
: "${MODEL:?}"
TEMPLATE=${TEMPLATE:-$OELLM_WORK/sft-migration-preflight/oellm_qwen3_two_stage_v2_1_rl_mixed.jinja}
SRC=$OELLM_SCRATCH/oellm-dpo/hf-cache/hub/datasets--birgermoell--oellm-math-rlvr/snapshots/0ffc9d6dc82717c25733b3172f4dbd63e48bab68/data/train-00000-of-00001.parquet
QUICK=$OELLM_WORK/sft-migration-preflight/data/eval-quick/reasoning_quick.jsonl
OUT=$ROOT/data/w2-math-profile; mkdir -p "$OUT"
# Thinking-default copy of the model (eval-reasoning renders with the model's own template).
M=$OUT/model; mkdir -p "$M"
for f in "$MODEL"/*; do case "$(basename "$f")" in chat_template.jinja) ;; *) ln -f "$f" "$M/" ;; esac; done
cp "$TEMPLATE" "$M/chat_template.jinja"
PY=$ROOT/tmax-reproduction/.venv/bin/python
run() { singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env PYTHONPATH=$ROOT/oellm-rlvr/src HF_HUB_OFFLINE=1 \
          VLLM_NO_USAGE_STATS=1 OELLM_PATCH_MATH_EQUIV_THREADS=1 "$PY" "$@"; }
if [[ ! -s $OUT/candidates-3.parquet ]]; then
run - "$SRC" "$QUICK" "$ROOT/data" "$OUT" "$OELLM_SCRATCH/oellm-dpo/format-v1/prompts.jsonl" <<'PY'
import collections, glob, json, random, sys
import pyarrow as pa, pyarrow.parquet as pq
src, quick, data, out, fmt = sys.argv[1:]
used_ids, used_text = set(), set()
for path in glob.glob(f"{data}/*/train-mixed.parquet") + glob.glob(f"{data}/*/train.parquet"):
    t = pq.read_table(path, columns=["id", "messages"]).to_pylist()
    for r in t:
        used_ids.add(r["id"]); used_text.add(r["messages"][-1]["content"].strip())
for line in open(quick, encoding="utf-8"):
    used_text.add(json.loads(line)["prompt"].strip())
try:
    for line in open(fmt, encoding="utf-8"):
        r = json.loads(line); used_text.add(r["prompt"].strip()); used_ids.add(r["id"].replace("fmt-math-", ""))
except FileNotFoundError:
    pass
table = pq.read_table(src, columns=["id", "messages", "ground_truth", "language", "difficulty"]).to_pylist()
by_lang = collections.defaultdict(list)
for r in table:
    msgs = r["messages"] if isinstance(r["messages"], list) else json.loads(r["messages"])
    text = msgs[-1]["content"].strip()
    if r["difficulty"] < 3 or r["id"] in used_ids or text in used_text:
        continue
    gt = r["ground_truth"]
    gt = gt[0] if isinstance(gt, (list, tuple)) and len(gt) == 1 else gt
    if isinstance(gt, (list, tuple)):
        continue
    by_lang[r["language"]].append({"id": r["id"], "messages": [{"role": "user", "content": text}], "ground_truth": str(gt),
                                   "language": r["language"], "difficulty": int(r["difficulty"])})
rng = random.Random(20261026)
picked = []
for lang, rows in sorted(by_lang.items()):
    rows.sort(key=lambda r: r["id"]); rng.shuffle(rows)
    picked += rows[: 240 if lang == "en" else 40]
rng.shuffle(picked)
for k in range(4):
    pq.write_table(pa.Table.from_pylist(picked[k::4]), f"{out}/candidates-{k}.parquet")
json.dump({"candidates": len(picked), "languages": dict(collections.Counter(r["language"] for r in picked)),
           "difficulty": dict(collections.Counter(r["difficulty"] for r in picked)), "excluded_ids": len(used_ids),
           "excluded_texts": len(used_text), "source": "birgermoell/oellm-math-rlvr@0ffc9d6d train"},
          open(f"{out}/candidates.json", "w"), indent=1)
print(open(f"{out}/candidates.json").read())
PY
fi
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g run -m oellm_rlvr.cli eval-reasoning --model "$M" --dataset "$OUT/candidates-$g.parquet" \
    --output "$OUT/predictions-$g.jsonl" --samples-per-prompt 8 --temperature 1.0 --max-new-tokens 8192 \
    --max-model-len 9216 --batch-size 32 --gpu-memory-utilization 0.85 --seed 2026102$g > "$OUT/sample-$g.log" 2>&1 &
done
wait
run - "$OUT" <<'PY'
import collections, glob, json, sys
out = sys.argv[1]
by = collections.defaultdict(list); meta = {}
for path in glob.glob(f"{out}/predictions-*.jsonl"):
    for line in open(path):
        r = json.loads(line); by[r["id"]].append(bool(r["analysis"].get("correct"))); meta[r["id"]] = r["metadata"]
def band(c, n):
    return "unsolved" if c == 0 else "solved" if c == n else "hard" if c / n < 0.375 else "medium" if c / n <= 0.625 else "easy"
rows = [{"id": i, "correct": sum(v), "attempts": len(v), "band": band(sum(v), len(v)), **meta[i]} for i, v in by.items()]
json.dump(rows, open(f"{out}/profile.json", "w"))
summary = collections.Counter(r["band"] for r in rows)
by_lang = collections.defaultdict(collections.Counter)
for r in rows:
    by_lang[r.get("language")][r["band"]] += 1
print(json.dumps({"prompts": len(rows), "bands": summary, "by_language": by_lang}, indent=1, default=dict))
PY
