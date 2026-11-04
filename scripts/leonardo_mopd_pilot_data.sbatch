#!/bin/bash
#SBATCH --job-name=mopd-pilot-data
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%j.out
# MOPD pilot prompt shards (qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md §10.7, §11), one per teacher, routed by
# data_source. The KL probe (eval/mopd-kl/e8) found every teacher same-origin (per-token KL 0.001-0.012) except on
# default-mode factual QA, where W3 (0.34) and 4b (0.16) diverge, so factual QA is not distilled at all.
#   stage4b: 1,300 safety-v1 + 700 identity-v2 prompts      -> 4b (EUROPA Alignment 72.0, identity)
#   merge:   2,000 multilingual instruct-SFT user turns    -> merge 0.5 (EUROPA Eval 72.2)
#   w3:      1,500 eu_doc (unused by E8) + 500 code_stdio  -> W3 (eu_doc 0.516)
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
W=${W:?work dir with data/openeurollm-instruct-sft-v1}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
PY=$ROOT/tmax-reproduction/.venv/bin/python
OUT=$ROOT/data/mopd-pilot
mkdir -p "$OUT"
run() { singularity exec -B /leonardo_work,/leonardo_scratch "$C" env PYTHONPATH=$ROOT/oellm-rlvr/src "$PY" "$@"; }
run - "$ROOT" "$W" "$OUT" <<'PY'
import glob, json, random, sys
import pyarrow.parquet as pq
root, w, out = sys.argv[1:4]
rng = random.Random(20261105)
def user(text):
    return {"messages": [{"role": "user", "content": text}]}
def dump(name, rows):
    with open(f"{out}/{name}-prompts.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"{name}-{i}", **r}, ensure_ascii=False) + "\n")
    print(name, len(rows))
safety = [json.loads(l)["prompt"] for l in open(f"{root}/../oellm-dpo/safety-v1/prompts.jsonl", encoding="utf-8")]
identity = [json.loads(l)["prompt"] for l in open(f"{root}/../oellm-dpo/identity-v2/prompts.jsonl", encoding="utf-8")]
rng.shuffle(safety); rng.shuffle(identity)
dump("stage4b", [user(p) for p in safety[:1300] + identity[:700]])
shards = sorted(glob.glob(f"{w}/data/openeurollm-instruct-sft-v1/data/*.parquet"))
rng.shuffle(shards)
general, seen = [], set()
for path in shards[:80]:
    rows = pq.read_table(path, columns=["messages"]).to_pylist()
    rng.shuffle(rows)
    for r in rows[:30]:
        first = next((m["content"] for m in r["messages"] if m["role"] == "user"), None)
        if first and 20 < len(first) < 2500 and first not in seen:
            seen.add(first); general.append(user(first))
rng.shuffle(general)
dump("merge", general[:2000])
used = {r["id"] for r in pq.read_table(f"{root}/data/e8/train-mixed.parquet", columns=["id"]).to_pylist()}
docs = [r for r in pq.read_table(f"{root}/data/eu-tasks/rl_doc.parquet").to_pylist() if r["id"] not in used]
code = [r for r in pq.read_table(f"{root}/data/a5-code/train.parquet").to_pylist() if r["id"] not in used]
rng.shuffle(docs); rng.shuffle(code)
def first_user(r):
    return next(m["content"] for m in r["messages"] if m["role"] == "user")
dump("w3", [user(first_user(r)) for r in docs[:1500] + code[:500]])
PY
for key in stage4b merge w3; do
  run -m oellm_rlvr.cli prepare-opd-data --source "$OUT/$key-prompts.jsonl" --output "$OUT/$key.parquet" --data-source "$key"
done
ls -la "$OUT"
