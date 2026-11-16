#!/bin/bash
# Natural-question hallucination benchmark (github.com/BirgerMoell/oellm-hallucination) on Leonardo, one 2-GPU node:
# closed-book answers per model (GPU 0, one after the other, default + thinking), then one gemma-4-31B-it judge load (TP=2), then the report.
# Same container, Python and judge as the baseline run in that repo, so the numbers are comparable with its results table.
#
#   git clone https://github.com/BirgerMoell/oellm-hallucination $SCRATCH/halu-bench-<tag>
#   BENCH=$SCRATCH/halu-bench-<tag> MODELS="e8-step160=$SCRATCH/oellm-rl/models/oellm9b-e8-step160" sbatch scripts/leonardo_halu_bench.sbatch
# MODELS: "label=/path/to/model ..."   NO_THINKING: labels whose chat template has no enable_thinking switch.
#SBATCH --job-name=halu-bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
set -euo pipefail
: "${BENCH:?clone the benchmark repo and set BENCH=<dir>}" "${MODELS:?label=/model/dir ...}"
: "${SCRATCH:?}"
: "${CONTAINER:=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif}"
: "${PY:=$SCRATCH/oellm-rl/verl/.venv/bin/python}"
: "${JUDGE:=/leonardo_work/OELLM_prod2026/models/google/gemma-4-31B-it}"
OUT="$BENCH/runs"
mkdir -p "$OUT" logs
py() { singularity exec --nv -B /leonardo_work,/leonardo_scratch "$CONTAINER" env HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 "$PY" "$@"; }

for spec in $MODELS; do
  name=${spec%%=*}; model=${spec#*=}
  modes="default thinking"; [[ " ${NO_THINKING:-} " == *" $name "* ]] && modes="default"
  [[ -s $OUT/$name/answers.jsonl ]] && { echo "skip $name (answers exist)"; continue; }
  mkdir -p "$OUT/$name"
  CUDA_VISIBLE_DEVICES=0 py "$BENCH/benchmark/generate.py" --questions "$BENCH/data/questions.jsonl" --model "$model" \
    --modes $modes --out "$OUT/$name/answers.jsonl"
done

# one judge load for the models of this job: tag rows with the run name, judge, split back (only this job's models)
names=$(for s in $MODELS; do echo "${s%%=*}"; done | tr '\n' ' ')
py - "$OUT" $names <<'PYEOF'
import json, sys
out, names = sys.argv[1], sys.argv[2:]
with open(f"{out}/all_answers.jsonl", "w") as f:
    for n in names:
        for line in open(f"{out}/{n}/answers.jsonl"):
            f.write(json.dumps({**json.loads(line), "run": n}, ensure_ascii=False) + "\n")
PYEOF
py "$BENCH/benchmark/judge.py" --answers "$OUT/all_answers.jsonl" --judge "$JUDGE" --tp 2 --out "$OUT/all_judged.jsonl"
py - "$OUT" <<'PYEOF'
import json, sys, collections
out = sys.argv[1]
by = collections.defaultdict(list)
for line in open(f"{out}/all_judged.jsonl"):
    by[json.loads(line)["run"]].append(line)
for name, lines in by.items():
    open(f"{out}/{name}/judged.jsonl", "w").writelines(lines)
PYEOF
py "$BENCH/benchmark/score.py" $(for n in $names; do echo "$OUT/$n/judged.jsonl"; done) --by-language --json "$OUT/report.json"
echo "halu_bench_artifact=$OUT"
