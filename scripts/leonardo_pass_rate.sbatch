#!/bin/bash
#SBATCH --job-name=pass-rate
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Pass-rate filter of an RL pool on the checkpoint it will train (src/oellm_rlvr/pass_rate.py; improvement loop in
# qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md). Submit from $SCRATCH/oellm-rl/runs.
#   POOL=<training parquet> MODEL=<staged model dir with the RL template> OUT=<work dir> [SAMPLES=8]
#   STAGE=sample  sbatch --array=0-3 scripts/leonardo_pass_rate.sbatch     one shard per 1-GPU task (backfills)
#   STAGE=score   sbatch --cpus-per-task=32 --time=01:00:00 (1 GPU node slot; the code API is started here when the
#                 pool has code rows, as in training)
#   STAGE=select  (CPU, seconds; SELECT_ARGS e.g. "--min-solve 0.125 --cap eu_doc=300")
# MAX_NEW_TOKENS should match the training config's response length; LIMIT samples only the first N per shard (smoke).
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
ROOT=$OELLM_SCRATCH/oellm-rl
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
PY=$ROOT/tmax-reproduction/.venv/bin/python
: "${POOL:?}" "${MODEL:?}" "${OUT:?}" "${STAGE:?sample|score|select}"
SAMPLES=${SAMPLES:-8}
SHARDS=${SHARDS:-4}
mkdir -p "$OUT"
# Same verifier patches as the training configs (leonardo-grpo-math-oellm9b-w3.yaml `environment`).
run() { singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env PYTHONPATH=$ROOT/oellm-rlvr/src:$ROOT/tmax-reproduction \
          HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=fork OELLM_PATCH_MATH_EQUIV_THREADS=1 \
          OELLM_PATCH_REQUIRE_THINK_CLOSE=1 OELLM_PATCH_TMAX_MULTILINGUAL_MATH=1 OELLM_PATCH_DIRECT_MODE_VERIFIERS=1 \
          OELLM_PATCH_TOOL_CALL_VERIFIER=1 OELLM_PATCH_EU_TASK_VERIFIERS=1 "$PY" "$@"; }
case $STAGE in
  sample)
    shard=${SLURM_ARRAY_TASK_ID:-0}
    run -m oellm_rlvr.cli pass-rate-sample --pool "$POOL" --model "$MODEL" --output "$OUT/samples-$shard.jsonl" \
      --samples "$SAMPLES" --shard "$shard" --shards "$SHARDS" --max-new-tokens "${MAX_NEW_TOKENS:-8192}" \
      --max-model-len "${MAX_MODEL_LEN:-12288}" ${LIMIT:+--limit $LIMIT} ;;
  score)
    code_args=()
    if run - "$POOL" <<'PY'
import sys, pyarrow.parquet as pq
sys.exit(0 if any(str(d).startswith("code") for d in pq.read_table(sys.argv[1], columns=["dataset"]).column(0).to_pylist()) else 1)
PY
    then
      # Model-written programs run as in training: contained, read-only backend, private /tmp, loopback only.
      BACKEND=$ROOT/tmax-reproduction; VENV=$(readlink -f "$BACKEND/.venv"); PYHOME=$(dirname "$(dirname "$(readlink -f "$PY")")")
      singularity exec --contain --cleanenv --no-home --pwd /tmp -B "$BACKEND:$BACKEND:ro" -B "$VENV:$VENV:ro" \
        -B "$PYHOME:$PYHOME:ro" "$C" env PYTHONPATH="$BACKEND" "$PY" -m uvicorn open_instruct.code_utils.api:app \
        --host 127.0.0.1 --port 18080 --workers 16 --log-level warning > "$OUT/code_api.log" 2>&1 &
      for i in $(seq 1 60); do curl -sf -o /dev/null http://127.0.0.1:18080/docs && break; sleep 2; done
      code_args=(--code-api-url http://127.0.0.1:18080/test_program)
    fi
    run -m oellm_rlvr.cli pass-rate-score --pool "$POOL" --samples-files "$OUT"/samples-*.jsonl \
      --output "$OUT/scores.jsonl" --threads "${SLURM_CPUS_PER_TASK:-16}" "${code_args[@]}" ;;
  select)
    run -m oellm_rlvr.cli pass-rate-select --pool "$POOL" --scores "$OUT/scores.jsonl" --output "$OUT/filtered.parquet" \
      --samples "$SAMPLES" --model "$MODEL" ${SELECT_ARGS:-} ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
