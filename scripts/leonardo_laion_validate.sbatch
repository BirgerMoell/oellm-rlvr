#!/bin/bash
# Re-validate a LAION dataset on Leonardo x86_64: oracle agent must pass and the nop agent must fail, each ATTEMPTS
# times per task, through Harbor's local Apptainer bridge (server + worker on this node, offline, CPU only).
# Writes $RESULTS/<job>/leonardo-compatible.jsonl = the tasks that qualify for RL here.
#
#   source leonardo.env
#   DATASET=rts-1k-rootless-apptainer SUBSET=overfit sbatch scripts/leonardo_laion_validate.sbatch   # 100-task pilot
#   DATASET=rts-1k-rootless-apptainer SUBSET=all     sbatch scripts/leonardo_laion_validate.sbatch
# SUBSET: overfit | diagnostic (manifests shipped with the dataset, 100 tasks each) | all | list (TASK_LIST=<file, one task id per line>).
# Partition: boost_usr_prod (this account has no dcgp_usr_prod budget); the sandboxes only use CPU. Override with sbatch --partition=.
#SBATCH --job-name=laion-validate
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
set -euo pipefail

: "${OELLM_SCRATCH:?source leonardo.env first}"
: "${DATASET:?set DATASET=<repo name without laion/>}"
: "${SUBSET:=overfit}"
: "${ATTEMPTS:=3}"
: "${TASK_LIST:=}"
: "${CONTROL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${LAION_ROOT:=$OELLM_SCRATCH/oellm-rl/data/laion-ot}"
: "${HARBOR_LAION_ROOT:=$OELLM_SCRATCH/oellm-rl/src/harbor-laion}"
: "${HARBOR_LAION_VENV:=$OELLM_SCRATCH/oellm-rl/venvs/harbor-laion}"
: "${SIF_CACHE:=$OELLM_SCRATCH/oellm-rl/sif-cache-x86_64}"
: "${RESULTS_ROOT:=$OELLM_SCRATCH/oellm-rl/laion-validation}"
: "${WORKERS_PER_NODE:=16}"          # sandboxes are CPU-bound; ~16-24 per 32 cores is a safe start
: "${BRIDGE_PORT:=19910}"
: "${BRIDGE_OVERLAY_SIZE_MB:=1024}"
RUN_ID="${SLURM_JOB_ID:-$$}"
OUT="$RESULTS_ROOT/$DATASET-$SUBSET-$RUN_ID"
STAGING_BASE="${TMPDIR:-/tmp}/harbor-apptainer-$RUN_ID"
export BRIDGE_AUTH_TOKEN="$("$HARBOR_LAION_VENV/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))')"
export BRIDGE_HOST=127.0.0.1 BRIDGE_PORT WORKERS_PER_NODE SIF_CACHE STAGING_BASE BRIDGE_OVERLAY_SIZE_MB
export BRIDGE_URL="http://127.0.0.1:$BRIDGE_PORT"
export APPTAINER_CONFIGDIR="$OUT/apptainer-config" PYTHONUNBUFFERED=1
export PYTHONPATH="$HARBOR_LAION_ROOT/src:$CONTROL_ROOT/src"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="$HARBOR_LAION_VENV/bin/python"

SRC="$LAION_ROOT/$DATASET"
[[ -d "$SRC/tasks" ]] || { echo "no tasks at $SRC/tasks" >&2; exit 1; }
mkdir -p "$OUT/pack" "$OUT/jobs" "$OUT/service-logs" "$STAGING_BASE" "$APPTAINER_CONFIGDIR" logs

# Task pack: symlink view of the chosen subset (the shipped manifests list {"task_id": ...} per line).
"$PY" - "$SRC" "$SUBSET" "$OUT/pack" <<'PY'
import json, os, sys
src, subset, pack = sys.argv[1:]
tasks = os.path.join(src, "tasks")
if subset == "all":
    ids = sorted(d for d in os.listdir(tasks) if os.path.isdir(os.path.join(tasks, d)))
elif subset == "list":
    ids = [l.strip() for l in open(os.environ["TASK_LIST"]) if l.strip()]
else:
    name = {"overfit": "overfit-train", "diagnostic": "diagnostic-validation"}[subset]
    ids = [json.loads(l)["task_id"] for l in open(os.path.join(src, "manifests", name + ".jsonl")) if l.strip()]
for t in ids:
    os.symlink(os.path.join(tasks, t), os.path.join(pack, t))
print(f"pack: {len(ids)} tasks")
PY
N_TASKS=$(find "$OUT/pack" -mindepth 1 -maxdepth 1 | wc -l)

server_pid=""; worker_pid=""
cleanup() { trap - EXIT INT TERM; [[ -n "$worker_pid" ]] && kill "$worker_pid" 2>/dev/null || true
            [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

APPT="$HARBOR_LAION_ROOT/src/harbor/environments/apptainer"
PYTHON_BIN="$PY" "$APPT/start_server.sh" >"$OUT/service-logs/server.log" 2>&1 & server_pid=$!
for _ in $(seq 1 30); do
  curl -fsS -H "Authorization: Bearer $BRIDGE_AUTH_TOKEN" "$BRIDGE_URL/status" >/dev/null 2>&1 && break; sleep 1
done
curl -fsS -H "Authorization: Bearer $BRIDGE_AUTH_TOKEN" "$BRIDGE_URL/status" >/dev/null \
  || { echo "bridge did not start; see $OUT/service-logs/server.log" >&2; exit 1; }
PYTHON_BIN="$PY" "$APPT/start_worker.sh" >"$OUT/service-logs/worker.log" 2>&1 & worker_pid=$!
sleep 3

run_arm() {  # agent
  "$PY" -m harbor.cli.main run --path "$OUT/pack" --agent "$1" --env apptainer \
    --environment-kwarg "bridge_url=$BRIDGE_URL" --environment-kwarg "sif_cache=$SIF_CACHE" \
    --jobs-dir "$OUT/jobs" --job-name "$1" --n-concurrent "$WORKERS_PER_NODE" -k "$ATTEMPTS" --yes
}
echo "dataset=$DATASET subset=$SUBSET tasks=$N_TASKS attempts=$ATTEMPTS out=$OUT"
# A failing arm must not abort the job: per-task results are summarised below either way.
run_arm oracle || echo "[warn] oracle arm exited non-zero"
run_arm nop    || echo "[warn] nop arm exited non-zero"

"$PY" "$CONTROL_ROOT/scripts/summarize_laion_validation.py" \
  --jobs "$OUT/jobs" --tasks-dir "$OUT/pack" --attempts "$ATTEMPTS" \
  --compatible-out "$OUT/leonardo-compatible.jsonl" --report-out "$OUT/validation-report.json"
echo "validation_artifact=$OUT"
