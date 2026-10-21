#!/usr/bin/env bash
# Frozen paired checks against the *actual* Phase B step-64 parent.
#SBATCH --job-name=oellm-math-discovery-eval
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=mi250:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail
module purge
ROOT=/scratch/project_465002530/users/bmoell/oellm-rlvr
CAMPAIGN="$ROOT/runs/oellm9b-math-discovery-20261005"
CONTROL="$CAMPAIGN/control"
CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
PYTHON=/scratch/project_465002530/users/bmoell/oellm-rlvr-runtime-20260924/venv/bin/python
PARENT="$ROOT/runs/oellm9b-math-phaseb-devg-20261002/eval"
test -f "$CAMPAIGN/pool/manifest.json"
mkdir -p "$CAMPAIGN/eval"
export PYTHONPATH="$CONTROL/src:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OELLM_PATCH_MATH_EQUIV_THREADS=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false HWLOC_COMPONENTS=-rsmi
export XDG_CACHE_HOME="/tmp/oellm-math-eval-${SLURM_JOB_ID}/xdg"
export TRITON_CACHE_DIR="/tmp/oellm-math-eval-${SLURM_JOB_ID}/triton"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
  unset ROCR_VISIBLE_DEVICES
fi

mapfile -t STEPS < <(find "$CAMPAIGN/training" -mindepth 2 -maxdepth 2 -type d -name step_256 | sort)
if [[ "${#STEPS[@]}" -ne 1 || ! -f "${STEPS[0]}/.checkpoint_complete" ]]; then
  echo "Expected one complete step-256 export; found ${#STEPS[@]}" >&2
  exit 2
fi
MODEL="${STEPS[0]}"
evaluate() {
  local label="$1" dataset="$2" baseline="$3" output="$CAMPAIGN/eval/step-256-$1"
  test -s "$baseline"
  mkdir -p "$output"
  singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl "$CONTAINER" \
    env PYTHONPATH="$PYTHONPATH" "$PYTHON" -m oellm_rlvr.cli eval-reasoning \
    --model "$MODEL" --dataset "$dataset" --output "$output/predictions.jsonl" \
    --samples-per-prompt 1 --temperature 0 --max-new-tokens 1024 \
    --max-model-len 3072 --batch-size 64 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.65 --seed 20260930
  singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl "$CONTAINER" \
    env PYTHONPATH="$PYTHONPATH" "$PYTHON" -m oellm_rlvr.cli compare-reasoning \
    --baseline "$baseline" --candidate "$output/predictions.jsonl" \
    --bootstrap-samples 5000 --seed 20260930 \
    --output "$output/paired-vs-phase-b-step64.json"
}
evaluate multilingual \
  "$ROOT/data/oellm9b-math-promotion-20260928/eval/all-math.parquet" \
  "$PARENT/step-64-multilingual/predictions.jsonl"
evaluate dapo \
  "$ROOT/data/dapo-math-17k-en-31dd3095-sanitized-v2/evaluation.parquet" \
  "$PARENT/step-64-dapo/predictions.jsonl"
