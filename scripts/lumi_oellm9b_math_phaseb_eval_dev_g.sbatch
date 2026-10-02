#!/usr/bin/env bash
# Submit with --dependency=afterok:<training-job>; fail closed if exports are incomplete.
#SBATCH --job-name=oellm9b-math-b-eval
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=mi250:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00

set -euo pipefail
module purge

ROOT=/scratch/project_465002530/users/bmoell/oellm-rlvr
CAMPAIGN="$ROOT/runs/oellm9b-math-phaseb-devg-20261002"
CONTROL=/scratch/project_465002530/users/bmoell/oellm-rlvr-runtime-20260928-promotion/control
CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
PYTHON=/scratch/project_465002530/users/bmoell/oellm-rlvr-runtime-20260924/venv/bin/python
EXPORT_ROOT="$ROOT/outputs/grpo-math-oellm9b-phaseb-devg-64step-mixfix"
BASELINE="$ROOT/evals/oellm9b-math-promotion-20260928"

test "$(git -C "$CONTROL" rev-parse HEAD)" = f22ba2d18e1763474b8bd00d280c6fe135d8eaa6
mkdir -p "$CAMPAIGN/eval"
export PYTHONPATH="$CONTROL/src:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OELLM_PATCH_MATH_EQUIV_THREADS=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false HWLOC_COMPONENTS=-rsmi
export XDG_CACHE_HOME="/tmp/oellm9b-math-eval-${SLURM_JOB_ID}/xdg"
export TRITON_CACHE_DIR="/tmp/oellm9b-math-eval-${SLURM_JOB_ID}/triton"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
  unset ROCR_VISIBLE_DEVICES
fi

read_model() {
  local step="$1"
  mapfile -t matches < <(find "$EXPORT_ROOT" -mindepth 2 -maxdepth 2 -type d -name "step_${step}" | sort)
  if [[ "${#matches[@]}" -ne 1 || ! -f "${matches[0]}/.checkpoint_complete" ]]; then
    echo "Expected one complete step_${step} export; found ${#matches[@]}" >&2
    exit 2
  fi
  printf '%s\n' "${matches[0]}"
}

evaluate() {
  local step="$1" label="$2" dataset="$3" baseline="$4"
  local model="$5" output="$CAMPAIGN/eval/step-${step}-${label}"
  mkdir -p "$output"
  singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl "$CONTAINER" \
    "$PYTHON" -m oellm_rlvr.cli eval-reasoning \
      --model "$model" --dataset "$dataset" --output "$output/predictions.jsonl" \
      --samples-per-prompt 1 --temperature 0 --max-new-tokens 1024 \
      --max-model-len 3072 --batch-size 64 --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.65 --seed 20260930
  singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl "$CONTAINER" \
    "$PYTHON" -m oellm_rlvr.cli compare-reasoning \
      --baseline "$baseline" --candidate "$output/predictions.jsonl" \
      --bootstrap-samples 5000 --seed 20260930 \
      --output "$output/paired-vs-phase-a.json"
}

STEP32="$(read_model 32)"
STEP64="$(read_model 64)"
for step in 32 64; do
  model="$STEP32"
  [[ "$step" == 64 ]] && model="$STEP64"
  evaluate "$step" multilingual \
    "$ROOT/data/oellm9b-math-promotion-20260928/eval/all-math.parquet" \
    "$BASELINE/phase-a-step32-multilingual/predictions.jsonl" "$model"
  evaluate "$step" dapo \
    "$ROOT/data/dapo-math-17k-en-31dd3095-sanitized-v2/evaluation.parquet" \
    "$BASELINE/phase-a-step32-dapo/predictions.jsonl" "$model"
done
