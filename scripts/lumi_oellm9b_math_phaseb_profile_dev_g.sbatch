#!/usr/bin/env bash
# Profile the exact Phase A step-32 parent against the disjoint Phase B components.
# Submit from any directory; all outputs are scoped to the dated campaign root.
#SBATCH --job-name=oellm9b-math-b-profile
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
MODEL="$ROOT/outputs/grpo-math-oellm9b-promotion-phase-a-32step-20260929/grpo_math_oellm9b_promotion_phase_a_32step_20260929__42__1790783063_checkpoints/step_32"

test -f "$MODEL/.checkpoint_complete"
test "$(git -C "$CONTROL" rev-parse HEAD)" = f22ba2d18e1763474b8bd00d280c6fe135d8eaa6
mkdir -p "$CAMPAIGN/profile"
export PYTHONPATH="$CONTROL/src:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OELLM_PATCH_MATH_EQUIV_THREADS=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export HWLOC_COMPONENTS=-rsmi
export XDG_CACHE_HOME="/tmp/oellm9b-math-profile-${SLURM_JOB_ID}/xdg"
export TRITON_CACHE_DIR="/tmp/oellm9b-math-profile-${SLURM_JOB_ID}/triton"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
  unset ROCR_VISIBLE_DEVICES
fi

profile_one() {
  local name="$1" dataset="$2" seed="$3"
  local output="$CAMPAIGN/profile/$name"
  mkdir -p "$output"
  singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl "$CONTAINER" \
    "$PYTHON" -m oellm_rlvr.cli eval-reasoning \
      --model "$MODEL" --dataset "$dataset" --output "$output/predictions.jsonl" \
      --limit 128 --limit-seed "$seed" --samples-per-prompt 8 \
      --temperature 1.0 --max-new-tokens 2048 --max-model-len 3072 \
      --batch-size 64 --tensor-parallel-size 1 --gpu-memory-utilization 0.65 \
      --seed "$seed"
}

profile_one eu-hard "$ROOT/data/oellm9b-math-promotion-20260928/phase-b/eu-math.parquet" 20261002
profile_one english-hard "$ROOT/data/oellm9b-math-promotion-20260928/phase-b/english-replay.parquet" 20261003
profile_one easy-replay "$ROOT/data/oellm9b-math-promotion-20260928/phase-a/all-math.parquet" 20261004
