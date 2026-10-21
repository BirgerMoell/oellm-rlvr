#!/usr/bin/env bash
# Frozen-checkpoint survey. Resume-safe: rerun with the same inputs after timeout.
#SBATCH --job-name=oellm-math-survey10k
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=mi250:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00

set -euo pipefail
module purge
ROOT=/scratch/project_465002530/users/bmoell/oellm-rlvr
CAMPAIGN="$ROOT/runs/oellm9b-math-discovery-20261005"
CONTROL="$CAMPAIGN/control"
CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
PYTHON=/scratch/project_465002530/users/bmoell/oellm-rlvr-runtime-20260924/venv/bin/python
MODEL="$ROOT/outputs/grpo-math-oellm9b-phaseb-devg-64step-mixfix/lumi_grpo_math_oellm9b_phaseb_devg_64step_mixfix__42__1790936468_checkpoints/step_64"
test -f "$MODEL/.checkpoint_complete"
test -f "$CAMPAIGN/pool/manifest.json"
mkdir -p "$CAMPAIGN/survey"
export PYTHONPATH="$CONTROL/src:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OELLM_PATCH_MATH_EQUIV_THREADS=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false HWLOC_COMPONENTS=-rsmi
export XDG_CACHE_HOME="/tmp/oellm-math-survey-${SLURM_JOB_ID}/xdg"
export TRITON_CACHE_DIR="/tmp/oellm-math-survey-${SLURM_JOB_ID}/triton"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
  unset ROCR_VISIBLE_DEVICES
fi
singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl "$CONTAINER" \
  env PYTHONPATH="$PYTHONPATH" "$PYTHON" -m oellm_rlvr.cli eval-reasoning \
  --model "$MODEL" --dataset "$CAMPAIGN/pool/all-math.parquet" \
  --output "$CAMPAIGN/survey/predictions.jsonl" \
  --samples-per-prompt 8 --temperature 1.0 --max-new-tokens 2048 \
  --max-model-len 3072 --batch-size 64 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.65 --seed 20261005
singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl "$CONTAINER" \
  env PYTHONPATH="$PYTHONPATH" "$PYTHON" "$CONTROL/scripts/annotate_math_difficulty.py" \
  --predictions "$CAMPAIGN/survey/predictions.jsonl" \
  --output-csv "$CAMPAIGN/survey/difficulty.csv" --expected-samples 8
