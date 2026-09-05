#!/usr/bin/env bash
# One-checkpoint, resumable GSM8K evaluation. Submit once before and once
# after RL with identical environment variables except MODEL and TAG.
#SBATCH --job-name=oellm-reasoning-eval
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail
module purge

: "${MODEL:?Set MODEL to an immutable local Hugging Face checkpoint}"
: "${TAG:?Set an immutable run label}"
: "${DATASET:=/scratch/project_465002530/users/bmoell/oellm-rlvr/data/gsm8k-main-740312ad/test.parquet}"
: "${RESULTS_ROOT:=/scratch/project_465002530/users/bmoell/oellm-rlvr/evals}"
: "${CONTROL_ROOT:=$SLURM_SUBMIT_DIR}"
: "${TOKENIZER:=$MODEL}"
: "${CONTAINER:=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
: "${PYTHON_BIN:=/scratch/project_465002530/users/bmoell/venvs/oellm-rlvr/bin/python}"
: "${HF_HOME:=/scratch/project_465002530/users/bmoell/hf_cache}"
: "${LIMIT:=}"
: "${LIMIT_SEED:=20260905}"
: "${SAMPLES_PER_PROMPT:=1}"
: "${TEMPERATURE:=0}"
: "${TOP_P:=1}"
: "${MAX_NEW_TOKENS:=1024}"
: "${MAX_MODEL_LEN:=3072}"
: "${BATCH_SIZE:=64}"
: "${SEED:=20260905}"

OUTPUT="$RESULTS_ROOT/$TAG/predictions.jsonl"
JOB_TMPDIR="/tmp/oellm-reasoning-eval-$SLURM_JOB_ID"
VENV_SITE="$(dirname "$PYTHON_BIN")/../lib/python3.12/site-packages"
PYTHONPATH="$CONTROL_ROOT/src:$VENV_SITE:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
mkdir -p logs "$(dirname "$OUTPUT")"
test -r "$MODEL/config.json"
test -r "$TOKENIZER/tokenizer_config.json"
test -r "$DATASET"
test -r "$CONTAINER"
test -r "$CONTROL_ROOT/src/oellm_rlvr/reasoning_eval.py"
export MODEL TAG DATASET RESULTS_ROOT CONTROL_ROOT TOKENIZER CONTAINER PYTHON_BIN HF_HOME
export LIMIT LIMIT_SEED SAMPLES_PER_PROMPT TEMPERATURE TOP_P MAX_NEW_TOKENS MAX_MODEL_LEN BATCH_SIZE SEED
export OUTPUT JOB_TMPDIR PYTHONPATH

echo "model=$MODEL tokenizer=$TOKENIZER dataset=$DATASET tag=$TAG output=$OUTPUT"
srun --label --gpu-bind=none --gpus-per-task=1 singularity exec \
  -B /pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd \
  "$CONTAINER" env PYTHONPATH="$PYTHONPATH" HF_HOME="$HF_HOME" bash -c '
    set -euo pipefail
    if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
      export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
      unset ROCR_VISIBLE_DEVICES
    fi
    mkdir -p "$JOB_TMPDIR"/{tmp,xdg,triton,torchinductor,miopen}
    export TMPDIR="$JOB_TMPDIR/tmp"
    export XDG_CACHE_HOME="$JOB_TMPDIR/xdg"
    export TRITON_CACHE_DIR="$JOB_TMPDIR/triton"
    export TORCHINDUCTOR_CACHE_DIR="$JOB_TMPDIR/torchinductor"
    export MIOPEN_USER_DB_PATH="$JOB_TMPDIR/miopen"
    export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
    export MIOPEN_FIND_MODE=FAST
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1 VLLM_DISABLE_COMPILE_CACHE=1
    export PYTHONPATH HF_HOME
    limit_args=()
    if [[ -n "$LIMIT" ]]; then limit_args=(--limit "$LIMIT"); fi
    env PYTHONPATH="$PYTHONPATH" "$PYTHON_BIN" -m oellm_rlvr.cli eval-reasoning \
      --model "$MODEL" --tokenizer "$TOKENIZER" --dataset "$DATASET" --output "$OUTPUT" \
      --limit-seed "$LIMIT_SEED" --samples-per-prompt "$SAMPLES_PER_PROMPT" \
      --temperature "$TEMPERATURE" --top-p "$TOP_P" --max-new-tokens "$MAX_NEW_TOKENS" \
      --max-model-len "$MAX_MODEL_LEN" --batch-size "$BATCH_SIZE" --seed "$SEED" \
      "${limit_args[@]}"
  '

echo "completed=$OUTPUT summary=${OUTPUT%.jsonl}.summary.json"
