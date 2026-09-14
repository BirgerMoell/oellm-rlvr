#!/usr/bin/env bash
#SBATCH --job-name=qwen35-textfix
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err

set -euo pipefail
module purge

CONTROL_ROOT="${CONTROL_ROOT:-/scratch/project_465002530/users/bmoell/oellm-rlvr-eu-qwen-20260914-src}"
SOURCE="${SOURCE:-/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/qwen35-9b-sft-bf16}"
OUTPUT="${OUTPUT:-/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/qwen35-9b-sft-bf16-textfix}"
CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
BIND="/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd"

mkdir -p logs
test -d "$SOURCE"
test ! -e "$OUTPUT"
test -r "$CONTAINER"

singularity exec -B "$BIND" "$CONTAINER" \
  env CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
  PYTHONPATH="$CONTROL_ROOT/src" /opt/venv/bin/python3 \
  "$CONTROL_ROOT/scripts/repair_qwen35_text_checkpoint.py" \
  --source "$SOURCE" --output "$OUTPUT" --max-shard-gib 4

test -r "$OUTPUT/repair-manifest.json"
sha256sum "$OUTPUT"/model*.safetensors "$OUTPUT/config.json" "$OUTPUT/repair-manifest.json"
