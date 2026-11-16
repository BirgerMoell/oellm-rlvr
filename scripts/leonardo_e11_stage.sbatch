#!/bin/bash
#SBATCH --job-name=e11-stage
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
# Build the E11 pool (scripts/build_e11_pool.py) in the training container. The parent is staged as models/e11-base
# like e10-web-base (hard links to the a07 + 0.7 (E10-web - a07) merge, the RL mixed template, a SOURCE file).
#   sbatch scripts/leonardo_e11_stage.sbatch
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
SEARCH=${SEARCH:-$SCRATCH/oellm-search}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
PY=$ROOT/tmax-reproduction/.venv/bin/python
CODE=$ROOT/oellm-rlvr
[[ -e $ROOT/models/e11-base/SOURCE ]] || { echo "stage models/e11-base first" >&2; exit 1; }
singularity exec -B /leonardo_work,/leonardo_scratch "$C" env PYTHONPATH=$CODE/src "$PY" \
  "$CODE/scripts/build_e11_pool.py" --data "$ROOT/data" --search "$SEARCH" --model "$ROOT/models/e11-base" --out "$ROOT/data/e11"
