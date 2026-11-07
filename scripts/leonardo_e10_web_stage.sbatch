#!/bin/bash
#SBATCH --job-name=e10-stage
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
# Stage the E10-web parent as models/e10-web-base: hard links to the weights, the RL mixed template from the W3 base
# and a SOURCE file. Then build the E10-web pool (scripts/build_e10_web_pool.py) in the training container.
#   PARENT=<model dir> sbatch scripts/leonardo_e10_web_stage.sbatch
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
SEARCH=${SEARCH:-$SCRATCH/oellm-search}
: "${PARENT:?}"
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
PY=$ROOT/tmax-reproduction/.venv/bin/python
CODE=$ROOT/oellm-rlvr
dst=$ROOT/models/e10-web-base
[[ ! -e $dst ]] || { echo "$dst exists; move it aside first" >&2; exit 1; }
mkdir -p "$dst"
for f in "$PARENT"/*; do
  case "$(basename "$f")" in chat_template.jinja) ;; *) ln "$f" "$dst/" ;; esac
done
cp "$ROOT/models/w3-base-mixed/chat_template.jinja" "$dst/chat_template.jinja"
echo "$PARENT" > "$dst/SOURCE"
ls -la "$dst"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" env PYTHONPATH=$CODE/src "$PY" \
  "$CODE/scripts/build_e10_web_pool.py" --data "$ROOT/data" --search "$SEARCH" --model "$dst" --out "$ROOT/data/e10-web"
