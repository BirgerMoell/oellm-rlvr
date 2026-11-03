#!/bin/bash
#SBATCH --job-name=e8-stage
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
# Stage the E8 parent (linear merge 0.5 x W2 + 0.5 x Stage 4b) as models/e8-base-mixed: hard links to the weights,
# the RL mixed template from the W3 base and a SOURCE file. Then build the E8 pool (scripts/build_e8_pool.py):
# made-up entities checked against the Wikipedia titles (search venv), then the pool (training container).
#   PARENT=<merge dir> W2_SAMPLES=<hallu-pilot-w2-v2 dir> EXCLUDE=<hallu-v1 eval.jsonl> sbatch scripts/leonardo_e8_stage.sbatch
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
SEARCH=${SEARCH:-$SCRATCH/oellm-search}
: "${PARENT:?}" "${W2_SAMPLES:?}" "${EXCLUDE:?}"
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
PY=$ROOT/tmax-reproduction/.venv/bin/python
CODE=$ROOT/oellm-rlvr
dst=$ROOT/models/e8-base-mixed
rm -rf "$dst"; mkdir -p "$dst"
for f in "$PARENT"/*; do
  case "$(basename "$f")" in chat_template.jinja) ;; *) ln "$f" "$dst/" ;; esac
done
cp "$ROOT/models/w3-base-mixed/chat_template.jinja" "$dst/chat_template.jinja"
echo "$PARENT" > "$dst/SOURCE"
ls -la "$dst"
out=$ROOT/data/e8
mkdir -p "$out"
"$SEARCH/.venv/bin/python" "$CODE/scripts/build_e8_pool.py" fabricated --root "$SEARCH" --out "$out"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" "$PY" "$CODE/scripts/build_e8_pool.py" pool \
  --data "$ROOT/data" --w2 "$W2_SAMPLES" --fabricated "$out" --model "$dst" --out "$out" --exclude "$EXCLUDE"
