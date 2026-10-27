#!/bin/bash
#SBATCH --job-name=w3-stage
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
# Stage W2's final checkpoint as the W3 base: hard links to the weights (no copy), the RL mixed template from the W2
# base, and a SOURCE file naming the checkpoint. Then build the W3 pool (scripts/leonardo_w3_data.sh). Run after W2.
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
src=$(ls -d "$ROOT"/outputs/w2/*_checkpoints/step_* | sort -t_ -k2 -V | tail -1)
[[ -f "$src/model.safetensors" || -f "$src/model.safetensors.index.json" ]] || { echo "no W2 checkpoint"; exit 1; }
dst=$ROOT/models/w3-base-mixed
rm -rf "$dst"; mkdir -p "$dst"
for f in "$src"/*; do
  case "$(basename "$f")" in chat_template.jinja) ;; *) ln "$f" "$dst/" ;; esac
done
cp "$ROOT/models/w2-base-mixed/chat_template.jinja" "$dst/chat_template.jinja"
echo "$src" > "$dst/SOURCE"
ls -la "$dst"
MODEL=$dst bash "$ROOT/oellm-rlvr/scripts/leonardo_w3_data.sh"
