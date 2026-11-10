#!/bin/bash
# Build x86_64 SIFs for one LAION dataset (the published SIFs are aarch64 and unusable on Leonardo).
# Needs registry access, so run it on an internet-enabled partition (Leonardo: lrd_all_serial, the login-type
# partition). Dedupes by build-context hash; the cache is shared across datasets and re-runs skip current images.
#
#   source leonardo.env
#   DATASET=rts-1k-rootless-apptainer sbatch scripts/leonardo_laion_sif_build.sbatch
#   DATASET=rts-1k-rootless-apptainer DRY_RUN=1 bash scripts/leonardo_laion_sif_build.sbatch   # plan only, no sbatch
#
# Override the partition/account on the command line if needed: sbatch --partition=... --account=...
#SBATCH --job-name=laion-sif-build
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.%x.out
#SBATCH --error=logs/%j.%x.err
set -euo pipefail

: "${OELLM_SCRATCH:?source leonardo.env first}"
: "${DATASET:?set DATASET=<repo name without laion/>}"
: "${LAION_ROOT:=$OELLM_SCRATCH/oellm-rl/data/laion-ot}"
: "${HARBOR_LAION_ROOT:=$OELLM_SCRATCH/oellm-rl/src/harbor-laion}"
: "${HARBOR_LAION_VENV:=$OELLM_SCRATCH/oellm-rl/venvs/harbor-laion}"
: "${SIF_CACHE:=$OELLM_SCRATCH/oellm-rl/sif-cache-x86_64}"
: "${SIF_BUILD_WORKERS:=3}"
: "${SIF_BUILD_TIMEOUT_SEC:=1800}"
: "${APPTAINER_CACHEDIR:=$OELLM_SCRATCH/oellm-rl/apptainer-cache}"
: "${APPTAINER_TMPDIR:=$OELLM_SCRATCH/oellm-rl/apptainer-tmp/${SLURM_JOB_ID:-manual}}"
: "${DRY_RUN:=0}"
: "${WHEEL_DIR:=}"            # optional flat wheelhouse for offline verifier deps (see the Harbor apptainer README)
: "${BASE_IMAGE_OVERRIDES:=}" # space-separated FROM=TO pairs, e.g. "debian:bullseye-slim=debian:bookworm-slim"

TASKS="$LAION_ROOT/$DATASET/tasks"
[[ -d "$TASKS" ]] || { echo "no extracted tasks at $TASKS (run leonardo_laion_download.sh)" >&2; exit 1; }
export APPTAINER_CACHEDIR APPTAINER_TMPDIR PYTHONUNBUFFERED=1
mkdir -p "$SIF_CACHE" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" logs

[[ "$(uname -m)" == "x86_64" ]] || { echo "refusing to build on $(uname -m); Leonardo SIFs must be x86_64" >&2; exit 1; }
command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1 \
  || { echo "no apptainer/singularity on PATH (module load?)" >&2; exit 1; }

args=("$TASKS" --sif-cache "$SIF_CACHE" --max-workers "$SIF_BUILD_WORKERS" --build-timeout-sec "$SIF_BUILD_TIMEOUT_SEC")
for o in $BASE_IMAGE_OVERRIDES; do args+=(--base-image-override "$o"); done
[[ -n "$WHEEL_DIR" ]] && args+=(--wheel-dir "$WHEEL_DIR")
[[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)

echo "node=$(hostname) arch=$(uname -m) dataset=$DATASET tasks=$TASKS cache=$SIF_CACHE"
cd "$HARBOR_LAION_ROOT"
PYTHONPATH="$HARBOR_LAION_ROOT/src" "$HARBOR_LAION_VENV/bin/python" -m harbor.environments.apptainer.prebuild_sifs "${args[@]}"
echo "sifs in cache: $(find "$SIF_CACHE" -maxdepth 1 -name '*.sif' | wc -l)"; du -sh "$SIF_CACHE"
