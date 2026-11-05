#!/bin/bash
#SBATCH --job-name=deep-search-build
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# deep_search suite (scripts/build_deep_search.py) and reference browsing trajectories
# (scripts/deep_search_trajectories.py) against the offline Wikipedia page store. CPU only.
#   RAW=<fetch_deep_search_wikidata.py output> [OUT=$OELLM_SCRATCH/oellm-rl/data/deep-search] sbatch ...
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
SEARCH=${SEARCH:-$OELLM_SCRATCH/oellm-search}
CODE=$OELLM_SCRATCH/oellm-rl/oellm-rlvr
OUT=${OUT:-$OELLM_SCRATCH/oellm-rl/data/deep-search}
: "${RAW:?}"
PY=$SEARCH/.venv/bin/python
"$PY" "$CODE/scripts/build_deep_search.py" --raw "$RAW" --search "$SEARCH" --out "$OUT"
"$PY" "$CODE/scripts/deep_search_trajectories.py" --raw "$RAW" --search "$SEARCH" --data "$OUT" \
  --out "$OUT/trajectories.jsonl" --limit "${TRAJ_LIMIT:-4000}"
sha256sum "$OUT"/*.jsonl "$OUT"/*.parquet
