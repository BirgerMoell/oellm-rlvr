#!/bin/bash
#SBATCH --job-name=wiki-search-index
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Leads + BM25 index per language for the offline Wikipedia search (oellm_rlvr.wiki_search), CPU work on one node
# (the GPU is idle; boost nodes have the memory). Snapshot downloaded on the login node to $SEARCH/wikipedia
# (wikimedia/wikipedia @ b04c8d1c, 20231101.<lang>). Submit from $SCRATCH/oellm-search.
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
SEARCH=${SEARCH:-$OELLM_SCRATCH/oellm-search}
CODE=$OELLM_SCRATCH/oellm-rl/oellm-rlvr
export OMP_NUM_THREADS=4
printf '%s\n' ${LANGS:-en de fr es it pl nl pt sv fi} | xargs -P "${PARALLEL:-5}" -I{} bash -c \
  "$SEARCH/.venv/bin/python $CODE/scripts/wiki_search.py build --source $SEARCH/wikipedia --root $SEARCH --language {} > $SEARCH/logs/build-{}.log 2>&1 && echo done {} || { echo FAILED {}; tail -5 $SEARCH/logs/build-{}.log; }"
for l in ${LANGS:-en de fr es it pl nl pt sv fi}; do cat "$SEARCH/index/$l/oellm_meta.json" 2>/dev/null || echo "missing $l"; done
"$SEARCH/.venv/bin/python" "$CODE/scripts/wiki_search.py" query --root "$SEARCH" --language sv "Selma Lagerlöf född år"
