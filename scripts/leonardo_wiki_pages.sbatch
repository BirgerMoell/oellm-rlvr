#!/bin/bash
#SBATCH --job-name=wiki-pages
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Full article text for the open_page / find_in_page tools (oellm_rlvr.wiki_search.build_pages), row-aligned with the
# leads that scripts/leonardo_wiki_search_index.sbatch built. I/O-bound, no GPU: the serial partition is enough.
# Submit from $SCRATCH/oellm-search.
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
SEARCH=${SEARCH:-$OELLM_SCRATCH/oellm-search}
CODE=$OELLM_SCRATCH/oellm-rl/oellm-rlvr
printf '%s\n' ${LANGS:-en de fr es it pl nl pt sv fi} | xargs -P "${PARALLEL:-4}" -I{} bash -c \
  "$SEARCH/.venv/bin/python $CODE/scripts/wiki_search.py pages --source $SEARCH/wikipedia --root $SEARCH --language {} > $SEARCH/logs/pages-{}.log 2>&1 && echo done {} \$(tail -1 $SEARCH/logs/pages-{}.log) || { echo FAILED {}; tail -5 $SEARCH/logs/pages-{}.log; }"
du -sh "$SEARCH/pages"
"$SEARCH/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$CODE/src')
from oellm_rlvr import wiki_search as W
s = W.Searcher('$SEARCH')
print(W.format_page(s.open('Selma Lagerlöf', 'sv', 2))[:600])
print(W.format_find(s.find('Selma Lagerlöf', 'Nobelpriset', 'sv'))[:600])"
