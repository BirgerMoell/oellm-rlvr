#!/bin/bash
#SBATCH --job-name=live-relay
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.%x.out
# Live Wikipedia relay (scripts/live_wiki_relay.py) on an internet-enabled serial node, for small live-web pilots.
# Writes relay.url and a 600 relay.token to $RELAY_DIR; the budget (default 100 outbound requests) persists in
# $RELAY_DIR/budget.json across restarts. Rollout nodes reach it through `live_wiki_relay.py forward`
# (search_api.python = $RELAY_DIR/forward.sh).
#   RELAY_DIR=<dir> [BUDGET=100] sbatch scripts/leonardo_live_relay.sbatch
set -euo pipefail
: "${RELAY_DIR:?}"
ROOT=${ROOT:-$SCRATCH/oellm-rl}
PY=${PY:-$SCRATCH/oellm-search/.venv/bin/python}
exec "$PY" -u "$ROOT/oellm-rlvr/scripts/live_wiki_relay.py" relay --dir "$RELAY_DIR" --budget "${BUDGET:-100}"
