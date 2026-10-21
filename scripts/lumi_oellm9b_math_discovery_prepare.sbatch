#!/usr/bin/env bash
# CPU-only: make a frozen, semantic-group-disjoint 10,000-problem survey/train pool.
#SBATCH --job-name=oellm-math-pool10k
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail
module purge
ROOT=/scratch/project_465002530/users/bmoell/oellm-rlvr
CAMPAIGN="$ROOT/runs/oellm9b-math-discovery-20261005"
CONTROL="$CAMPAIGN/control"
CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif
PYTHON=/scratch/project_465002530/users/bmoell/oellm-rlvr-runtime-20260924/venv/bin/python
DATA="$ROOT/data/oellm9b-math-promotion-20260928"
mkdir -p "$CAMPAIGN"
export PYTHONPATH="$CONTROL/src:/opt/venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl "$CONTAINER" \
  env PYTHONPATH="$PYTHONPATH" "$PYTHON" "$CONTROL/scripts/build_eu_math_pools.py" \
  --source "$ROOT/data/sources/oellm-math-rlvr-0ffc9d6c.parquet" \
  --output-dir "$CAMPAIGN/pool" --eu-per-language 128 --english-count 7056 \
  --min-difficulty 1 --max-difficulty 5 --seed 20261005 --source-split train \
  --exclude "$DATA/phase-a/all-math.parquet" \
  --exclude "$DATA/phase-b/all-math.parquet" \
  --exclude "$DATA/eval/all-math.parquet" \
  > "$CAMPAIGN/pool-build-report.json"
singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl "$CONTAINER" \
  env PYTHONPATH="$PYTHONPATH" "$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["combined"]["rows"]==10000; assert m["combined"]["semantic_groups"]==10000; assert m["semantic_group_overlap"]==0; print("validated 10000 unique semantic groups")' \
  "$CAMPAIGN/pool/manifest.json"
