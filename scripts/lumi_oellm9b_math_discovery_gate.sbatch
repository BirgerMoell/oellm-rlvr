#!/usr/bin/env bash
# A fail-closed, CPU-only survey gate before spending a two-node RL allocation.
#SBATCH --job-name=oellm-math-survey-gate
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail
CAMPAIGN=/scratch/project_465002530/users/bmoell/oellm-rlvr/runs/oellm9b-math-discovery-20261005
/usr/bin/python3 - "$CAMPAIGN" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
pool = json.loads((root / "pool/manifest.json").read_text())
survey = json.loads((root / "survey/predictions.summary.json").read_text())
annotations = json.loads((root / "survey/difficulty.summary.json").read_text())
metrics = survey["metrics"]
assert pool["combined"]["rows"] == pool["combined"]["semantic_groups"] == 10000
assert metrics["prompts"] == annotations["problems"] == 10000
assert metrics["samples"] == annotations["attempts"] == 80000
assert metrics["samples_per_prompt"] == [8]
assert 0.08 <= metrics["sample_accuracy"] <= 0.80
assert metrics["mixed_reward_prompt_rate"] >= 0.20
assert metrics["length_stop_rate"] <= 0.15
assert metrics["unbalanced_think_tag_rate"] <= 0.02
with (root / "survey/difficulty.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 10000
assert len({r["semantic_group_id"] for r in rows}) == 10000
print(json.dumps({"survey_admitted": True, "metrics": metrics}, indent=2))
PY
