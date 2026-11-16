#!/bin/bash
# Install the Harbor build that provides the `apptainer` environment (bridge server + per-node workers + SIF
# prebuild) used by the LAION "OT validated RL datasets". Upstream harbor-framework/harbor has no such environment;
# marianna13/harbor does (based on Harbor 0.22.0). Run on a LOGIN node (network needed only here).
#   source leonardo.env && scripts/leonardo_harbor_apptainer_bootstrap.sh
#
# Env overrides: HARBOR_LAION_ROOT (checkout), HARBOR_LAION_VENV, PYTHON (>=3.12, e.g. a module or `uv` python).
set -euo pipefail

: "${OELLM_SCRATCH:?source leonardo.env first}"
: "${HARBOR_LAION_ROOT:=$OELLM_SCRATCH/oellm-rl/src/harbor-laion}"
: "${HARBOR_LAION_VENV:=$OELLM_SCRATCH/oellm-rl/venvs/harbor-laion}"
: "${PYTHON:=}"

# Leonardo only ships Python 3.11 as a module; Harbor needs >= 3.12. Use uv (managed Python) when PYTHON is unset.
if [[ -z "$PYTHON" ]]; then
  command -v uv >/dev/null 2>&1 || { echo "set PYTHON=<python>=3.12> or install uv" >&2; exit 1; }
  [[ -x "$HARBOR_LAION_VENV/bin/python" ]] || { mkdir -p "$(dirname "$HARBOR_LAION_VENV")"; uv venv --python 3.12 "$HARBOR_LAION_VENV"; }
  PYTHON="$HARBOR_LAION_VENV/bin/python"
fi
HARBOR_URL=https://github.com/marianna13/harbor.git
HARBOR_COMMIT=82efa984407740f695e1bca60542cac00e2497b5   # main, 2026-09-25 "Harden offline Apptainer rollouts and terminal recovery"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), sys.version' \
  || { echo "Harbor needs Python >= 3.12; set PYTHON=/path/to/python3.12" >&2; exit 1; }

if [[ ! -d "$HARBOR_LAION_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$HARBOR_LAION_ROOT")"
  git clone --filter=blob:none "$HARBOR_URL" "$HARBOR_LAION_ROOT"
fi
git -C "$HARBOR_LAION_ROOT" fetch origin "$HARBOR_COMMIT"
git -C "$HARBOR_LAION_ROOT" checkout --detach "$HARBOR_COMMIT"
[[ "$(git -C "$HARBOR_LAION_ROOT" rev-parse HEAD)" == "$HARBOR_COMMIT" ]]

mkdir -p "$(dirname "$HARBOR_LAION_VENV")"
[[ -x "$HARBOR_LAION_VENV/bin/python" ]] || "$PYTHON" -m venv "$HARBOR_LAION_VENV"
if "$HARBOR_LAION_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  "$HARBOR_LAION_VENV/bin/python" -m pip install --upgrade pip
  "$HARBOR_LAION_VENV/bin/python" -m pip install -e "$HARBOR_LAION_ROOT"
else
  uv pip install --python "$HARBOR_LAION_VENV/bin/python" -e "$HARBOR_LAION_ROOT"   # uv venvs have no pip
fi
# The control plane's result parser is stdlib-only; the oellm_rlvr package is used via PYTHONPATH by the sbatch files.

"$HARBOR_LAION_VENV/bin/python" - <<'PY'
import harbor.environments.apptainer.prebuild_sifs as p, harbor.environments.apptainer.harbor_env as e
print("apptainer environment importable:", p.__name__, e.__name__)
PY

echo "--- container runtime on this node (builds and runs need one of these; compute nodes may differ):"
for b in apptainer singularity; do command -v "$b" >/dev/null 2>&1 && "$b" --version || echo "no $b on PATH (try: module avail apptainer singularity)"; done
echo "harbor_root=$HARBOR_LAION_ROOT"
echo "venv=$HARBOR_LAION_VENV"
