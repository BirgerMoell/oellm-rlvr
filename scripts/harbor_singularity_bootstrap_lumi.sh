#!/bin/bash
# LUMI-specific replacement for Harbor v0.22.0's online bootstrap. The LAIF
# image already contains FastAPI/uvicorn and compute nodes have no Internet.
set -euo pipefail

export WORKDIR="${1:-/app}"
shift
export HARBOR_STAGING=/staging/env_files
export HARBOR_PYTHON=/opt/venv/bin/python
export TMUX_TMPDIR="${TMUX_TMPDIR:-/tmp/.harbor-tmux}"
mkdir -p "$WORKDIR" "$TMUX_TMPDIR"

"$HARBOR_PYTHON" -c 'import fastapi, uvicorn'
command -v tmux >/dev/null
if [[ -f "$HARBOR_STAGING/setup.sh" ]]; then
  source "$HARBOR_STAGING/setup.sh"
fi
exec "$HARBOR_PYTHON" "$@"
