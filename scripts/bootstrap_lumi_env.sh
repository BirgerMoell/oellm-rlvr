#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="${1:?usage: bootstrap_lumi_env.sh CONTROL_ROOT BACKEND_ROOT VENV}"
BACKEND_ROOT="${2:?usage: bootstrap_lumi_env.sh CONTROL_ROOT BACKEND_ROOT VENV}"
VENV="${3:?usage: bootstrap_lumi_env.sh CONTROL_ROOT BACKEND_ROOT VENV}"
BASE_IMAGE="${BASE_IMAGE:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
BIND="/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd"
EXPECTED_COMMIT="3f80d37042402b8363f39c9535723b0d4cb8de54"

test -r "$BASE_IMAGE"
test -d "$CONTROL_ROOT/.git"
test -d "$BACKEND_ROOT/.git"
actual="$(git -C "$BACKEND_ROOT" rev-parse HEAD)"
[[ "$actual" == "$EXPECTED_COMMIT" ]] || {
  echo "Expected TMAX backend $EXPECTED_COMMIT, got $actual" >&2
  exit 2
}

mkdir -p "$(dirname "$VENV")"
singularity exec -B "$BIND" "$BASE_IMAGE" python -m venv --system-site-packages "$VENV"

run_python() {
  singularity exec -B "$BIND" "$BASE_IMAGE" "$VENV/bin/python" "$@"
}

run_python -m pip install --upgrade pip setuptools wheel
# Do not `uv sync` the TMAX checkout on ROCm: its upstream resolver contains
# CUDA-specific torch/flash-attention sources. The LUMI image owns torch/vLLM.
run_python -m pip install \
  'ray[default]==2.54.0' \
  'liger-kernel==0.8.0' \
  'flash-linear-attention>=0.4.2' \
  'openenv-core>=0.2.1' \
  'mcp>=1.9.0' \
  'docker>=7.0.0' \
  'immutabledict==1.2.0' \
  'antlr4-python3-runtime==4.11'
run_python -m pip install --no-deps -e "$BACKEND_ROOT"
run_python -m pip install -e "$CONTROL_ROOT[data,math]"

run_python - <<'PY'
import importlib
for name in ("torch", "vllm", "ray", "open_instruct", "liger_kernel", "fla"):
    importlib.import_module(name)
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
print("LUMI RLVR environment imports are healthy")
PY
